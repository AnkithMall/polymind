"""Pipeline orchestrator — the main entry point for prompt processing.

Flow: analyze complexity → decompose (if needed) → assign → schedule → execute → regenerate

Uses all artifacts:
  - hardware.yaml: GPU info for model loading
  - runtime.yaml: per-model optimized configs
  - confidence.yaml: domain scores for model selection
  - registry.yaml: installed models
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from pathlib import Path

from llama_cpp import Llama

from polymind.core.hardware.loader import load_hardware_profile
from polymind.core.model.registry import ModelRegistry
from polymind.core.pipeline.decomposer import _detect_domain, decompose_prompt
from polymind.core.pipeline.regenerator import quick_synthesize, regenerate_response
from polymind.core.pipeline.scheduler import (
    mark_completed,
    mark_failed,
    schedule_tasks,
)
from polymind.core.pipeline.selector import auto_select_models_for_tasks, select_model_for_task
from polymind.core.pipeline.types import (
    ModelRole,
    PipelineConfig,
    PipelineResult,
    Task,
    TaskStatus,
    TaskType,
)
from polymind.core.runtime.artifact import load_runtime_config
from polymind.core.runtime.config import default_runtime_config

# ── Simple prompt patterns (skip decomposition) ──────────────
_SIMPLE_PATTERNS = [
    r"^(what|who|when|where|how|is|are|do|does|can|could|would|should|will)\s",
    r"^\d+\s*[\+\-\*\/\%]\s*\d+",
    r"^(hi|hello|hey|thanks|ok|yes|no|bye)\s*[!.?]*$",
    r"^(define|name|list|give me|tell me)\s",
]

_SIMPLE_MIN_WORDS = 8
_SIMPLE_MAX_WORDS = 12


class Pipeline:
    """Main pipeline orchestrator.

    Executes: analyze → decompose → assign → schedule → execute → regenerate
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.registry = ModelRegistry()
        self._loaded_models: dict[str, Llama] = {}
        self._progress_callback: Callable[[str, Task], None] | None = None
        self._warnings: list[str] = []

    def on_progress(self, callback: Callable[[str, Task], None]) -> None:
        """Register a progress callback: callback(event, task)."""
        self._progress_callback = callback

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)

    def run(self, prompt: str) -> PipelineResult:
        """Execute the full pipeline for a user prompt."""
        import signal

        start_time = time.time()
        result = PipelineResult(prompt=prompt, response="")
        self._warnings = []
        self._cancelled = False

        # Handle Ctrl+C gracefully
        def _signal_handler(sig, frame):
            self._cancelled = True
            self._warnings.append("Pipeline cancelled by user (Ctrl+C)")

        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _signal_handler)

        try:
            return self._run_inner(prompt, result, start_time)
        except KeyboardInterrupt:
            self._warnings.append("Pipeline cancelled by user (Ctrl+C)")
            result.response = result.response or "Pipeline cancelled."
            return result
        finally:
            signal.signal(signal.SIGINT, original_handler)
            self._unload_models()

    def _run_inner(
        self,
        prompt: str,
        result: PipelineResult,
        start_time: float,
    ) -> PipelineResult:
        """Inner pipeline execution (can be cancelled)."""
        self._warnings = []

        # ── Step 0: Validate artifacts ────────────────────────
        self._validate_artifacts()

        # ── Step 1: Analyze complexity ────────────────────────
        is_simple = self._is_simple(prompt)
        word_count = len(prompt.split())

        self._emit(
            "analyze",
            Task(
                id="analyze",
                prompt=prompt,
                domain="",
                task_type=TaskType.CUSTOM,
                metadata={
                    "is_simple": str(is_simple).lower(),
                    "word_count": str(word_count),
                },
            ),
        )

        if is_simple:
            domain = _detect_domain(prompt)
            tasks = [
                Task(
                    id="task_1",
                    prompt=prompt,
                    domain=domain,
                    task_type=TaskType.GENERATE,
                    model_role=ModelRole.GENERATOR,
                )
            ]
        else:
            # ── Step 2: Decompose ────────────────────────────
            self._emit(
                "decompose_start",
                Task(id="decompose", prompt=prompt, domain="", task_type=TaskType.DECOMPOSE),
            )

            tasks = self._decompose(prompt)

            self._emit(
                "decompose_done",
                Task(
                    id="decompose",
                    prompt=prompt,
                    domain="",
                    task_type=TaskType.DECOMPOSE,
                    metadata={"tasks": [t.to_dict() for t in tasks]},
                ),
            )

        result.tasks = tasks

        if not tasks:
            result.response = "Failed to process prompt."
            return result

        # ── Step 3: Assign models ────────────────────────────
        self._emit(
            "assign_start",
            Task(id="assign", prompt="", domain="", task_type=TaskType.GENERATE),
        )

        assignments = auto_select_models_for_tasks(tasks, self.config, self.registry)

        assignment_map: dict[str, str] = {}
        assign_metadata: dict[str, dict] = {}
        for task in tasks:
            a = assignments.get(task.id)
            if a and a.model:
                task.model_id = a.model_id
                task.status = TaskStatus.ASSIGNED
                assignment_map[task.id] = a.model_id
                task.metadata["assign_reason"] = a.reason
                assign_metadata[task.id] = {
                    "domain": task.domain,
                    "model_id": a.model_id,
                    "confidence": f"{a.confidence:.1f}%" if a.confidence > 0 else "size-based",
                    "reason": a.reason,
                }
            else:
                self._warnings.append(f"No model found for task {task.id} ({task.domain})")

        self._emit(
            "assign_done",
            Task(
                id="assign",
                prompt="",
                domain="",
                task_type=TaskType.GENERATE,
                metadata={"assignments": assign_metadata},
            ),
        )

        # ── Step 4: Schedule ─────────────────────────────────
        plan = schedule_tasks(tasks, assignment_map)
        result.task_groups = plan.groups
        result.model_loads = plan.estimated_model_loads
        result.models_used = list(set(g.model_id for g in plan.groups if g.model_id))

        self._emit(
            "schedule",
            Task(
                id="schedule",
                prompt="",
                domain="",
                task_type=TaskType.CUSTOM,
                metadata={
                    "groups": [g.to_dict() for g in plan.groups],
                    "model_loads": str(plan.estimated_model_loads),
                },
            ),
        )

        # ── Step 5: Execute ──────────────────────────────────
        completed_ids: set[str] = set()
        max_retries = self.config.max_retries

        for group in plan.groups:
            if not group.model_id:
                for task in group.tasks:
                    mark_failed(task, "no model assigned")
                    self._emit("task_done", task)
                    completed_ids.add(task.id)
                continue

            # Load model once for the group (uses runtime.yaml config)
            model = self._get_or_load_model(group.model_id)
            if model is None:
                for task in group.tasks:
                    mark_failed(task, f"failed to load model {group.model_id}")
                    self._emit("task_done", task)
                    completed_ids.add(task.id)
                continue

            for task in group.tasks:
                if self._cancelled:
                    mark_failed(task, "cancelled")
                    self._emit("task_done", task)
                    completed_ids.add(task.id)
                    continue

                self._emit("task_start", task)
                task.status = TaskStatus.RUNNING

                deps_met = all(d in completed_ids for d in task.depends_on)
                if not deps_met:
                    mark_failed(task, "dependencies not met")
                    self._emit("task_done", task)
                    completed_ids.add(task.id)
                    continue

                success = False
                for attempt in range(max_retries + 1):
                    try:
                        response = self._execute_task(task, model)
                        if response:
                            mark_completed(task, response, score=0.8)
                            success = True
                            break
                    except Exception as e:
                        if attempt == max_retries:
                            mark_failed(task, str(e))

                if not success and task.status != TaskStatus.COMPLETED:
                    mark_failed(task, "execution failed")

                self._emit("task_done", task)
                completed_ids.add(task.id)

        # ── Step 6: Regenerate ───────────────────────────────
        self._emit(
            "regenerate_start",
            Task(id="regenerate", prompt=prompt, domain="", task_type=TaskType.REGENERATE),
        )

        completed_tasks = [t for t in tasks if t.status == TaskStatus.COMPLETED]
        if len(completed_tasks) == 1 and is_simple:
            result.response = completed_tasks[0].result
        else:
            result.response = self._regenerate(prompt, tasks)

        result.domain_scores = self._compute_domain_scores(tasks)

        scores = [t.score for t in tasks if t.score > 0]
        result.overall_score = sum(scores) / len(scores) if scores else 0.0

        self._emit(
            "regenerate_done",
            Task(id="regenerate", prompt=prompt, domain="", task_type=TaskType.REGENERATE),
        )

        self._unload_models()
        result.total_time_ms = (time.time() - start_time) * 1000
        return result

    # ── Complexity Analysis ──────────────────────────────────

    def _is_simple(self, prompt: str) -> bool:
        """Determine if a prompt is simple enough to skip decomposition."""
        words = prompt.split()
        word_count = len(words)

        if word_count <= _SIMPLE_MAX_WORDS:
            for pattern in _SIMPLE_PATTERNS:
                if re.match(pattern, prompt.strip(), re.IGNORECASE):
                    return True

            if word_count <= _SIMPLE_MIN_WORDS:
                connectors = [" and ", " then ", " also ", " first ", " next ", " finally "]
                if not any(c in prompt.lower() for c in connectors):
                    return True
                return False

        multi_step = [
            "step 1",
            "step 2",
            "first,",
            "then,",
            "also,",
            "and then",
            "after that",
            "finally,",
            "write a",
            "create a",
            "build a",
            "implement",
            "design a",
            "make a",
        ]
        prompt_lower = prompt.lower()
        if any(m in prompt_lower for m in multi_step):
            return False

        if word_count > 100:
            return False

        return False

    # ── Decomposition ────────────────────────────────────────

    def _decompose(self, prompt: str) -> list[Task]:
        """Decompose prompt into tasks."""
        dummy_task = Task(
            id="decompose",
            prompt=prompt,
            domain="",
            task_type=TaskType.DECOMPOSE,
            model_role=ModelRole.DECOMPOSER,
        )
        assignment = select_model_for_task(dummy_task, self.config, self.registry)

        if assignment.model is None:
            return self._fallback_single_task(prompt, "no decomposer model available")

        model_path = Path(assignment.model.local_path)
        if not model_path.exists():
            return self._fallback_single_task(prompt, f"model file not found: {model_path}")

        model_config = self._get_model_config(assignment.model_id, assignment.model.size_bytes)

        try:
            return decompose_prompt(prompt, model_path, model_config=model_config)
        except Exception as e:
            self._warnings.append(f"Decomposition failed ({e}), using single task")
            return self._fallback_single_task(prompt, str(e))

    def _fallback_single_task(self, prompt: str, reason: str) -> list[Task]:
        """Create a single task as fallback."""
        return [
            Task(
                id="task_1",
                prompt=prompt,
                domain=_detect_domain(prompt),
                task_type=TaskType.GENERATE,
                model_role=ModelRole.GENERATOR,
                metadata={"fallback_reason": reason},
            )
        ]

    # ── Task Execution ───────────────────────────────────────

    def _execute_task(self, task: Task, model: Llama) -> str:
        """Execute a single task using the loaded model. Returns result string."""
        import time

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are an expert in {task.domain}. "
                    "Provide accurate, detailed, and well-structured responses. "
                    "Be concise but complete."
                ),
            },
            {"role": "user", "content": task.prompt},
        ]

        start = time.time()
        output = model.create_chat_completion(
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )
        elapsed = time.time() - start

        content = output["choices"][0]["message"]["content"] or ""
        result = content.strip()

        # Calculate timing metadata
        usage = output.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0

        task.metadata["elapsed_s"] = f"{elapsed:.1f}"
        task.metadata["tokens"] = str(completion_tokens)
        task.metadata["tok_per_s"] = f"{tok_per_sec:.1f}"

        return result

    # ── Regeneration ─────────────────────────────────────────

    def _regenerate(self, prompt: str, tasks: list[Task]) -> str:
        """Regenerate final response from task results."""
        completed = [t for t in tasks if t.status == TaskStatus.COMPLETED and t.result]
        if not completed:
            return "No task results available."
        if len(completed) == 1:
            return completed[0].result

        dummy_task = Task(
            id="regenerate",
            prompt=prompt,
            domain="",
            task_type=TaskType.REGENERATE,
            model_role=ModelRole.REGENERATOR,
        )
        assignment = select_model_for_task(dummy_task, self.config, self.registry)

        if assignment.model is None:
            return quick_synthesize(tasks)

        model_path = Path(assignment.model.local_path)
        if not model_path.exists():
            return quick_synthesize(tasks)

        model_config = self._get_model_config(assignment.model_id, assignment.model.size_bytes)

        try:
            return regenerate_response(prompt, tasks, model_path, model_config=model_config)
        except Exception:
            return quick_synthesize(tasks)

    # ── Model Loading (uses runtime.yaml) ────────────────────

    def _get_or_load_model(self, model_id: str) -> Llama | None:
        """Load a model using its runtime.yaml config, or return cached."""
        if model_id in self._loaded_models:
            return self._loaded_models[model_id]

        models = self.registry.load()
        model = None
        for m in models:
            if str(m.id) == model_id:
                model = m
                break

        if model is None:
            self._warnings.append(f"Model {model_id} not found in registry")
            return None

        model_path = Path(model.local_path)
        if not model_path.exists():
            self._warnings.append(f"Model file not found: {model.local_path}")
            return None

        config = self._get_model_config(model_id, model.size_bytes)

        # Emit model loading event with full metadata
        self._emit(
            "model_load",
            Task(
                id="load",
                prompt="",
                domain="",
                task_type=TaskType.CUSTOM,
                metadata={
                    "model_id": model_id,
                    "model_file": model.filename,
                    "gpu_layers": str(config["gpu_layers"]),
                    "threads": str(config["threads"]),
                    "context_size": str(config["context_size"]),
                    "size_bytes": str(model.size_bytes),
                },
            ),
        )

        try:
            llm = Llama(
                model_path=str(model_path),
                n_gpu_layers=config["gpu_layers"],
                n_threads=config["threads"],
                n_ctx=config["context_size"],
                n_batch=config["batch_size"],
                verbose=False,
            )
            self._loaded_models[model_id] = llm

            self._emit(
                "model_loaded",
                Task(
                    id="load",
                    prompt="",
                    domain="",
                    task_type=TaskType.CUSTOM,
                    metadata={"model_id": model_id},
                ),
            )

            return llm
        except Exception as e:
            self._warnings.append(f"Failed to load model {model_id}: {e}")
            self._emit(
                "model_load_failed",
                Task(
                    id="load",
                    prompt="",
                    domain="",
                    task_type=TaskType.CUSTOM,
                    metadata={"model_id": model_id, "reason": str(e)},
                ),
            )
            return None

    def _get_model_config(self, model_id: str, size_bytes: int) -> dict:
        """Get model config: runtime.yaml → hardware-aware defaults."""
        runtime = load_runtime_config(model_id)
        if runtime is not None:
            return {
                "gpu_layers": runtime.gpu_layers,
                "threads": runtime.threads,
                "context_size": runtime.context_size,
                "batch_size": runtime.batch_size,
            }

        default = default_runtime_config(model_id, size_bytes)
        return {
            "gpu_layers": default.gpu_layers,
            "threads": default.threads,
            "context_size": default.context_size,
            "batch_size": default.batch_size,
        }

    # ── Artifact Validation ──────────────────────────────────

    def _validate_artifacts(self) -> None:
        """Check that required artifacts exist, emit status for verbose mode."""
        artifacts: dict[str, str] = {}

        # Registry
        models = self.registry.load()
        if models:
            artifacts["registry.yaml"] = f"{len(models)} model(s) installed"
        else:
            artifacts["registry.yaml"] = "missing (no models)"
            self._warnings.append("No models installed. Run: polymind model download <repo>")

        # Hardware profile
        try:
            hw = load_hardware_profile()
            if hw.gpus:
                gpu_names = ", ".join(g.model for g in hw.gpus)
                artifacts["hardware.yaml"] = f"{gpu_names} ({len(hw.gpus)} GPU(s))"
            else:
                artifacts["hardware.yaml"] = "no GPUs detected (CPU only)"
                self._warnings.append("No GPUs detected. Using CPU only.")
        except Exception as e:
            artifacts["hardware.yaml"] = f"error: {e}"
            self._warnings.append("Hardware profile error. Run: polymind hardware scan")

        # Runtime config
        from polymind.core.paths import runtime_path

        rt_path = runtime_path()
        if rt_path.exists():
            import yaml

            with rt_path.open() as f:
                data = yaml.safe_load(f) or {}
            rt_models = data.get("models", {})
            artifacts["runtime.yaml"] = f"{len(rt_models)} model(s) configured"
        else:
            artifacts["runtime.yaml"] = "missing (using defaults)"
            self._warnings.append("No runtime config. Run: polymind runtime optimize.")

        # Confidence scores
        from polymind.core.confidence.artifact import load_confidence

        confidence = load_confidence()
        if confidence:
            artifacts["confidence.yaml"] = f"{len(confidence)} model(s) scored"
        else:
            artifacts["confidence.yaml"] = "missing (no scores)"
            self._warnings.append(
                "No confidence scores. Run: polymind confidence compute for optimal model selection."
            )

        self._emit(
            "artifact_check",
            Task(
                id="artifacts",
                prompt="",
                domain="",
                task_type=TaskType.CUSTOM,
                metadata=artifacts,
            ),
        )

    # ── Helpers ──────────────────────────────────────────────

    def _unload_models(self) -> None:
        for model_id in list(self._loaded_models.keys()):
            del self._loaded_models[model_id]
        self._loaded_models.clear()

    def _compute_domain_scores(self, tasks: list[Task]) -> dict[str, float]:
        domain_scores: dict[str, list[float]] = {}
        for task in tasks:
            if task.score > 0:
                domain_scores.setdefault(task.domain, []).append(task.score)
        return {domain: sum(s) / len(s) for domain, s in domain_scores.items()}

    def _emit(self, event: str, task: Task) -> None:
        if self._progress_callback:
            self._progress_callback(event, task)

"""Model selector — auto-assigns the best model for each task.

Selection strategy:
1. If user specified a model in PipelineConfig, use it
2. Otherwise, find the best model for the task's domain using confidence scores
3. Fall back to the largest available model if no confidence data exists
"""

from __future__ import annotations

from dataclasses import dataclass

from polymind.core.confidence.artifact import load_confidence
from polymind.core.model.registry import InstalledModel, ModelRegistry
from polymind.core.pipeline.types import ModelRole, PipelineConfig, Task


@dataclass
class ModelAssignment:
    """Result of model selection for a task."""

    model_id: str
    model: InstalledModel | None
    reason: str
    confidence: float = 0.0


def select_model_for_task(
    task: Task,
    config: PipelineConfig,
    registry: ModelRegistry | None = None,
) -> ModelAssignment:
    """Select the best model for a given task.

    Priority:
    1. Explicit assignment in config (by role)
    2. Explicit assignment on the task itself
    3. Auto-select based on domain confidence scores
    4. Fall back to largest available model
    """
    if registry is None:
        registry = ModelRegistry()

    models = registry.load()
    if not models:
        return ModelAssignment(model_id="", model=None, reason="no models installed")

    # 1. Check config role assignment
    role_model = _get_role_model(task.model_role, config)
    if role_model:
        model = _find_model(role_model, models)
        if model:
            return ModelAssignment(
                model_id=role_model,
                model=model,
                reason=f"configured for role {task.model_role.value}",
            )

    # 2. Check task-level assignment
    if task.model_id:
        model = _find_model(task.model_id, models)
        if model:
            return ModelAssignment(
                model_id=task.model_id,
                model=model,
                reason="assigned to task",
            )

    # 3. Auto-select based on confidence scores
    if config.auto_select_models:
        best = _auto_select(task.domain, task.model_role, models)
        if best:
            return best

    # 4. Fallback: largest model
    largest = max(models, key=lambda m: m.size_bytes)
    return ModelAssignment(
        model_id=str(largest.id),
        model=largest,
        reason="fallback: largest available model",
    )


def auto_select_models_for_tasks(
    tasks: list[Task],
    config: PipelineConfig,
    registry: ModelRegistry | None = None,
) -> dict[str, ModelAssignment]:
    """Select models for all tasks, returning a mapping task_id → assignment."""
    assignments: dict[str, ModelAssignment] = {}

    for task in tasks:
        assignments[task.id] = select_model_for_task(task, config, registry)

    return assignments


def _get_role_model(role: ModelRole, config: PipelineConfig) -> str:
    """Get the model ID configured for a specific role."""
    if role == ModelRole.DECOMPOSER:
        return config.decomposer_model
    elif role == ModelRole.GENERATOR:
        return config.generator_model
    elif role == ModelRole.REGENERATOR:
        return config.regenerator_model
    elif role == ModelRole.JUDGE:
        return config.judge_model
    return ""


def _find_model(model_id: str, models: list[InstalledModel]) -> InstalledModel | None:
    """Find a model by ID (string of int)."""
    for m in models:
        if str(m.id) == model_id:
            return m
    return None


def _auto_select(
    domain: str,
    role: ModelRole,
    models: list[InstalledModel],
) -> ModelAssignment | None:
    """Auto-select the best model based on confidence scores."""
    confidence_data = load_confidence()

    if not confidence_data:
        return _select_by_size(role, models)

    best_model: InstalledModel | None = None
    best_score = -1.0
    best_reason = ""

    for model in models:
        model_conf = confidence_data.get(str(model.id))
        if model_conf is None:
            continue

        # Get domain-specific score
        domain_score = model_conf.domains.get(domain)
        if domain_score is not None:
            score = domain_score.overall
            reason = f"confidence {score:.1f}% in '{domain}'"
        else:
            # Use overall score
            score = model_conf.overall_score
            reason = f"overall confidence {score:.1f}%"

        # Apply role-based multiplier
        if role == ModelRole.DECOMPOSER:
            # Decomposer benefits from instruction-following and reasoning
            instruction_score = model_conf.domains.get("instruction", None)
            if instruction_score:
                score = (score + instruction_score.overall) / 2
                reason += " (instruction avg)"
        elif role == ModelRole.REGENERATOR:
            # Regenerator benefits from writing and synthesis
            writing_score = model_conf.domains.get("writing", None)
            if writing_score:
                score = (score + writing_score.overall) / 2
                reason += " (writing avg)"
        elif role == ModelRole.JUDGE:
            # Judge benefits from reasoning
            reasoning_score = model_conf.domains.get("reasoning", None)
            if reasoning_score:
                score = (score + reasoning_score.overall) / 2
                reason += " (reasoning avg)"

        if score > best_score:
            best_score = score
            best_model = model
            best_reason = reason

    if best_model is not None:
        return ModelAssignment(
            model_id=str(best_model.id),
            model=best_model,
            reason=best_reason,
            confidence=best_score,
        )

    return _select_by_size(role, models)


def _select_by_size(
    role: ModelRole,
    models: list[InstalledModel],
) -> ModelAssignment:
    """Select model by size — larger models for harder tasks."""
    if role in (ModelRole.DECOMPOSER, ModelRole.JUDGE):
        # Pick smallest suitable model (decompose/judge don't need huge models)
        suitable = sorted(models, key=lambda m: m.size_bytes)
    else:
        # Pick largest model for generation/synthesis
        suitable = sorted(models, key=lambda m: m.size_bytes, reverse=True)

    if not suitable:
        return ModelAssignment(model_id="", model=None, reason="no models")

    best = suitable[0]
    return ModelAssignment(
        model_id=str(best.id),
        model=best,
        reason=f"selected by size for {role.value}",
    )


def suggest_models(
    registry: ModelRegistry | None = None,
) -> dict[str, list[dict[str, str | float]]]:
    """Suggest models for each role based on what's installed.

    Returns a dict mapping role → list of suggestions with reasons.
    """
    if registry is None:
        registry = ModelRegistry()

    models = registry.load()
    confidence_data = load_confidence()

    suggestions: dict[str, list[dict[str, str | float]]] = {
        "decomposer": [],
        "generator": [],
        "regenerator": [],
        "judge": [],
    }

    for model in models:
        mid = str(model.id)
        conf = confidence_data.get(mid)

        for role_name in suggestions:
            role = ModelRole(role_name)
            score = 0.0
            reason = ""

            if conf:
                if role == ModelRole.DECOMPOSER:
                    # Prefer instruction + reasoning domains
                    instr = conf.domains.get("instruction", None)
                    reasoning = conf.domains.get("reasoning", None)
                    if instr and reasoning:
                        score = (instr.overall + reasoning.overall) / 2
                        reason = (
                            f"instruction={instr.overall:.0f}% reasoning={reasoning.overall:.0f}%"
                        )
                    else:
                        score = conf.overall_score
                        reason = f"overall={conf.overall_score:.0f}%"
                elif role == ModelRole.GENERATOR:
                    # Prefer domain-specific confidence
                    score = conf.overall_score
                    reason = f"overall={conf.overall_score:.0f}%"
                elif role == ModelRole.REGENERATOR:
                    # Prefer writing + synthesis
                    writing = conf.domains.get("writing", None)
                    if writing:
                        score = writing.overall
                        reason = f"writing={writing.overall:.0f}%"
                    else:
                        score = conf.overall_score
                        reason = f"overall={conf.overall_score:.0f}%"
                elif role == ModelRole.JUDGE:
                    # Prefer reasoning + safety
                    reasoning = conf.domains.get("reasoning", None)
                    safety = conf.domains.get("safety", None)
                    if reasoning and safety:
                        score = (reasoning.overall + safety.overall) / 2
                        reason = f"reasoning={reasoning.overall:.0f}% safety={safety.overall:.0f}%"
                    else:
                        score = conf.overall_score
                        reason = f"overall={conf.overall_score:.0f}%"
            else:
                score = model.size_bytes / (1024**3)  # Use size as proxy
                reason = f"size={model.size_bytes / (1024**3):.1f}GB (no confidence data)"

            suggestions[role_name].append(
                {
                    "model_id": mid,
                    "filename": model.filename,
                    "score": round(score, 1),
                    "reason": reason,
                }
            )

    # Sort each role by score descending
    for role_name in suggestions:
        suggestions[role_name].sort(key=lambda x: x["score"], reverse=True)

    return suggestions

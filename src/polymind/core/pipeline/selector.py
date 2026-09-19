"""Model selector — auto-assigns the best model for each task.

Selection strategy:
1. If user specified a model in PipelineConfig, use it
2. Otherwise, find the best model for the task's domain using confidence scores
3. Filter by runtime feasibility (valid profile exists and is safe)
4. Fall back to the largest available model if no confidence data exists

Runtime feasibility is checked AFTER capability scoring. A model with
domain score 0.95 but UNSAFE runtime must not beat a model with score
0.88 and READY runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

from polymind.core.confidence.artifact import load_confidence
from polymind.core.hardware.loader import load_hardware_profile
from polymind.core.model.registry import InstalledModel, ModelRegistry
from polymind.core.pipeline.types import ModelRole, PipelineConfig, Task
from polymind.core.runtime.artifact import load_runtime_profile
from polymind.core.runtime.types import ValidationStatus


@dataclass
class RuntimeFeasibility:
    """Result of checking whether a model can safely run."""

    feasible: bool
    status: str  # ready, stale, no_profile, unsafe, unknown
    profile_workload: str = ""
    reason: str = ""


@dataclass
class ModelAssignment:
    """Result of model selection for a task."""

    model_id: str
    model: InstalledModel | None
    reason: str
    confidence: float = 0.0
    runtime_feasibility: RuntimeFeasibility | None = None


def check_runtime_feasibility(
    model_id: str,
    workload: str = "default",
) -> RuntimeFeasibility:
    """Check if a model has a valid runtime profile for the current hardware.

    Returns RuntimeFeasibility with status and reason.
    """
    try:
        hardware = load_hardware_profile()
    except Exception:
        return RuntimeFeasibility(
            feasible=False,
            status="unknown",
            reason="Cannot load hardware profile",
        )

    from polymind.core.runtime.types import HardwareFingerprint

    hw_fp = HardwareFingerprint.from_hardware_profile(hardware)
    hw_fp_hash = hw_fp.compute_hash()

    profile = load_runtime_profile(model_id)

    if profile is None:
        return RuntimeFeasibility(
            feasible=False,
            status="no_profile",
            reason="No runtime profile — run: polymind runtime optimize",
        )

    # Check validation status
    if profile.validation.status == ValidationStatus.FAILED:
        return RuntimeFeasibility(
            feasible=False,
            status="unsafe",
            reason=f"Profile marked FAILED: {', '.join(profile.validation.failure_reasons[:2])}",
        )

    if (
        profile.validation.status == ValidationStatus.READY
        and profile.validation.successful_runs > 0
    ):
        # Check hardware fingerprint
        hw_valid, hw_reason = _validate_hardware(profile, hw_fp_hash)
        if not hw_valid:
            return RuntimeFeasibility(
                feasible=False,
                status="stale",
                reason=hw_reason,
            )
        return RuntimeFeasibility(
            feasible=True,
            status="ready",
            profile_workload=profile.workload,
            reason=f"Validated profile ({profile.validation.successful_runs} successful runs)",
        )

    # STALE or UNKNOWN — might work but not validated
    if profile.validation.status == ValidationStatus.STALE:
        return RuntimeFeasibility(
            feasible=True,
            status="stale",
            profile_workload=profile.workload,
            reason="Legacy profile (not validated for current hardware)",
        )

    return RuntimeFeasibility(
        feasible=True,
        status="unknown",
        profile_workload=profile.workload,
        reason="Profile exists but validation status unknown",
    )


def _validate_hardware(profile: object, current_hw_fp_hash: str) -> tuple[bool, str]:
    """Check if the profile's hardware fingerprint matches current hardware."""
    hw_fp = getattr(profile, "hardware_fingerprint", "")
    if not hw_fp:
        return True, "No fingerprint (legacy)"
    if hw_fp == current_hw_fp_hash:
        return True, "Hardware matches"
    return False, f"Hardware changed (profile: {hw_fp}, current: {current_hw_fp_hash})"


def select_model_for_task(
    task: Task,
    config: PipelineConfig,
    registry: ModelRegistry | None = None,
) -> ModelAssignment:
    """Select the best model for a given task.

    Priority:
    1. Explicit assignment in config (by role)
    2. Explicit assignment on the task itself
    3. Auto-select based on domain confidence scores + runtime feasibility
    4. Fall back to largest feasible model
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
            # Still check runtime feasibility even for explicit assignments
            feasibility = check_runtime_feasibility(role_model)
            return ModelAssignment(
                model_id=role_model,
                model=model,
                reason=f"configured for role {task.model_role.value}",
                runtime_feasibility=feasibility,
            )

    # 2. Check task-level assignment
    if task.model_id:
        model = _find_model(task.model_id, models)
        if model:
            feasibility = check_runtime_feasibility(task.model_id)
            return ModelAssignment(
                model_id=task.model_id,
                model=model,
                reason="assigned to task",
                runtime_feasibility=feasibility,
            )

    # 3. Auto-select based on confidence scores + runtime feasibility
    if config.auto_select_models:
        best = _auto_select_with_feasibility(task.domain, task.model_role, models)
        if best:
            return best

    # 4. Fallback: largest feasible model
    return _select_largest_feasible(models)


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
    """Legacy alias for _auto_select_with_feasibility."""
    return _auto_select_with_feasibility(domain, role, models)


def _auto_select_with_feasibility(
    domain: str,
    role: ModelRole,
    models: list[InstalledModel],
) -> ModelAssignment | None:
    """Auto-select the best model using capability + runtime feasibility.

    Algorithm:
    1. Compute capability score for each model in the domain
    2. Check runtime feasibility for each model
    3. Rank by: (capability_score, feasibility_score)
    4. A model with UNSAFE runtime cannot beat a model with READY runtime
    """
    from polymind.core.confidence.artifact import load_all_domains

    confidence_data = load_confidence()

    if not confidence_data:
        return _select_by_size_with_feasibility(role, models)

    # Resolve domain aliases
    all_domains = load_all_domains()
    resolved_domain = domain
    for d in all_domains:
        if d.id == domain or domain in d.aliases:
            resolved_domain = d.id
            break

    scored_models: list[tuple[InstalledModel, float, str, RuntimeFeasibility]] = []

    for model in models:
        mid = str(model.id)
        model_conf = confidence_data.get(mid)

        # Get capability score
        score = 0.0
        reason = ""

        if model_conf:
            domain_score = model_conf.domains.get(resolved_domain)
            if domain_score is None and resolved_domain != domain:
                domain_score = model_conf.domains.get(domain)

            if domain_score is not None:
                score = domain_score.overall
                reason = f"confidence {score:.1f}% in '{resolved_domain}'"
            else:
                score = model_conf.overall_score
                reason = f"overall confidence {score:.1f}%"

            # Role-based adjustments
            if role == ModelRole.DECOMPOSER:
                instr = model_conf.domains.get("instruction", None)
                if instr:
                    score = (score + instr.overall) / 2
                    reason += " (instruction avg)"
            elif role == ModelRole.REGENERATOR:
                writing = model_conf.domains.get("writing", None)
                if writing:
                    score = (score + writing.overall) / 2
                    reason += " (writing avg)"
            elif role == ModelRole.JUDGE:
                reasoning = model_conf.domains.get("reasoning", None)
                if reasoning:
                    score = (score + reasoning.overall) / 2
                    reason += " (reasoning avg)"
        else:
            score = model.size_bytes / (1024**3) * 10  # Size as proxy
            reason = f"size={model.size_bytes / (1024**3):.1f}GB (no confidence data)"

        # Check runtime feasibility
        feasibility = check_runtime_feasibility(mid)

        scored_models.append((model, score, reason, feasibility))

    if not scored_models:
        return _select_by_size_with_feasibility(role, models)

    # Sort: primary = feasibility (ready > stale > unknown > no_profile > unsafe),
    # secondary = capability score
    feasibility_rank = {
        "ready": 0,
        "stale": 1,
        "unknown": 2,
        "no_profile": 3,
        "unsafe": 4,
    }

    scored_models.sort(
        key=lambda x: (
            feasibility_rank.get(x[3].status, 5),
            -x[1],  # Higher score is better
        ),
    )

    best_model, best_score, best_reason, best_feasibility = scored_models[0]

    return ModelAssignment(
        model_id=str(best_model.id),
        model=best_model,
        reason=f"{best_reason} | runtime: {best_feasibility.status}",
        confidence=best_score,
        runtime_feasibility=best_feasibility,
    )


def _select_by_size_with_feasibility(
    role: ModelRole,
    models: list[InstalledModel],
) -> ModelAssignment:
    """Select model by size with runtime feasibility check."""
    if role in (ModelRole.DECOMPOSER, ModelRole.JUDGE):
        suitable = sorted(models, key=lambda m: m.size_bytes)
    else:
        suitable = sorted(models, key=lambda m: m.size_bytes, reverse=True)

    if not suitable:
        return ModelAssignment(model_id="", model=None, reason="no models")

    # Try each model, prefer ones with valid runtime profiles
    for model in suitable:
        feasibility = check_runtime_feasibility(str(model.id))
        if feasibility.status in ("ready", "stale"):
            return ModelAssignment(
                model_id=str(model.id),
                model=model,
                reason=f"selected by size for {role.value} (runtime: {feasibility.status})",
                runtime_feasibility=feasibility,
            )

    # Fallback to largest regardless of runtime status
    best = suitable[0]
    feasibility = check_runtime_feasibility(str(best.id))
    return ModelAssignment(
        model_id=str(best.id),
        model=best,
        reason=f"selected by size for {role.value} (no validated runtime)",
        runtime_feasibility=feasibility,
    )


def _select_largest_feasible(
    models: list[InstalledModel],
) -> ModelAssignment:
    """Select the largest model with a feasible runtime profile."""
    sorted_models = sorted(models, key=lambda m: m.size_bytes, reverse=True)

    for model in sorted_models:
        feasibility = check_runtime_feasibility(str(model.id))
        if feasibility.status in ("ready", "stale"):
            return ModelAssignment(
                model_id=str(model.id),
                model=model,
                reason=f"largest feasible model (runtime: {feasibility.status})",
                runtime_feasibility=feasibility,
            )

    # No feasible model — pick largest anyway
    largest = sorted_models[0]
    feasibility = check_runtime_feasibility(str(largest.id))
    return ModelAssignment(
        model_id=str(largest.id),
        model=largest,
        reason="fallback: largest available model (no validated runtime)",
        runtime_feasibility=feasibility,
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
        feasibility = check_runtime_feasibility(mid)

        for role_name in suggestions:
            role = ModelRole(role_name)
            score = 0.0
            reason = ""

            if conf:
                if role == ModelRole.DECOMPOSER:
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
                    score = conf.overall_score
                    reason = f"overall={conf.overall_score:.0f}%"
                elif role == ModelRole.REGENERATOR:
                    writing = conf.domains.get("writing", None)
                    if writing:
                        score = writing.overall
                        reason = f"writing={writing.overall:.0f}%"
                    else:
                        score = conf.overall_score
                        reason = f"overall={conf.overall_score:.0f}%"
                elif role == ModelRole.JUDGE:
                    reasoning = conf.domains.get("reasoning", None)
                    safety = conf.domains.get("safety", None)
                    if reasoning and safety:
                        score = (reasoning.overall + safety.overall) / 2
                        reason = f"reasoning={reasoning.overall:.0f}% safety={safety.overall:.0f}%"
                    else:
                        score = conf.overall_score
                        reason = f"overall={conf.overall_score:.0f}%"
            else:
                score = model.size_bytes / (1024**3)
                reason = f"size={model.size_bytes / (1024**3):.1f}GB (no confidence data)"

            runtime_tag = f"runtime={feasibility.status}"
            suggestions[role_name].append(
                {
                    "model_id": mid,
                    "filename": model.filename,
                    "score": round(score, 1),
                    "reason": f"{reason} | {runtime_tag}",
                }
            )

    for role_name in suggestions:
        suggestions[role_name].sort(key=lambda x: x["score"], reverse=True)

    return suggestions

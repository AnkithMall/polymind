from __future__ import annotations

import logging
from collections import deque

from polymind.core.providers import (
    ONLINE_PROVIDERS,
    ProviderType,
    provider_model_source,
    resolve_model_string,
)
from polymind.core.types import (
    AnalyzerPlan,
    DomainType,
    ExecutionSchedule,
    ExecutionStrategy,
    ModelBatch,
    ModelSource,
    RankingMode,
    RankStore,
    Subtask,
)

logger = logging.getLogger(__name__)


def _from_provider_type(ref: str) -> ProviderType:
    if "/" in ref:
        prov = ref.split("/", 1)[0]
        try:
            return ProviderType(prov)
        except ValueError:
            pass
    return ProviderType.ollama


def _model_matches_source(ref: str, source: ModelSource) -> bool:
    if source == ModelSource.all:
        return True
    ptype = _from_provider_type(ref)
    ms = provider_model_source(ptype)
    return ms == source


def _assign_models(
    plan: AnalyzerPlan,
    rank_store: RankStore | None,
    default_model: str,
    fallback_models: list[str] | None,
    ranking_mode: RankingMode = RankingMode.accuracy,
    model_source: ModelSource = ModelSource.all,
) -> None:
    for subtask in plan.subtasks:
        if subtask.assigned_model:
            logger.debug(
                "Subtask %s already assigned model %s, skipping",
                subtask.id,
                subtask.assigned_model,
            )
            continue

        if rank_store is not None:
            logger.debug(
                "Looking up rank for subtask %s (domain=%s)",
                subtask.id,
                subtask.domain,
            )
            top = rank_store.top_for_domain(subtask.domain, ranking_mode=ranking_mode)
            if top is not None and _model_matches_source(top.model, model_source):
                subtask.assigned_model = top.model
                logger.debug(
                    "Subtask %s assigned ranked model %s", subtask.id, top.model
                )
                continue

            fallback_entries = rank_store.top_n_for_domain(
                subtask.domain, len(fallback_models or []), ranking_mode=ranking_mode
            )
            for i, entry in enumerate(fallback_entries):
                try:
                    fallback_models[i]
                except IndexError:
                    break
                if entry.model != default_model and _model_matches_source(
                    entry.model, model_source
                ):
                    subtask.assigned_model = fallback_models[i]
                    logger.debug(
                        "Subtask %s assigned fallback model %s",
                        subtask.id,
                        fallback_models[i],
                    )
                    break

        if subtask.assigned_model is None and not _model_matches_source(
            default_model, model_source
        ):
            subtask.assigned_model = default_model
            logger.debug(
                "Subtask %s assigned default model %s", subtask.id, default_model
            )
        elif subtask.assigned_model is None:
            subtask.assigned_model = default_model
            logger.debug(
                "Subtask %s assigned default model %s", subtask.id, default_model
            )


def _topological_batches(
    plan: AnalyzerPlan,
) -> list[list[Subtask]]:
    subtask_map = {s.id: s for s in plan.subtasks}
    in_degree: dict[str, int] = {}
    dependents: dict[str, list[str]] = {}

    for s in plan.subtasks:
        in_degree[s.id] = len(s.depends_on)
        dependents[s.id] = []

    for s in plan.subtasks:
        for dep_id in s.depends_on:
            if dep_id in dependents:
                dependents[dep_id].append(s.id)

    queue: deque[str] = deque(sid for sid, deg in in_degree.items() if deg == 0)
    ordered: list[str] = []

    while queue:
        sid = queue.popleft()
        ordered.append(sid)
        for dep_of in dependents.get(sid, []):
            in_degree[dep_of] -= 1
            if in_degree[dep_of] == 0:
                queue.append(dep_of)

    if len(ordered) != len(plan.subtasks):
        unscheduled = set(subtask_map.keys()) - set(ordered)
        logger.warning(
            "Cycle detected: %d/%d tasks scheduled, unscheduled: %s",
            len(ordered),
            len(plan.subtasks),
            unscheduled,
        )
        ordered.extend(unscheduled)

    return [[subtask_map[sid]] for sid in ordered]


def _model_aware_batches(
    plan: AnalyzerPlan,
) -> list[ModelBatch]:
    subtask_map = {s.id: s for s in plan.subtasks}
    in_degree: dict[str, int] = {}
    dependents: dict[str, list[str]] = {}

    for s in plan.subtasks:
        in_degree[s.id] = len(s.depends_on)
        dependents[s.id] = []

    for s in plan.subtasks:
        for dep_id in s.depends_on:
            if dep_id in dependents:
                dependents[dep_id].append(s.id)

    ready: deque[str] = deque(sid for sid, deg in in_degree.items() if deg == 0)
    scheduled: set[str] = set()
    batches: list[ModelBatch] = []

    while ready:
        sid = ready.popleft()
        if sid in scheduled:
            continue

        model = subtask_map[sid].assigned_model or "unknown"
        batch_ids = [sid]
        scheduled.add(sid)

        # Phase 1 — group all currently-ready tasks that share the same model.
        # Tasks for other models stay in the ready queue.
        remaining: list[str] = list(ready)
        ready.clear()
        for candidate_id in remaining:
            if candidate_id in scheduled:
                continue
            candidate = subtask_map[candidate_id]
            if (candidate.assigned_model or "unknown") == model:
                batch_ids.append(candidate_id)
                scheduled.add(candidate_id)
            else:
                ready.append(candidate_id)

        # Phase 2 — process dependencies of the batch so far and check whether
        # newly-unblocked tasks can reuse this model load (recursive lookahead).
        newly_ready: list[str] = []
        to_process: list[str] = list(batch_ids)
        while to_process:
            batch_id = to_process.pop()
            for dep_of in dependents.get(batch_id, []):
                in_degree[dep_of] -= 1
                if in_degree[dep_of] == 0 and dep_of not in scheduled:
                    if (subtask_map[dep_of].assigned_model or "unknown") == model:
                        batch_ids.append(dep_of)
                        scheduled.add(dep_of)
                        to_process.append(dep_of)
                    else:
                        newly_ready.append(dep_of)

        for nid in newly_ready:
            ready.append(nid)

        batches.append(ModelBatch(model=model, subtask_ids=batch_ids))

    return batches


def _sequential_batches(
    plan: AnalyzerPlan,
) -> list[ModelBatch]:
    batches = _topological_batches(plan)
    return [
        ModelBatch(
            model=s.assigned_model or "unknown",
            subtask_ids=[s.id],
        )
        for batch in batches
        for s in batch
    ]


def build_schedule(
    plan: AnalyzerPlan,
    rank_store: RankStore | None = None,
    strategy: ExecutionStrategy = ExecutionStrategy.model_aware,
    default_model: str = "ollama/llama3.2:1b",
    fallback_models: list[str] | None = None,
    ranking_mode: RankingMode = RankingMode.accuracy,
    model_source: ModelSource = ModelSource.all,
) -> ExecutionSchedule:
    _assign_models(
        plan,
        rank_store,
        default_model,
        fallback_models,
        ranking_mode=ranking_mode,
        model_source=model_source,
    )

    if strategy == ExecutionStrategy.sequential:
        batches = _sequential_batches(plan)
    elif strategy == ExecutionStrategy.model_aware:
        batches = _model_aware_batches(plan)
    elif strategy == ExecutionStrategy.parallel:
        batches = _parallel_batches(plan)
    else:
        batches = _model_aware_batches(plan)

    return ExecutionSchedule(strategy=strategy, batches=batches)


def _parallel_batches(
    plan: AnalyzerPlan,
) -> list[ModelBatch]:
    subtask_map = {s.id: s for s in plan.subtasks}
    in_degree: dict[str, int] = {}
    dependents: dict[str, list[str]] = {}

    for s in plan.subtasks:
        in_degree[s.id] = len(s.depends_on)
        dependents[s.id] = []

    for s in plan.subtasks:
        for dep_id in s.depends_on:
            if dep_id in dependents:
                dependents[dep_id].append(s.id)

    batches: list[ModelBatch] = []
    scheduled: set[str] = set()

    while len(scheduled) < len(plan.subtasks):
        current_batch: list[str] = [
            sid
            for sid in subtask_map
            if sid not in scheduled
            and all(dep in scheduled for dep in subtask_map[sid].depends_on)
        ]

        if not current_batch:
            remaining = set(subtask_map.keys()) - scheduled
            current_batch = list(remaining)

        by_model: dict[str, list[str]] = {}
        for sid in current_batch:
            model = subtask_map[sid].assigned_model or "unknown"
            by_model.setdefault(model, []).append(sid)

        for model, ids in by_model.items():
            batches.append(ModelBatch(model=model, subtask_ids=ids))
            scheduled.update(ids)

    return batches


def describe_schedule(plan: AnalyzerPlan, schedule: ExecutionSchedule) -> str:
    """Return a human-readable block showing the DAG, model assignments,
    batches, and load-count comparison."""
    lines: list[str] = []

    # ── 1. Subtask table ──────────────────────────────────────────────
    lines.append("Subtasks:")
    lines.append(f"  {'ID':<6} {'Domain':<16} {'Model':<28} {'Deps':<20}")
    lines.append(f"  {'-'*6} {'-'*16} {'-'*28} {'-'*20}")
    smap = {s.id: s for s in plan.subtasks}
    for s in plan.subtasks:
        deps = ", ".join(s.depends_on) if s.depends_on else "—"
        assigned = s.assigned_model or "unassigned"
        lines.append(f"  {s.id:<6} {s.domain.value:<16} {assigned:<28} {deps:<20}")
    lines.append("")

    # ── 2. Dependency edges (DAG) ─────────────────────────────────────
    edges = [(s.id, d) for s in plan.subtasks for d in s.depends_on]
    if edges:
        lines.append("Dependency edges (a → b  means  b depends on a):")
        for src, dst in edges:
            lines.append(f"  {src} → {dst}")
        topo_lines = _topological_trace(plan)
        lines.append(f"  Topological order: {' → '.join(topo_lines)}")
    else:
        lines.append("No dependencies — all tasks are independent.")
    lines.append("")

    # ── 3. Batches ────────────────────────────────────────────────────
    lines.append(f"Execution strategy: [bold]{schedule.strategy.value}[/]")
    lines.append(f"Batches ({len(schedule.batches)}):")
    for i, batch in enumerate(schedule.batches, 1):
        ids = ", ".join(batch.subtask_ids)
        model_domains = []
        for sid in batch.subtask_ids:
            s = smap.get(sid)
            if s:
                model_domains.append(f"{s.domain.value}")
        domain_str = ", ".join(model_domains)
        lines.append(f"  Batch {i}: model={batch.model}")
        lines.append(f"           tasks=({ids})")
        lines.append(f"           domains=({domain_str})")
    lines.append("")

    # ── 4. Load-count comparison ────────────────────────────────────
    actual = count_model_loads(schedule)
    # Naive baseline: each task in topological order, worst-case
    naive = _naive_load_count(plan)
    saved = naive - actual
    pct = int((saved / naive) * 100) if naive > 0 else 0
    lines.append(f"Model load comparison:")
    lines.append(f"  Naive (one task at a time): {naive} loads")
    lines.append(f"  {schedule.strategy.value}:           {actual} loads")
    lines.append(f"  Saved:                      {saved} loads ({pct}% fewer)")

    return "\n".join(lines)


def _topological_trace(plan: AnalyzerPlan) -> list[str]:
    """Return task IDs in topological order."""
    smap = {s.id: s for s in plan.subtasks}
    in_deg = {s.id: len(s.depends_on) for s in plan.subtasks}
    deps_of: dict[str, list[str]] = {s.id: [] for s in plan.subtasks}
    for s in plan.subtasks:
        for d in s.depends_on:
            if d in deps_of:
                deps_of[d].append(s.id)
    from collections import deque
    q = deque(sid for sid, d in in_deg.items() if d == 0)
    result: list[str] = []
    while q:
        sid = q.popleft()
        result.append(sid)
        for dep in deps_of.get(sid, []):
            in_deg[dep] -= 1
            if in_deg[dep] == 0:
                q.append(dep)
    result.extend(sid for sid in smap if sid not in result)  # cycle fallback
    return result


def _naive_load_count(plan: AnalyzerPlan) -> int:
    """Count model loads in naive sequential execution."""
    order = _topological_trace(plan)
    if not order:
        return 0
    smap = {s.id: s for s in plan.subtasks}
    loads = 1
    prev = smap[order[0]].assigned_model
    for sid in order[1:]:
        cur = smap[sid].assigned_model
        if cur != prev:
            loads += 1
            prev = cur
    return loads


def count_model_loads(schedule: ExecutionSchedule) -> int:
    if not schedule.batches:
        return 0
    load_count = 1
    for i in range(1, len(schedule.batches)):
        if schedule.batches[i].model != schedule.batches[i - 1].model:
            load_count += 1
    return load_count

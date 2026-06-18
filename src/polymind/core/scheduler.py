from __future__ import annotations

import logging
from collections import deque

from polymind.core.types import (
    AnalyzerPlan,
    DomainType,
    ExecutionSchedule,
    ExecutionStrategy,
    ModelBatch,
    RankStore,
    Subtask,
)

logger = logging.getLogger(__name__)


def _assign_models(
    plan: AnalyzerPlan,
    rank_store: RankStore | None,
    default_model: str,
    fallback_models: list[str] | None,
) -> None:
    for subtask in plan.subtasks:
        if subtask.assigned_model:
            continue

        if rank_store is not None:
            top = rank_store.top_for_domain(subtask.domain)
            if top is not None:
                subtask.assigned_model = top.model
                continue

            fallback_entries = rank_store.top_n_for_domain(
                subtask.domain, len(fallback_models or [])
            )
            for i, entry in enumerate(fallback_entries):
                try:
                    fallback_models[i]
                except IndexError:
                    break
                if entry.model != default_model:
                    subtask.assigned_model = fallback_models[i]
                    break

        subtask.assigned_model = default_model


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
        logger.warning(
            "Cycle detected: %d/%d tasks scheduled",
            len(ordered),
            len(plan.subtasks),
        )
        unscheduled = set(subtask_map.keys()) - set(ordered)
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

        remaining: list[str] = []
        while ready:
            remaining.append(ready.popleft())

        for candidate_id in remaining:
            if candidate_id in scheduled:
                continue
            candidate = subtask_map[candidate_id]
            if candidate.assigned_model == model:
                batch_ids.append(candidate_id)
                scheduled.add(candidate_id)
            else:
                ready.append(candidate_id)

        batches.append(ModelBatch(model=model, subtask_ids=batch_ids))

        for batch_id in batch_ids:
            for dep_of in dependents.get(batch_id, []):
                in_degree[dep_of] -= 1
                if in_degree[dep_of] == 0 and dep_of not in scheduled:
                    ready.append(dep_of)

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
) -> ExecutionSchedule:
    _assign_models(plan, rank_store, default_model, fallback_models)

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


def count_model_loads(schedule: ExecutionSchedule) -> int:
    if not schedule.batches:
        return 0
    load_count = 1
    for i in range(1, len(schedule.batches)):
        if schedule.batches[i].model != schedule.batches[i - 1].model:
            load_count += 1
    return load_count

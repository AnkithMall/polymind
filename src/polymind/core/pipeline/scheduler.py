"""Scheduler — groups tasks by model to minimize reloads, handles dependencies.

The scheduler builds a DAG of tasks, resolves dependencies, and groups
consecutive tasks that use the same model into TaskGroups for efficient execution.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from polymind.core.pipeline.types import Task, TaskGroup, TaskStatus


@dataclass
class SchedulePlan:
    """Execution plan from the scheduler."""

    groups: list[TaskGroup]
    total_tasks: int
    estimated_model_loads: int
    execution_order: list[list[str]]  # list of waves (tasks that can run in parallel)


def schedule_tasks(
    tasks: list[Task],
    assignments: dict[str, str],  # task_id → model_id
) -> SchedulePlan:
    """Create an execution plan that minimizes model loads.

    Strategy:
    1. Build dependency graph
    2. Topological sort to find execution waves
    3. Within each wave, group consecutive tasks with the same model
    4. Each group = one model load → multiple task executions
    """
    task_map = {t.id: t for t in tasks}

    # Apply assignments
    for task in tasks:
        if task.id in assignments:
            task.model_id = assignments[task.id]

    # Build dependency graph
    in_degree: dict[str, int] = {t.id: 0 for t in tasks}
    dependents: dict[str, list[str]] = defaultdict(list)

    for task in tasks:
        for dep_id in task.depends_on:
            if dep_id in task_map:
                dependents[dep_id].append(task.id)
                in_degree[task.id] = in_degree.get(task.id, 0) + 1

    # Topological sort — find execution waves (BFS)
    waves: list[list[str]] = []
    ready = [tid for tid, deg in in_degree.items() if deg == 0]

    while ready:
        # This wave = all currently ready tasks
        waves.append(list(ready))
        next_ready: list[str] = []

        for tid in ready:
            for dep_id in dependents[tid]:
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    next_ready.append(dep_id)

        ready = next_ready

    # Handle cycles — add remaining tasks as their own wave
    remaining = [
        t.id for t in tasks if t.status == TaskStatus.PENDING and not any(t.id in w for w in waves)
    ]
    if remaining:
        waves.append(remaining)

    # Within each wave, group by model_id
    groups: list[TaskGroup] = []
    model_load_count = 0
    last_model = ""

    for wave in waves:
        # Sort wave by model_id to maximize consecutive same-model tasks
        wave_tasks = [task_map[tid] for tid in wave if tid in task_map]
        wave_tasks.sort(key=lambda t: t.model_id)

        for task in wave_tasks:
            model_id = task.model_id or "unknown"

            if model_id != last_model:
                groups.append(TaskGroup(model_id=model_id, tasks=[task]))
                model_load_count += 1
                last_model = model_id
            else:
                # Same model as previous task — add to same group
                groups[-1].tasks.append(task)

    return SchedulePlan(
        groups=groups,
        total_tasks=len(tasks),
        estimated_model_loads=model_load_count,
        execution_order=waves,
    )


def get_ready_tasks(
    tasks: list[Task],
    completed_ids: set[str],
) -> list[Task]:
    """Get tasks whose dependencies are all completed."""
    ready: list[Task] = []

    for task in tasks:
        if task.status != TaskStatus.PENDING:
            continue

        deps_met = all(dep_id in completed_ids for dep_id in task.depends_on)
        if deps_met:
            ready.append(task)

    return ready


def mark_completed(task: Task, result: str, score: float = 0.0) -> None:
    """Mark a task as completed with its result."""
    task.status = TaskStatus.COMPLETED
    task.result = result
    task.score = score


def mark_failed(task: Task, error: str) -> None:
    """Mark a task as failed."""
    task.status = TaskStatus.FAILED
    task.error = error

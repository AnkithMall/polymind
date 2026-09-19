"""Pipeline types — task decomposition, scheduling, and execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class TaskType(StrEnum):
    """Types of tasks the pipeline can execute."""

    DECOMPOSE = "decompose"
    GENERATE = "generate"
    REGENERATE = "regenerate"
    REVIEW = "review"
    SYNTHESIZE = "synthesize"
    CUSTOM = "custom"


class TaskStatus(StrEnum):
    """Execution status of a task."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ModelRole(StrEnum):
    """Specialized roles for model selection."""

    DECOMPOSER = "decomposer"
    GENERATOR = "generator"
    REGENERATOR = "regenerator"
    JUDGE = "judge"
    GENERAL = "general"


@dataclass
class Task:
    """A single unit of work in the pipeline."""

    id: str
    prompt: str
    domain: str
    task_type: TaskType = TaskType.GENERATE

    # Model assignment
    model_id: str = ""
    model_role: ModelRole = ModelRole.GENERATOR

    # Dependencies — task IDs that must complete before this one
    depends_on: list[str] = field(default_factory=list)

    # Execution context
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    error: str = ""
    score: float = 0.0

    # Metadata
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "domain": self.domain,
            "task_type": self.task_type.value,
            "model_id": self.model_id,
            "model_role": self.model_role.value,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "result": self.result[:200] if self.result else "",
            "error": self.error,
            "score": self.score,
        }


@dataclass
class TaskGroup:
    """Group of tasks assigned to the same model (minimizes reloads)."""

    model_id: str
    tasks: list[Task] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "task_count": len(self.tasks),
            "task_ids": [t.id for t in self.tasks],
        }


@dataclass
class PipelineConfig:
    """Configuration for the pipeline."""

    # Model assignments (model_id → role)
    decomposer_model: str = ""
    generator_model: str = ""
    regenerator_model: str = ""
    judge_model: str = ""

    # Execution settings
    max_concurrent: int = 1
    timeout_seconds: int = 120
    max_retries: int = 1

    # Quality settings
    min_task_score: float = 0.5
    auto_select_models: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "decomposer_model": self.decomposer_model,
            "generator_model": self.generator_model,
            "regenerator_model": self.regenerator_model,
            "judge_model": self.judge_model,
            "max_concurrent": self.max_concurrent,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "min_task_score": self.min_task_score,
            "auto_select_models": self.auto_select_models,
        }


@dataclass
class PipelineResult:
    """Result of executing a full pipeline."""

    prompt: str
    response: str

    # Task breakdown
    tasks: list[Task] = field(default_factory=list)
    task_groups: list[TaskGroup] = field(default_factory=list)

    # Scores
    overall_score: float = 0.0
    domain_scores: dict[str, float] = field(default_factory=dict)

    # Metadata
    models_used: list[str] = field(default_factory=list)
    total_time_ms: float = 0.0
    model_loads: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "tasks": [t.to_dict() for t in self.tasks],
            "task_groups": [g.to_dict() for g in self.task_groups],
            "overall_score": round(self.overall_score, 2),
            "domain_scores": {k: round(v, 2) for k, v in self.domain_scores.items()},
            "models_used": self.models_used,
            "total_time_ms": round(self.total_time_ms, 1),
            "model_loads": self.model_loads,
        }

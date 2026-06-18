from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RankingMode(str, Enum):
    accuracy = "accuracy"
    cost = "cost"
    cost_effective = "cost_effective"


class ModelSource(str, Enum):
    local = "local"
    online = "online"
    all = "all"


class DomainType(str, Enum):
    code = "code"
    math = "math"
    reasoning = "reasoning"
    creative = "creative"
    research = "research"
    summarization = "summarization"
    translation = "translation"
    qa = "qa"
    general = "general"


ALL_DOMAINS = list(DomainType)


class Subtask(BaseModel):
    id: str
    domain: DomainType
    prompt: str
    depends_on: list[str] = Field(default_factory=list)
    assigned_model: str | None = None


class AnalyzerPlan(BaseModel):
    original_prompt: str
    subtasks: list[Subtask]


def _rank_key(entry: RankEntry, mode: RankingMode) -> float:
    if mode == RankingMode.cost:
        return -(entry.cost or 0.0)
    if mode == RankingMode.cost_effective:
        eff = entry.score / max(entry.cost or 0.001, 0.001)
        return eff
    return entry.score


class RankEntry(BaseModel):
    model: str
    domain: DomainType
    score: float = Field(ge=0.0, le=1.0)
    latency_ms: float | None = None
    cost: float | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


class RankStore(BaseModel):
    entries: list[RankEntry] = Field(default_factory=list)

    def top_for_domain(
        self,
        domain: DomainType,
        ranking_mode: RankingMode = RankingMode.accuracy,
    ) -> RankEntry | None:
        candidates = [e for e in self.entries if e.domain == domain]
        if not candidates:
            return None
        return max(candidates, key=lambda e: _rank_key(e, ranking_mode))

    def top_n_for_domain(
        self,
        domain: DomainType,
        n: int,
        ranking_mode: RankingMode = RankingMode.accuracy,
    ) -> list[RankEntry]:
        candidates = sorted(
            (e for e in self.entries if e.domain == domain),
            key=lambda e: _rank_key(e, ranking_mode),
            reverse=True,
        )
        return candidates[:n]

    def is_stale(self, max_age_days: int = 30) -> bool:
        if not self.entries:
            return True
        now = datetime.now()
        return any((now - e.timestamp).days > max_age_days for e in self.entries)


class ExecutionStrategy(str, Enum):
    model_aware = "model_aware"
    sequential = "sequential"
    parallel = "parallel"


class ModelBatch(BaseModel):
    model: str
    subtask_ids: list[str]


class ExecutionSchedule(BaseModel):
    strategy: ExecutionStrategy
    batches: list[ModelBatch]


class SubtaskResult(BaseModel):
    subtask_id: str
    model: str
    output: str
    latency_ms: float | None = None
    error: str | None = None
    token_count: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None


class PipelineResult(BaseModel):
    original_prompt: str
    subtask_results: list[SubtaskResult]
    synthesis: str | None = None
    total_latency_ms: float | None = None
    schedule: ExecutionSchedule | None = None

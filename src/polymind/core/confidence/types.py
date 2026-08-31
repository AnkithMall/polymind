"""Confidence scoring types and data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

EvaluationMethod = Literal[
    "exact_match",
    "keyword_match",
    "code_execution",
    "llm_judge",
    "hybrid",
]


@dataclass
class TestQuestion:
    """A single test question for evaluating a model."""

    id: str
    prompt: str
    expected: str
    evaluation: EvaluationMethod = "hybrid"
    keywords: list[str] = field(default_factory=list)
    code: str | None = None
    weight: float = 1.0
    explanation: str = ""

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "id": self.id,
            "prompt": self.prompt,
            "expected": self.expected,
            "evaluation": self.evaluation,
            "weight": self.weight,
        }
        if self.keywords:
            data["keywords"] = self.keywords
        if self.code:
            data["code"] = self.code
        if self.explanation:
            data["explanation"] = self.explanation
        return data


@dataclass
class TestSuite:
    """A collection of test questions for a specific sub-topic."""

    id: str
    name: str
    description: str
    difficulty: Literal["easy", "medium", "hard", "expert"]
    questions: list[TestQuestion] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "difficulty": self.difficulty,
            "questions": [q.to_dict() for q in self.questions],
        }


@dataclass
class Domain:
    """A domain/category for confidence scoring."""

    id: str
    name: str
    description: str
    custom: bool = False
    suites: list[TestSuite] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "custom": self.custom,
            "suites": [s.to_dict() for s in self.suites],
        }


@dataclass
class SuiteScore:
    """Score for a single test suite."""

    suite_id: str
    score: float
    passed: int
    total: int
    details: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "suite_id": self.suite_id,
            "score": round(self.score, 2),
            "passed": self.passed,
            "total": self.total,
            "details": self.details,
        }


@dataclass
class DomainScore:
    """Aggregated score for a domain."""

    domain_id: str
    overall: float
    suites: dict[str, SuiteScore] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "domain_id": self.domain_id,
            "overall": round(self.overall, 2),
            "suites": {k: v.to_dict() for k, v in self.suites.items()},
        }


@dataclass
class ModelConfidence:
    """All confidence scores for a single model."""

    model_id: str
    domains: dict[str, DomainScore] = field(default_factory=dict)
    best_domain: str = ""
    overall_score: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "domains": {k: v.to_dict() for k, v in self.domains.items()},
            "best_domain": self.best_domain,
            "overall_score": round(self.overall_score, 2),
        }

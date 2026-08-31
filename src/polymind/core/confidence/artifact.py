"""Confidence artifact persistence.

Handles loading and saving .polymind/confidence.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from polymind.core.confidence.suites import get_all_domains, get_domain
from polymind.core.confidence.types import (
    Domain,
    DomainScore,
    ModelConfidence,
    SuiteScore,
    TestQuestion,
    TestSuite,
)
from polymind.core.paths import artifact_dir


def confidence_path() -> Path:
    """Path to the confidence artifact file."""
    return artifact_dir() / "confidence.yaml"


def custom_domains_path() -> Path:
    """Path to custom domains directory."""
    return artifact_dir() / "domains"


def load_confidence(
    path: Path | None = None,
) -> dict[str, ModelConfidence]:
    """Load all model confidence scores from confidence.yaml."""
    if path is None:
        path = confidence_path()

    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    models_data = data.get("scores", {})
    result: dict[str, ModelConfidence] = {}

    for model_id, model_data in models_data.items():
        domains: dict[str, DomainScore] = {}
        domains_data = model_data.get("domains", {})

        for domain_id, domain_data in domains_data.items():
            suites: dict[str, SuiteScore] = {}
            suites_data = domain_data.get("suites", {})

            for suite_id, suite_data in suites_data.items():
                suites[suite_id] = SuiteScore(
                    suite_id=suite_data.get("suite_id", suite_id),
                    score=suite_data.get("score", 0.0),
                    passed=suite_data.get("passed", 0),
                    total=suite_data.get("total", 0),
                    details=suite_data.get("details", {}),
                )

            domains[domain_id] = DomainScore(
                domain_id=domain_id,
                overall=domain_data.get("overall", 0.0),
                suites=suites,
            )

        result[model_id] = ModelConfidence(
            model_id=model_id,
            domains=domains,
            best_domain=model_data.get("best_domain", ""),
            overall_score=model_data.get("overall_score", 0.0),
        )

    return result


def save_confidence(
    scores: dict[str, ModelConfidence],
    path: Path | None = None,
) -> Path:
    """Save all model confidence scores to confidence.yaml."""
    if path is None:
        path = confidence_path()

    path.parent.mkdir(parents=True, exist_ok=True)

    artifact: dict[str, Any] = {
        "version": 1,
        "scores": {},
    }

    for model_id, conf in scores.items():
        artifact["scores"][model_id] = conf.to_dict()

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(artifact, f, sort_keys=False, default_flow_style=False)

    return path


def load_all_domains() -> list[Domain]:
    """Load all domains (predefined + custom)."""
    domains = list(get_all_domains())

    # Load custom domains from artifacts
    custom_dir = custom_domains_path()
    if custom_dir.exists():
        for yaml_file in sorted(custom_dir.glob("*.yaml")):
            domain = _load_custom_domain(yaml_file)
            if domain is not None:
                domains.append(domain)

    return domains


def load_domain_by_id(domain_id: str) -> Domain | None:
    """Load a specific domain by ID."""
    # Check predefined first
    predefined = get_domain(domain_id)
    if predefined is not None:
        return predefined

    # Check custom
    custom_file = custom_domains_path() / f"{domain_id}.yaml"
    if custom_file.exists():
        return _load_custom_domain(custom_file)

    return None


def save_custom_domain(domain: Domain) -> Path:
    """Save a custom domain to artifacts."""
    custom_dir = custom_domains_path()
    custom_dir.mkdir(parents=True, exist_ok=True)

    domain.custom = True
    path = custom_dir / f"{domain.id}.yaml"

    data = domain.to_dict()
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

    return path


def delete_custom_domain(domain_id: str) -> bool:
    """Delete a custom domain."""
    custom_file = custom_domains_path() / f"{domain_id}.yaml"
    if custom_file.exists():
        custom_file.unlink()
        return True
    return False


def _load_custom_domain(path: Path) -> Domain | None:
    """Load a custom domain from a YAML file."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        suites: list[TestSuite] = []
        for suite_data in data.get("suites", []):
            questions: list[TestQuestion] = []
            for q_data in suite_data.get("questions", []):
                questions.append(
                    TestQuestion(
                        id=q_data.get("id", ""),
                        prompt=q_data.get("prompt", ""),
                        expected=q_data.get("expected", ""),
                        evaluation=q_data.get("evaluation", "hybrid"),
                        keywords=q_data.get("keywords", []),
                        code=q_data.get("code"),
                        weight=q_data.get("weight", 1.0),
                        explanation=q_data.get("explanation", ""),
                    )
                )
            suites.append(
                TestSuite(
                    id=suite_data.get("id", ""),
                    name=suite_data.get("name", ""),
                    description=suite_data.get("description", ""),
                    difficulty=suite_data.get("difficulty", "medium"),
                    questions=questions,
                )
            )

        return Domain(
            id=data.get("id", path.stem),
            name=data.get("name", path.stem),
            description=data.get("description", ""),
            custom=data.get("custom", True),
            suites=suites,
        )
    except Exception:
        return None

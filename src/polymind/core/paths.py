"""
Central location resolution for Polymind artifacts and models.

Defaults:
    artifacts: ./.polymind/
    models:    ./.polymind/models/

Both can be overridden with environment variables:
    POLYMIND_ARTIFACT_DIR
    POLYMIND_MODEL_DIR
"""

from __future__ import annotations

import os
from pathlib import Path

ARTIFACT_DIR_ENV = "POLYMIND_ARTIFACT_DIR"
MODEL_DIR_ENV = "POLYMIND_MODEL_DIR"

DEFAULT_ARTIFACT_DIR = Path(".polymind")


def artifact_dir() -> Path:
    """Directory where Polymind artifact files are stored."""
    override = os.environ.get(ARTIFACT_DIR_ENV)
    if override:
        return Path(override)
    return DEFAULT_ARTIFACT_DIR


def model_dir() -> Path:
    """Directory where downloaded GGUF models are stored."""
    override = os.environ.get(MODEL_DIR_ENV)
    if override:
        return Path(override)
    return artifact_dir() / "models"


def hardware_path() -> Path:
    return artifact_dir() / "hardware.yaml"


def runtime_path() -> Path:
    return artifact_dir() / "runtime.yaml"


def registry_path() -> Path:
    return artifact_dir() / "registry.yaml"


def confidence_path() -> Path:
    return artifact_dir() / "confidence.yaml"


def custom_domains_dir() -> Path:
    return artifact_dir() / "domains"


def logs_dir() -> Path:
    """Directory for pipeline log files."""
    d = artifact_dir() / ".logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def config_path() -> Path:
    """Path to polymind.yaml configuration file."""
    return artifact_dir() / "polymind.yaml"

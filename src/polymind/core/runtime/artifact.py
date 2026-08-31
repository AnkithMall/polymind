from pathlib import Path
from typing import Any

import yaml

from polymind.core.paths import runtime_path
from polymind.core.runtime.types import RuntimeConfig


def load_runtime_config(
    model_id: str,
    path: Path | None = None,
) -> RuntimeConfig | None:
    """
    Load runtime configuration for a model from runtime.yaml.

    Returns None if no config exists for this model.
    """
    if path is None:
        path = runtime_path()

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    models = data.get("models", {})
    model_data = models.get(model_id)

    if not model_data:
        return None

    return RuntimeConfig(
        model_id=model_data.get("model_id", model_id),
        gpu_layers=model_data.get("gpu_layers", -1),
        threads=model_data.get("threads", 4),
        context_size=model_data.get("context_size", 4096),
        batch_size=model_data.get("batch_size", 512),
        benchmark=model_data.get("benchmark", {}),
    )


def write_runtime_config(
    config: RuntimeConfig,
    path: Path | None = None,
) -> Path:
    if path is None:
        path = runtime_path()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    models: dict[str, dict[str, Any]] = {}

    artifact: dict[str, Any] = {
        "version": 1,
        "models": models,
    }

    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            existing = yaml.safe_load(file) or {}

        artifact["version"] = existing.get(
            "version",
            1,
        )

        existing_models = existing.get(
            "models",
            {},
        )

        if isinstance(existing_models, dict):
            models.update(existing_models)

    models[config.model_id] = config.to_dict()

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            artifact,
            file,
            sort_keys=False,
            default_flow_style=False,
        )

    return path

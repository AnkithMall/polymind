from pathlib import Path
from typing import Any

import yaml

from polymind.core.runtime.types import RuntimeConfig


DEFAULT_ARTIFACT_PATH = Path(".polymind/runtime.yaml")


def write_runtime_config(
    config: RuntimeConfig,
    path: Path = DEFAULT_ARTIFACT_PATH,
) -> Path:
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

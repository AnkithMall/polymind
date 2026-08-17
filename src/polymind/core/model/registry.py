from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


DEFAULT_MODEL_DIR = (
    Path.home() / ".cache" / "polymind" / "models"
)

DEFAULT_REGISTRY_PATH = (
    Path.home() / ".cache" / "polymind" / "registry.yaml"
)


@dataclass
class InstalledModel:
    id: int
    repo_id: str
    filename: str
    local_path: str
    size_bytes: int
    quantization: str | None


class ModelRegistry:
    def __init__(
        self,
        path: Path = DEFAULT_REGISTRY_PATH,
    ) -> None:
        self.path = path

    def load(self) -> list[InstalledModel]:
        if not self.path.exists():
            return []

        with self.path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = yaml.safe_load(file) or {}

        models = data.get("models", [])

        return [
            InstalledModel(**model)
            for model in models
        ]

    def save(
        self,
        models: list[InstalledModel],
    ) -> None:
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "version": 1,
            "models": [
                asdict(model)
                for model in models
            ],
        }

        with self.path.open(
            "w",
            encoding="utf-8",
        ) as file:
            yaml.safe_dump(
                data,
                file,
                sort_keys=False,
            )

    def add(
        self,
        repo_id: str,
        filename: str,
        local_path: Path,
        size_bytes: int,
        quantization: str | None,
    ) -> InstalledModel:
        models = self.load()

        # Don't register the same repository/file twice.
        for model in models:
            if (
                model.repo_id == repo_id
                and model.filename == filename
            ):
                model.local_path = str(local_path)
                model.size_bytes = size_bytes
                model.quantization = quantization

                self.save(models)

                return model

        next_id = max(
            (model.id for model in models),
            default=0,
        ) + 1

        model = InstalledModel(
            id=next_id,
            repo_id=repo_id,
            filename=filename,
            local_path=str(local_path),
            size_bytes=size_bytes,
            quantization=quantization,
        )

        models.append(model)

        self.save(models)

        return model

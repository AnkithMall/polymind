from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
    field,
)
from pathlib import Path

import yaml

from polymind.core.model.hf import is_text_model
from polymind.core.paths import (
    model_dir,
    registry_path,
)


@dataclass
class InstalledModel:
    id: int
    repo_id: str
    filename: str
    local_path: str
    size_bytes: int
    quantization: str | None


@dataclass
class ScanReport:
    """Result of reconciling the registry with files on disk."""

    added: list[InstalledModel] = field(default_factory=list)
    fixed: list[InstalledModel] = field(default_factory=list)
    duplicates_removed: int = 0
    missing: list[InstalledModel] = field(default_factory=list)


class ModelRegistry:
    def __init__(
        self,
        path: Path | None = None,
    ) -> None:
        self.path = path if path is not None else registry_path()

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
            InstalledModel(**model) for model in models if is_text_model(model.get("filename", ""))
        ]

    def find_by_filename(
        self,
        filename: str,
    ) -> InstalledModel | None:
        """Find a registered model by filename."""
        for model in self.load():
            if model.filename == filename:
                return model
        return None

    def find_by_repo_and_filename(
        self,
        repo_id: str,
        filename: str,
    ) -> InstalledModel | None:
        """Find a registered model by repository and filename."""
        for model in self.load():
            if model.repo_id == repo_id and model.filename == filename:
                return model
        return None

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
            "models": [asdict(model) for model in models],
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
            if model.repo_id == repo_id and model.filename == filename:
                model.local_path = str(local_path)
                model.size_bytes = size_bytes
                model.quantization = quantization

                self.save(models)

                return model

        next_id = (
            max(
                (model.id for model in models),
                default=0,
            )
            + 1
        )

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

    def remove(
        self,
        model_id: int,
    ) -> bool:
        """Remove a model from the registry by ID.

        Returns True if the model was found and removed.
        """
        models = self.load()
        original_count = len(models)

        models = [m for m in models if m.id != model_id]

        if len(models) == original_count:
            return False

        self.save(models)
        return True

    def scan(
        self,
        directory: Path | None = None,
    ) -> ScanReport:
        """
        Scan a directory for GGUF files and reconcile the
        registry with what is actually on disk.

        - Registers GGUF files not yet in the registry
          (e.g. manually downloaded models).
        - Fixes registry entries whose recorded path no
          longer exists when a matching file is found.
        - Removes duplicate registry entries pointing at
          the same file.

        Models whose files can no longer be found anywhere
        are reported as missing, but kept in the registry.
        """
        from polymind.core.model.hf import (
            detect_quantization,
        )

        if directory is None:
            directory = model_dir()

        report = ScanReport()
        models = self.load()

        exclude_dirs = {
            ".venv",
            ".git",
            "__pycache__",
            "node_modules",
            ".cache",
        }

        files = (
            sorted(
                f
                for f in directory.rglob("*.gguf")
                if not any(part in exclude_dirs for part in f.relative_to(directory).parts[:-1])
            )
            if directory.exists()
            else []
        )

        # -------------------------------------------------
        # Remove duplicate registry entries that point to
        # the same file on disk.
        # -------------------------------------------------

        seen_paths: dict[str, InstalledModel] = {}
        deduped: list[InstalledModel] = []

        for model in models:
            key = str(Path(model.local_path).resolve())

            existing = seen_paths.get(key)

            if existing is None:
                seen_paths[key] = model
                deduped.append(model)
                continue

            # Keep the entry with the smallest id so
            # references remain stable.
            keep, drop = (existing, model) if existing.id <= model.id else (model, existing)

            deduped.remove(existing)
            seen_paths[key] = keep
            deduped.append(keep)
            report.duplicates_removed += 1

        models = deduped

        # -------------------------------------------------
        # Fix entries whose recorded path is missing when
        # a file with the same name exists on disk.
        # -------------------------------------------------

        by_name: dict[str, list[Path]] = {}

        for file in files:
            by_name.setdefault(file.name, []).append(file)

        registered_paths = {str(Path(m.local_path).resolve()) for m in models}

        for model in models:
            recorded = Path(model.local_path)

            if recorded.exists():
                continue

            candidates = [
                path
                for path in by_name.get(recorded.name, [])
                if str(path.resolve()) not in registered_paths
            ]

            if len(candidates) == 1:
                fixed_path = candidates[0]

                model.local_path = str(fixed_path)
                model.size_bytes = fixed_path.stat().st_size

                quantization = detect_quantization(fixed_path.name)

                if quantization:
                    model.quantization = quantization

                registered_paths.add(str(fixed_path.resolve()))
                report.fixed.append(model)
            else:
                report.missing.append(model)

        # -------------------------------------------------
        # Register files that are not in the registry yet.
        # Skip non-text models (projection adapters, etc.)
        # -------------------------------------------------

        for file in files:
            if not is_text_model(file.name):
                continue

            if (
                file.resolve() in {Path(m.local_path).resolve() for m in models}
                or str(file.resolve()) in registered_paths
            ):
                continue

            next_id = (
                max(
                    (m.id for m in models),
                    default=0,
                )
                + 1
            )

            model = InstalledModel(
                id=next_id,
                repo_id="manual/unknown",
                filename=file.name,
                local_path=str(file),
                size_bytes=file.stat().st_size,
                quantization=detect_quantization(file.name),
            )

            models.append(model)
            registered_paths.add(str(file.resolve()))
            report.added.append(model)

        if report.added or report.fixed or report.duplicates_removed:
            self.save(models)

        return report

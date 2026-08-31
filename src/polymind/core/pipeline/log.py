"""Pipeline logging — writes verbose output to .polymind/.logs/ with rotation and cleanup."""

from __future__ import annotations

import logging
import logging.handlers
import time
from pathlib import Path

import yaml

from polymind.core.paths import artifact_dir

# ── Default config ───────────────────────────────────────────
_DEFAULT_LOG_CONFIG = {
    "enabled": True,
    "max_bytes": 5 * 1024 * 1024,  # 5 MB per log file
    "backup_count": 3,              # Keep 3 rotated files
    "max_age_days": 7,              # Delete logs older than 7 days
    "max_total_files": 10,          # Max total log files in directory
    "max_total_size_mb": 50,        # Max total size of all logs
}


def _logs_dir() -> Path:
    """Get or create the logs directory."""
    d = artifact_dir() / ".logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_log_config() -> dict:
    """Load log config from polymind.yaml, falling back to defaults."""
    config_path = artifact_dir() / "polymind.yaml"
    cfg = dict(_DEFAULT_LOG_CONFIG)

    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            log_cfg = data.get("logging", {})
            cfg.update({k: v for k, v in log_cfg.items() if k in _DEFAULT_LOG_CONFIG})
        except Exception:
            pass

    return cfg


def _cleanup_old_logs(cfg: dict) -> None:
    """Delete logs exceeding age, count, or size limits."""
    logs_dir = _logs_dir()
    log_files = sorted(logs_dir.glob("pipeline_*.log"), key=lambda p: p.stat().st_mtime)

    # ── Max age ──────────────────────────────────────────────
    max_age = cfg.get("max_age_days", 7)
    if max_age > 0:
        cutoff = time.time() - (max_age * 86400)
        for f in log_files:
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    log_files.remove(f)
            except OSError:
                pass

    # Re-scan after deletions
    log_files = sorted(logs_dir.glob("pipeline_*.log"), key=lambda p: p.stat().st_mtime)

    # ── Max file count ───────────────────────────────────────
    max_files = cfg.get("max_total_files", 10)
    if max_files > 0 and len(log_files) > max_files:
        for f in log_files[: len(log_files) - max_files]:
            try:
                f.unlink()
            except OSError:
                pass
        log_files = sorted(logs_dir.glob("pipeline_*.log"), key=lambda p: p.stat().st_mtime)

    # ── Max total size ───────────────────────────────────────
    max_size_mb = cfg.get("max_total_size_mb", 50)
    if max_size_mb > 0:
        total = sum(f.stat().st_size for f in log_files)
        max_bytes = max_size_mb * 1024 * 1024
        while total > max_bytes and log_files:
            oldest = log_files.pop(0)
            try:
                total -= oldest.stat().st_size
                oldest.unlink()
            except OSError:
                pass


def get_pipeline_logger(name: str = "pipeline") -> logging.Logger:
    """Get a logger that writes to .polymind/.logs/ with rotation."""
    cfg = _load_log_config()

    if not cfg.get("enabled", True):
        logger = logging.getLogger(f"polymind.{name}")
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
        return logger

    # Cleanup before creating new log
    _cleanup_old_logs(cfg)

    logs_dir = _logs_dir()
    log_file = logs_dir / f"{name}.log"

    logger = logging.getLogger(f"polymind.{name}")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = logging.handlers.RotatingFileHandler(
            str(log_file),
            maxBytes=cfg.get("max_bytes", 5 * 1024 * 1024),
            backupCount=cfg.get("backup_count", 3),
            encoding="utf-8",
        )
        handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class PipelineLogWriter:
    """Writes structured pipeline events to the log file.

    Used by non-verbose mode to capture everything to disk.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or get_pipeline_logger()
        self._task_count = 0

    def log_event(self, event: str, task) -> None:
        """Log a pipeline event."""
        meta = getattr(task, "metadata", {})

        if event == "artifact_check":
            self.logger.info("=== Artifact Detection ===")
            for name, info in meta.items():
                self.logger.info("  %s: %s", name, info)

        elif event == "analyze":
            is_simple = meta.get("is_simple", "?")
            word_count = meta.get("word_count", "?")
            self.logger.info("=== Complexity Analysis ===")
            self.logger.info("  Words: %s, Simple: %s", word_count, is_simple)

        elif event == "decompose_start":
            self.logger.info("=== Decomposition ===")
            self.logger.info("  Loading decomposer model...")

        elif event == "decompose_done":
            tasks = meta.get("tasks", [])
            self.logger.info("  Decomposed into %d task(s)", len(tasks))
            for t in tasks:
                self.logger.info(
                    "    %s | %s | %s",
                    t.get("id", "?"),
                    t.get("domain", "?"),
                    t.get("prompt", "?")[:60],
                )

        elif event == "assign_start":
            self.logger.info("=== Model Assignment ===")

        elif event == "assign_done":
            assignments = meta.get("assignments", {})
            for tid, a in assignments.items():
                self.logger.info(
                    "  %s → %s (conf=%s, reason=%s)",
                    tid,
                    a.get("model_id", "?"),
                    a.get("confidence", "?"),
                    a.get("reason", "?"),
                )

        elif event == "schedule":
            groups = meta.get("groups", [])
            loads = meta.get("model_loads", "?")
            self.logger.info("=== Execution Plan ===")
            self.logger.info("  Model loads: %s, Groups: %d", loads, len(groups))
            for i, g in enumerate(groups, 1):
                self.logger.info(
                    "  Group %d: %s → %s",
                    i,
                    g.get("model_id", "?"),
                    ", ".join(g.get("task_ids", [])),
                )

        elif event == "model_load":
            self.logger.info(
                "  Loading model: %s (%s, gpu=%s, threads=%s, ctx=%s)",
                meta.get("model_id", "?"),
                meta.get("model_file", "?"),
                meta.get("gpu_layers", "?"),
                meta.get("threads", "?"),
                meta.get("context_size", "?"),
            )

        elif event == "model_loaded":
            self.logger.info("  ✓ Model loaded: %s", meta.get("model_id", "?"))

        elif event == "model_load_failed":
            self.logger.error(
                "  ✗ Failed to load %s: %s",
                meta.get("model_id", "?"),
                meta.get("reason", "?"),
            )

        elif event == "task_start":
            self.logger.info(
                "  ▶ %s [%s] %s",
                task.id,
                task.domain,
                task.prompt[:70],
            )

        elif event == "task_done":
            if task.status.value == "completed":
                self.logger.info(
                    "    ✓ %s (score=%.0f%%) %s",
                    task.id,
                    task.score * 100,
                    (task.result or "")[:80],
                )
            elif task.status.value == "failed":
                self.logger.error("    ✗ %s: %s", task.id, task.error)

        elif event == "regenerate_start":
            self.logger.info("=== Regeneration ===")

        elif event == "regenerate_done":
            self.logger.info("  ✓ Final response generated")

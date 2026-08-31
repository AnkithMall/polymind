"""Polymind configuration — reads/writes polymind.yaml in .polymind/."""

from __future__ import annotations

from pathlib import Path

import yaml

from polymind.core.paths import config_path

# ── Default configuration ────────────────────────────────────
DEFAULTS = {
    "logging": {
        "enabled": True,
        "max_bytes": 5 * 1024 * 1024,  # 5 MB per log file
        "backup_count": 3,              # Keep 3 rotated files
        "max_age_days": 7,              # Delete logs older than N days
        "max_total_files": 10,          # Max total log files in .logs/
        "max_total_size_mb": 50,        # Max total size of all logs combined
    },
    "pipeline": {
        "max_concurrent": 1,
        "timeout_seconds": 120,
        "max_retries": 1,
        "min_task_score": 0.5,
    },
}


def load_config() -> dict:
    """Load polymind.yaml, merging with defaults for missing keys."""
    path = config_path()
    cfg = dict(DEFAULTS)

    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            # Deep merge
            for section, values in data.items():
                if isinstance(values, dict) and section in cfg:
                    cfg[section] = {**cfg[section], **values}
                else:
                    cfg[section] = values
        except Exception:
            pass

    return cfg


def save_config(cfg: dict) -> Path:
    """Write polymind.yaml."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    return path


def get_logging_config() -> dict:
    """Get just the logging section of the config."""
    return load_config().get("logging", DEFAULTS["logging"])


def update_logging_config(**kwargs) -> dict:
    """Update logging config keys and save."""
    cfg = load_config()
    logging_cfg = cfg.get("logging", {})
    logging_cfg.update(kwargs)
    cfg["logging"] = logging_cfg
    save_config(cfg)
    return logging_cfg

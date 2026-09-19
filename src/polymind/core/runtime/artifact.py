"""Runtime artifact persistence.

Handles loading and saving runtime.yaml with backward compatibility.

Legacy format (version 1):
    models:
      '1':
        model_id: '1'
        gpu_layers: 0
        threads: 5
        context_size: 2048
        batch_size: 256

New format (version 2):
    version: 2
    models:
      '2':
        model_id: '2'
        hardware_fingerprint: "abc123"
        model_fingerprint: "def456"
        workload: "default"
        placement: {backend: cuda, devices: [0], split_mode: none}
        gpu_layers: 12
        threads: 6
        context_size: 2048
        batch_size: 128
        benchmark: {generation_tokens_per_sec: 11.9, ...}
        validation: {status: ready, successful_runs: 3, ...}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from polymind.core.paths import runtime_path
from polymind.core.runtime.types import (
    BenchmarkMetrics,
    Placement,
    RuntimeConfig,
    RuntimeProfile,
    SplitMode,
    ValidationInfo,
    ValidationStatus,
)


def load_runtime_config(
    model_id: str,
    path: Path | None = None,
) -> RuntimeConfig | None:
    """Load runtime configuration for a model from runtime.yaml.

    Returns None if no config exists for this model.
    Handles both legacy and new format.
    """
    if path is None:
        path = runtime_path()

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    models = data.get("models", {})

    # BUG-003 fix: try string key first, then integer key
    model_data = models.get(model_id)
    if model_data is None and model_id.isdigit():
        model_data = models.get(int(model_id))
    if model_data is None:
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


def load_runtime_profile(
    model_id: str,
    path: Path | None = None,
) -> RuntimeProfile | None:
    """Load a full RuntimeProfile for a model.

    Handles both legacy (v1) and new (v2) formats. Legacy profiles
    are returned with status=STALE so they get revalidated.
    """
    if path is None:
        path = runtime_path()

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    version = data.get("version", 1)
    models = data.get("models", {})

    model_data = models.get(model_id)
    if model_data is None and model_id.isdigit():
        model_data = models.get(int(model_id))
    if model_data is None:
        return None

    if version >= 2 and "placement" in model_data:
        return _parse_v2_profile(model_id, model_data)
    else:
        return _parse_v1_as_profile(model_id, model_data)


def _parse_v2_profile(model_id: str, data: dict[str, Any]) -> RuntimeProfile:
    """Parse a v2 runtime profile."""
    placement_data = data.get("placement", {})
    benchmark_data = data.get("benchmark", {})
    validation_data = data.get("validation", {})

    placement = Placement(
        backend=placement_data.get("backend", "cpu"),
        devices=placement_data.get("devices", []),
        split_mode=SplitMode(placement_data.get("split_mode", "none")),
        main_gpu=placement_data.get("main_gpu"),
        tensor_split=placement_data.get("tensor_split"),
    )

    benchmark = BenchmarkMetrics(
        load_time_ms=benchmark_data.get("load_time_ms", 0),
        prompt_tokens_per_sec=benchmark_data.get("prompt_tokens_per_sec", 0),
        generation_tokens_per_sec=benchmark_data.get("generation_tokens_per_sec", 0),
        time_to_first_token_ms=benchmark_data.get("time_to_first_token_ms", 0),
        total_latency_ms=benchmark_data.get("total_latency_ms", 0),
        peak_vram_mb=benchmark_data.get("peak_vram_mb", 0),
        peak_ram_mb=benchmark_data.get("peak_ram_mb", 0),
        warmup_time_ms=benchmark_data.get("warmup_time_ms", 0),
        stability=benchmark_data.get("stability", 0),
        runs=benchmark_data.get("runs", 0),
        benchmark_timestamp=benchmark_data.get("benchmark_timestamp", ""),
    )

    validation = ValidationInfo(
        status=ValidationStatus(validation_data.get("status", "unknown")),
        successful_runs=validation_data.get("successful_runs", 0),
        failed_runs=validation_data.get("failed_runs", 0),
        last_validated=validation_data.get("last_validated", ""),
        failure_reasons=validation_data.get("failure_reasons", []),
        runtime_version=validation_data.get("runtime_version", ""),
    )

    return RuntimeProfile(
        model_id=model_id,
        hardware_fingerprint=data.get("hardware_fingerprint", ""),
        model_fingerprint=data.get("model_fingerprint", ""),
        workload=data.get("workload", "default"),
        placement=placement,
        gpu_layers=data.get("gpu_layers", -1),
        threads=data.get("threads", 4),
        threads_batch=data.get("threads_batch", data.get("threads", 4)),
        context_size=data.get("context_size", 4096),
        batch_size=data.get("batch_size", 512),
        ubatch_size=data.get("ubatch_size", 64),
        flash_attention=data.get("flash_attention"),
        kv_cache_type=data.get("kv_cache_type"),
        benchmark=benchmark,
        validation=validation,
        estimated_memory_mb=data.get("estimated_memory_mb", 0),
        actual_peak_memory_mb=data.get("actual_peak_memory_mb", 0),
        safety_margin_mb=data.get("safety_margin_mb", 0),
    )


def _parse_v1_as_profile(model_id: str, data: dict[str, Any]) -> RuntimeProfile:
    """Convert legacy v1 config to a RuntimeProfile with STALE status."""
    return RuntimeProfile(
        model_id=model_id,
        hardware_fingerprint="",
        model_fingerprint="",
        workload="default",
        placement=Placement(
            backend="cuda" if data.get("gpu_layers", 0) > 0 else "cpu",
            devices=[0] if data.get("gpu_layers", 0) > 0 else [],
        ),
        gpu_layers=data.get("gpu_layers", -1),
        threads=data.get("threads", 4),
        threads_batch=data.get("threads", 4),
        context_size=data.get("context_size", 4096),
        batch_size=data.get("batch_size", 512),
        benchmark=BenchmarkMetrics(
            generation_tokens_per_sec=data.get("benchmark", {}).get("generation_tps", 0),
            prompt_tokens_per_sec=data.get("benchmark", {}).get("prompt_tps", 0),
            runs=data.get("benchmark", {}).get("runs", 0),
        ),
        validation=ValidationInfo(
            status=ValidationStatus.STALE,
            successful_runs=0,
            last_validated="legacy",
        ),
    )


def write_runtime_config(
    config: RuntimeConfig,
    path: Path | None = None,
) -> Path:
    """Write a RuntimeConfig (backward compatible)."""
    if path is None:
        path = runtime_path()

    path.parent.mkdir(parents=True, exist_ok=True)

    models: dict[str, dict[str, Any]] = {}
    artifact: dict[str, Any] = {"version": 1, "models": models}

    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            existing = yaml.safe_load(file) or {}
        artifact["version"] = existing.get("version", 1)
        existing_models = existing.get("models", {})
        if isinstance(existing_models, dict):
            for k, v in existing_models.items():
                models[str(k)] = v

    models[str(config.model_id)] = config.to_dict()

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(artifact, file, sort_keys=False, default_flow_style=False)

    return path


def write_runtime_profile(
    profile: RuntimeProfile,
    path: Path | None = None,
) -> Path:
    """Write a full RuntimeProfile to runtime.yaml.

    Preserves other model profiles when updating one model.
    """
    if path is None:
        path = runtime_path()

    path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing data to preserve other models
    models: dict[str, dict[str, Any]] = {}
    artifact: dict[str, Any] = {"version": 2, "models": models}

    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            existing = yaml.safe_load(file) or {}

        artifact["version"] = max(existing.get("version", 1), 2)
        existing_models = existing.get("models", {})
        if isinstance(existing_models, dict):
            for k, v in existing_models.items():
                models[str(k)] = v

    # Write the profile (preserving other models)
    models[str(profile.model_id)] = profile.to_dict()

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(artifact, file, sort_keys=False, default_flow_style=False)

    return path


def validate_profile_for_current_hardware(
    profile: RuntimeProfile,
    current_hw_fp_hash: str,
) -> tuple[bool, str]:
    """Check if a profile was built for the current hardware.

    Returns (is_valid, reason).
    """
    if not profile.hardware_fingerprint:
        return True, "No hardware fingerprint (legacy profile)"

    if profile.hardware_fingerprint == current_hw_fp_hash:
        return True, "Hardware matches"

    return False, (
        f"Hardware mismatch: profile was built for {profile.hardware_fingerprint}, "
        f"current hardware is {current_hw_fp_hash}"
    )


def validate_profile_for_current_model(
    profile: RuntimeProfile,
    current_model_fp_hash: str,
) -> tuple[bool, str]:
    """Check if a profile was built for the current model build.

    Returns (is_valid, reason).
    """
    if not profile.model_fingerprint:
        return True, "No model fingerprint (legacy profile)"

    if profile.model_fingerprint == current_model_fp_hash:
        return True, "Model matches"

    return False, (
        f"Model mismatch: profile was built for {profile.model_fingerprint}, "
        f"current model is {current_model_fp_hash}"
    )

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field

from polymind.core.types import ExecutionStrategy

logger = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    total_ram_gb: float = 0.0
    available_ram_gb: float = 0.0
    vram_gb: float = 0.0
    cpu_cores: int = 0
    has_nvidia_gpu: bool = False
    has_metal: bool = False
    ollama_models_loaded: list[str] = field(default_factory=list)

    @property
    def recommended_strategy(self) -> ExecutionStrategy:
        if self.vram_gb > 0 and self.vram_gb < 8:
            return ExecutionStrategy.sequential
        if self.total_ram_gb > 0 and self.total_ram_gb < 16:
            return ExecutionStrategy.sequential
        return ExecutionStrategy.model_aware

    @property
    def summary(self) -> str:
        lines = [
            f"RAM: {self.total_ram_gb:.1f} GB total, {self.available_ram_gb:.1f} GB available",
            f"CPU cores: {self.cpu_cores}",
        ]
        if self.vram_gb > 0:
            lines.append(f"VRAM: {self.vram_gb:.1f} GB")
        if self.has_nvidia_gpu:
            lines.append("GPU: NVIDIA (detected)")
        if self.has_metal:
            lines.append("GPU: Metal (detected)")
        if self.ollama_models_loaded:
            lines.append(
                f"Ollama models loaded: {', '.join(self.ollama_models_loaded)}"
            )
        lines.append(f"Recommended strategy: {self.recommended_strategy.value}")
        return "\n".join(lines)


def _get_ram_info() -> tuple[float, float]:
    try:
        import psutil

        mem = psutil.virtual_memory()
        return mem.total / (1024**3), mem.available / (1024**3)
    except ImportError:
        pass

    try:
        with open("/proc/meminfo") as f:
            data = f.read()
        total_kb = 0
        available_kb = 0
        for line in data.split("\n"):
            if line.startswith("MemTotal:"):
                total_kb = float(line.split()[1])
            elif line.startswith("MemAvailable:"):
                available_kb = float(line.split()[1])
        return total_kb / (1024**2), available_kb / (1024**2)
    except FileNotFoundError:
        pass

    return 0.0, 0.0


def _get_nvidia_vram() -> float:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return 0.0
    try:
        import subprocess

        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if lines and lines[0].strip().isdigit():
                return float(lines[0].strip()) / 1024.0
    except Exception as e:
        logger.debug("nvidia-smi failed: %s", e)
    return 0.0


def _get_metal_vram() -> float:
    try:
        import subprocess

        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if "Apple" in result.stdout:
            return 1.0
    except Exception:
        pass
    return 0.0


def _get_cpu_cores() -> int:
    try:
        return os.cpu_count() or 0
    except Exception:
        return 0


def _get_ollama_models() -> list[str]:
    try:
        import subprocess

        result = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []
        lines = result.stdout.strip().split("\n")[1:]
        return [line.split()[0] for line in lines if line.strip()]
    except Exception:
        return []


def get_all_ollama_models() -> list[str]:
    """Return all installed Ollama models via `ollama list`."""
    try:
        import subprocess

        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
        lines = result.stdout.strip().split("\n")[1:]
        return [line.split()[0] for line in lines if line.strip()]
    except Exception:
        return []


def detect_lm_studio_models(base_url: str = "http://localhost:1234/v1") -> list[str]:
    """Return models available from a running LM Studio instance."""
    try:
        import urllib.request
        import json

        req = urllib.request.Request(f"{base_url}/models", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
        return [m["id"] for m in data.get("data", [])]
    except Exception:
        return []


def detect_ollama_models() -> list[dict]:
    """Return detected Ollama models as config-ready dicts."""
    return [{"name": m, "provider": "ollama"} for m in get_all_ollama_models()]


def detect_lm_studio() -> list[dict]:
    """Return detected LM Studio models as config-ready dicts."""
    models = detect_lm_studio_models()
    return [{"name": m, "provider": "lm_studio"} for m in models]


def detect_local_providers() -> list[dict]:
    """Auto-detect all available local providers and return their model configs."""
    detected: list[dict] = []
    if shutil.which("ollama"):
        detected.extend(detect_ollama_models())
        logger.debug("Ollama detected, %d models found", len(detected))
    lm_studio_detected = shutil.which("lm_studio") or _check_url("http://localhost:1234/v1/models")
    logger.debug("LM Studio detected: %s", lm_studio_detected)
    if lm_studio_detected:
        detected.extend(detect_lm_studio())
    return detected


def _check_url(url: str, timeout: int = 2) -> bool:
    try:
        import urllib.request

        req = urllib.request.Request(url, method="HEAD")
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except Exception:
        return False


def scan_hardware() -> HardwareInfo:
    logger.debug("Starting hardware scan")
    total_ram, available_ram = _get_ram_info()
    logger.debug("RAM detected: %.1f GB total, %.1f GB available", total_ram, available_ram)

    vram = _get_nvidia_vram()
    has_nvidia = vram > 0
    has_metal = False

    if vram == 0:
        metal_vram = _get_metal_vram()
        if metal_vram > 0:
            vram = metal_vram
            has_metal = True
            logger.debug("Metal VRAM detected: %.1f GB", vram)
    else:
        logger.debug("NVIDIA VRAM detected: %.1f GB", vram)

    cores = _get_cpu_cores()
    logger.debug("CPU cores detected: %d", cores)

    models = _get_ollama_models()
    logger.debug("Ollama loaded models count: %d", len(models))

    return HardwareInfo(
        total_ram_gb=round(total_ram, 1),
        available_ram_gb=round(available_ram, 1),
        vram_gb=round(vram, 1),
        cpu_cores=cores,
        has_nvidia_gpu=has_nvidia,
        has_metal=has_metal,
        ollama_models_loaded=models,
    )

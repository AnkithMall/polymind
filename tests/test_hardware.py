from polymind.core.hardware import HardwareInfo, scan_hardware
from polymind.core.types import ExecutionStrategy


def test_hardware_info_defaults():
    info = HardwareInfo()
    assert info.total_ram_gb == 0.0
    assert info.cpu_cores == 0


def test_recommended_strategy_low_vram():
    info = HardwareInfo(vram_gb=4.0)
    assert info.recommended_strategy == ExecutionStrategy.sequential


def test_recommended_strategy_high_vram():
    info = HardwareInfo(vram_gb=16.0)
    assert info.recommended_strategy == ExecutionStrategy.model_aware


def test_recommended_strategy_low_ram():
    info = HardwareInfo(total_ram_gb=8.0, vram_gb=0.0)
    assert info.recommended_strategy == ExecutionStrategy.sequential


def test_recommended_strategy_high_ram():
    info = HardwareInfo(total_ram_gb=32.0, vram_gb=0.0)
    assert info.recommended_strategy == ExecutionStrategy.model_aware


def test_summary_contains_info():
    info = HardwareInfo(
        total_ram_gb=16.0,
        available_ram_gb=8.0,
        cpu_cores=8,
        vram_gb=8.0,
        has_nvidia_gpu=True,
    )
    summary = info.summary
    assert "16.0" in summary
    assert "8" in summary
    assert "NVIDIA" in summary
    assert "model_aware" in summary


def test_scam_hardware_runs():
    info = scan_hardware()
    assert isinstance(info, HardwareInfo)
    assert info.cpu_cores > 0
    assert info.total_ram_gb > 0

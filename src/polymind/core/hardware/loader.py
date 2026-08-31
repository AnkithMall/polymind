from pathlib import Path
from typing import Any

import yaml

from polymind.core.hardware.models import HardwareProfile
from polymind.core.paths import hardware_path


def load_hardware_profile(
    path: Path | None = None,
) -> HardwareProfile:
    if path is None:
        path = hardware_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Hardware profile not found: {path}. Run 'polymind hardware scan' first."
        )

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("Hardware profile must contain a YAML mapping.")

    return _parse_profile(data)


def _parse_profile(data: dict[str, Any]) -> HardwareProfile:
    """
    Convert the YAML representation back into the domain model.

    Keep this conversion in the core layer so CLI/TUI/Web clients
    can all consume the same hardware profile.
    """

    from polymind.core.hardware.models import (
        CPUInfo,
        GPUComputeInfo,
        GPUInfo,
        GPUMemoryInfo,
        GPUSelection,
        LlamaCppGPUInfo,
        LlamaCppHardwareOptions,
        MemoryInfo,
        SystemInfo,
    )

    required = [
        "version",
        "system",
        "cpu",
        "memory",
        "gpus",
        "llama_cpp",
    ]

    missing = [key for key in required if key not in data]

    if missing:
        raise ValueError(f"Hardware profile is missing required fields: {', '.join(missing)}")

    system_data = data["system"]
    cpu_data = data["cpu"]
    memory_data = data["memory"]
    llama_data = data["llama_cpp"]

    gpus: list[GPUInfo] = []

    for gpu_data in data["gpus"]:
        memory = gpu_data["memory"]
        compute = gpu_data["compute"]
        selection = gpu_data["selection"]
        llama_gpu = gpu_data["llama_cpp"]

        gpus.append(
            GPUInfo(
                id=gpu_data["id"],
                vendor=gpu_data["vendor"],
                model=gpu_data["model"],
                pci_address=gpu_data["pci"]["address"],
                pci_device_id=gpu_data["pci"].get("device_id"),
                memory=GPUMemoryInfo(
                    total_bytes=memory["total_bytes"],
                    available_bytes=memory.get("available_bytes"),
                    used_bytes=memory.get("used_bytes"),
                    shared=memory["shared"],
                ),
                driver_version=gpu_data["driver"].get("version"),
                compute=GPUComputeInfo(
                    llama_cpp_usable=compute["llama_cpp_usable"],
                    backend=compute.get("backend"),
                ),
                selection=GPUSelection(
                    enabled=selection["enabled"],
                    priority=selection["priority"],
                    reason=selection.get("reason"),
                ),
                llama_cpp=LlamaCppGPUInfo(
                    device=llama_gpu.get("device"),
                ),
            )
        )

    return HardwareProfile(
        version=data["version"],
        system=SystemInfo(
            operating_system=system_data["operating_system"],
            kernel=system_data["kernel"],
            architecture=system_data["architecture"],
        ),
        cpu=CPUInfo(
            model=cpu_data["model"],
            architecture=cpu_data["architecture"],
            physical_cores=cpu_data["physical_cores"],
            logical_cores=cpu_data["logical_cores"],
        ),
        memory=MemoryInfo(
            total_bytes=memory_data["total_bytes"],
        ),
        gpus=gpus,
        llama_cpp=LlamaCppHardwareOptions(
            available=llama_data["available"],
            backends=llama_data.get("backends", []),
            usable_gpus=llama_data.get("usable_gpus", []),
            selected_gpus=llama_data.get("selected_gpus", []),
            multi_gpu_available=llama_data.get(
                "multi_gpu_available",
                False,
            ),
        ),
    )

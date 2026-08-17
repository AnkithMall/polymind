from dataclasses import dataclass, field
from typing import Any


@dataclass
class CPUInfo:
    model: str
    architecture: str
    physical_cores: int
    logical_cores: int


@dataclass
class MemoryInfo:
    total_bytes: int


@dataclass
class GPUComputeInfo:
    llama_cpp_usable: bool
    backend: str | None


@dataclass
class GPUMemoryInfo:
    total_bytes: int
    available_bytes: int | None
    used_bytes: int | None
    shared: bool


@dataclass
class GPUSelection:
    enabled: bool
    priority: int
    reason: str | None = None


@dataclass
class LlamaCppGPUInfo:
    device: int | None


@dataclass
class GPUInfo:
    id: int

    vendor: str
    model: str

    pci_address: str
    pci_device_id: str | None

    memory: GPUMemoryInfo

    driver_version: str | None

    compute: GPUComputeInfo

    selection: GPUSelection

    llama_cpp: LlamaCppGPUInfo


@dataclass
class SystemInfo:
    operating_system: str
    kernel: str
    architecture: str


@dataclass
class LlamaCppHardwareOptions:
    available: bool
    backends: list[str] = field(default_factory=list)

    usable_gpus: list[int] = field(default_factory=list)
    selected_gpus: list[int] = field(default_factory=list)

    multi_gpu_available: bool = False


@dataclass
class HardwareProfile:
    version: int

    system: SystemInfo
    cpu: CPUInfo
    memory: MemoryInfo

    gpus: list[GPUInfo]

    llama_cpp: LlamaCppHardwareOptions

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,

            "system": {
                "operating_system": self.system.operating_system,
                "kernel": self.system.kernel,
                "architecture": self.system.architecture,
            },

            "cpu": {
                "model": self.cpu.model,
                "architecture": self.cpu.architecture,
                "physical_cores": self.cpu.physical_cores,
                "logical_cores": self.cpu.logical_cores,
            },

            "memory": {
                "total_bytes": self.memory.total_bytes,
            },

            "gpus": [
                {
                    "id": gpu.id,

                    "vendor": gpu.vendor,
                    "model": gpu.model,

                    "pci": {
                        "address": gpu.pci_address,
                        "device_id": gpu.pci_device_id,
                    },

                    "memory": {
                        "total_bytes": gpu.memory.total_bytes,
                        "available_bytes": gpu.memory.available_bytes,
                        "used_bytes": gpu.memory.used_bytes,
                        "shared": gpu.memory.shared,
                    },

                    "driver": {
                        "version": gpu.driver_version,
                    },

                    "compute": {
                        "llama_cpp_usable": gpu.compute.llama_cpp_usable,
                        "backend": gpu.compute.backend,
                    },

                    "selection": {
                        "enabled": gpu.selection.enabled,
                        "priority": gpu.selection.priority,
                        "reason": gpu.selection.reason,
                    },

                    "llama_cpp": {
                        "device": gpu.llama_cpp.device,
                    },
                }
                for gpu in self.gpus
            ],

            "llama_cpp": {
                "available": self.llama_cpp.available,
                "backends": self.llama_cpp.backends,
                "usable_gpus": self.llama_cpp.usable_gpus,
                "selected_gpus": self.llama_cpp.selected_gpus,
                "multi_gpu_available": self.llama_cpp.multi_gpu_available,
            },
        }

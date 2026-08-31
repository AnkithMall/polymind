from polymind.core.hardware.detectors.cpu import detect_cpu
from polymind.core.hardware.detectors.gpu import detect_gpus
from polymind.core.hardware.detectors.memory import detect_memory
from polymind.core.hardware.detectors.system import detect_system
from polymind.core.hardware.models import (
    HardwareProfile,
    LlamaCppHardwareOptions,
)


def scan_hardware() -> HardwareProfile:
    cpu = detect_cpu()
    memory = detect_memory()
    gpus = detect_gpus()
    system = detect_system()

    usable_gpus = [gpu.id for gpu in gpus if gpu.compute.llama_cpp_usable]

    selected_gpus = [
        gpu.id for gpu in gpus if gpu.compute.llama_cpp_usable and gpu.selection.enabled
    ]

    backends = sorted(
        {
            gpu.compute.backend
            for gpu in gpus
            if gpu.compute.llama_cpp_usable and gpu.compute.backend is not None
        }
    )

    llama_cpp = LlamaCppHardwareOptions(
        available=bool(usable_gpus),
        backends=backends,
        usable_gpus=usable_gpus,
        selected_gpus=selected_gpus,
        multi_gpu_available=len(usable_gpus) > 1,
    )

    return HardwareProfile(
        version=1,
        system=system,
        cpu=cpu,
        memory=memory,
        gpus=gpus,
        llama_cpp=llama_cpp,
    )

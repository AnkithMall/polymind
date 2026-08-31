import re
import shutil
import subprocess
from dataclasses import dataclass

from polymind.core.hardware.models import (
    GPUComputeInfo,
    GPUInfo,
    GPUMemoryInfo,
    GPUSelection,
    LlamaCppGPUInfo,
)


@dataclass
class PCIGPU:
    address: str
    vendor: str
    model: str
    device_id: str | None


@dataclass
class NvidiaGPU:
    index: int
    model: str
    total_memory: int
    used_memory: int
    available_memory: int
    driver_version: str


def _run_command(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    return result.stdout


def _detect_pci_gpus() -> list[PCIGPU]:
    if shutil.which("lspci") is None:
        return []

    output = _run_command(
        [
            "lspci",
            "-nn",
        ]
    )

    if not output:
        return []

    gpus: list[PCIGPU] = []

    for line in output.splitlines():
        if not re.search(
            r"\b(VGA compatible controller|3D controller|Display controller)\b",
            line,
            re.IGNORECASE,
        ):
            continue

        address_match = re.match(r"^([0-9a-fA-F:.]+)", line)

        if not address_match:
            continue

        address = address_match.group(1)

        pci_match = re.search(
            r"\[([0-9a-fA-F]{4}):([0-9a-fA-F]{4})\]",
            line,
        )

        vendor_id = None
        device_id = None

        if pci_match:
            vendor_id = pci_match.group(1).lower()
            device_id = pci_match.group(2).lower()

        if vendor_id == "10de":
            vendor = "NVIDIA"
        elif vendor_id == "8086":
            vendor = "Intel"
        elif vendor_id == "1002":
            vendor = "AMD"
        else:
            vendor = "Unknown"

        # Remove the PCI ID suffix from the human-readable description.
        model = line

        model = re.sub(
            r"\s*\[[0-9a-fA-F]{4}:[0-9a-fA-F]{4}\]",
            "",
            model,
        )

        model = re.sub(
            r"^[0-9a-fA-F:.]+\s*",
            "",
            model,
        )

        model = re.sub(
            r"^(VGA compatible controller|3D controller|Display controller)"
            r"\s*\[[0-9a-fA-F]{4}\]\s*:\s*",
            "",
            model,
            flags=re.IGNORECASE,
        )

        model = re.sub(
            r"\s*\(rev [^)]+\)",
            "",
            model,
            flags=re.IGNORECASE,
        )

        gpus.append(
            PCIGPU(
                address=address,
                vendor=vendor,
                model=model.strip(),
                device_id=device_id,
            )
        )

    return gpus


def _detect_nvidia_gpus() -> list[NvidiaGPU]:
    if shutil.which("nvidia-smi") is None:
        return []

    output = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    if not output:
        return []

    gpus: list[NvidiaGPU] = []

    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]

        if len(parts) != 6:
            continue

        (
            index,
            model,
            total_memory,
            used_memory,
            available_memory,
            driver,
        ) = parts

        try:
            gpu_index = int(index)

            total_bytes = int(float(total_memory) * 1024 * 1024)
            used_bytes = int(float(used_memory) * 1024 * 1024)
            available_bytes = int(float(available_memory) * 1024 * 1024)

        except ValueError:
            continue

        gpus.append(
            NvidiaGPU(
                index=gpu_index,
                model=model,
                total_memory=total_bytes,
                used_memory=used_bytes,
                available_memory=available_bytes,
                driver_version=driver,
            )
        )

    return gpus


def _create_gpu(
    gpu_id: int,
    pci_gpu: PCIGPU,
    nvidia_gpu: NvidiaGPU | None,
) -> GPUInfo:

    if pci_gpu.vendor == "NVIDIA" and nvidia_gpu is not None:
        memory = GPUMemoryInfo(
            total_bytes=nvidia_gpu.total_memory,
            available_bytes=nvidia_gpu.available_memory,
            used_bytes=nvidia_gpu.used_memory,
            shared=False,
        )

        compute = GPUComputeInfo(
            llama_cpp_usable=True,
            backend="cuda",
        )

        selection = GPUSelection(
            enabled=True,
            priority=100,
            reason=None,
        )

        llama_cpp = LlamaCppGPUInfo(
            device=nvidia_gpu.index,
        )

        return GPUInfo(
            id=gpu_id,
            vendor=pci_gpu.vendor,
            model=nvidia_gpu.model,
            pci_address=pci_gpu.address,
            pci_device_id=pci_gpu.device_id,
            memory=memory,
            driver_version=nvidia_gpu.driver_version,
            compute=compute,
            selection=selection,
            llama_cpp=llama_cpp,
        )

    # Integrated/non-NVIDIA GPU.
    #
    # We still record it because it is real hardware.
    # It is not automatically considered llama.cpp usable.
    memory = GPUMemoryInfo(
        total_bytes=0,
        available_bytes=None,
        used_bytes=None,
        shared=True,
    )

    compute = GPUComputeInfo(
        llama_cpp_usable=False,
        backend=None,
    )

    selection = GPUSelection(
        enabled=False,
        priority=0,
        reason="No supported llama.cpp backend detected",
    )

    llama_cpp = LlamaCppGPUInfo(
        device=None,
    )

    return GPUInfo(
        id=gpu_id,
        vendor=pci_gpu.vendor,
        model=pci_gpu.model,
        pci_address=pci_gpu.address,
        pci_device_id=pci_gpu.device_id,
        memory=memory,
        driver_version=None,
        compute=compute,
        selection=selection,
        llama_cpp=llama_cpp,
    )


def detect_gpus() -> list[GPUInfo]:
    pci_gpus = _detect_pci_gpus()
    nvidia_gpus = _detect_nvidia_gpus()

    nvidia_by_index = {gpu.index: gpu for gpu in nvidia_gpus}

    result: list[GPUInfo] = []

    gpu_id = 0

    for pci_gpu in pci_gpus:
        nvidia_gpu = None

        if pci_gpu.vendor == "NVIDIA":
            # nvidia-smi's GPU index is independent from our inventory ID.
            #
            # For now, match NVIDIA GPUs in enumeration order.
            nvidia_position = sum(
                1 for gpu in pci_gpus[: pci_gpus.index(pci_gpu)] if gpu.vendor == "NVIDIA"
            )

            nvidia_gpu = nvidia_by_index.get(nvidia_position)

        result.append(
            _create_gpu(
                gpu_id=gpu_id,
                pci_gpu=pci_gpu,
                nvidia_gpu=nvidia_gpu,
            )
        )

        gpu_id += 1

    return result

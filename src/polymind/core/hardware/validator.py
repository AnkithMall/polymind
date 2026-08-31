from dataclasses import dataclass
from pathlib import Path

from polymind.core.hardware.loader import load_hardware_profile
from polymind.core.hardware.models import HardwareProfile


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]


def validate_hardware_profile(
    profile: HardwareProfile,
) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    # ---------------------------------------------------------
    # Basic profile validation
    # ---------------------------------------------------------

    if profile.version != 1:
        errors.append(f"Unsupported hardware profile version: {profile.version}")

    # ---------------------------------------------------------
    # GPU IDs
    # ---------------------------------------------------------

    gpu_ids = [gpu.id for gpu in profile.gpus]

    if len(gpu_ids) != len(set(gpu_ids)):
        errors.append("GPU IDs must be unique.")

    gpu_map = {gpu.id: gpu for gpu in profile.gpus}

    # ---------------------------------------------------------
    # llama.cpp availability
    # ---------------------------------------------------------

    if profile.llama_cpp.available:
        if not profile.llama_cpp.backends:
            errors.append("llama.cpp is marked available but no backends are configured.")

    # ---------------------------------------------------------
    # Usable GPUs
    # ---------------------------------------------------------

    usable_gpu_ids = set(profile.llama_cpp.usable_gpus)

    for gpu_id in profile.llama_cpp.usable_gpus:
        gpu = gpu_map.get(gpu_id)

        if gpu is None:
            errors.append(f"llama.cpp usable GPU {gpu_id} does not exist.")
            continue

        if not gpu.compute.llama_cpp_usable:
            errors.append(
                f"GPU {gpu_id} is listed as llama.cpp usable "
                "but its GPU capability says it is not usable."
            )

        if gpu.compute.backend is None:
            errors.append(f"GPU {gpu_id} is llama.cpp usable but has no backend.")

    # ---------------------------------------------------------
    # Selected GPUs
    # ---------------------------------------------------------

    selected_gpu_ids = profile.llama_cpp.selected_gpus

    if len(selected_gpu_ids) != len(set(selected_gpu_ids)):
        errors.append("Selected GPU IDs must be unique.")

    for gpu_id in selected_gpu_ids:
        gpu = gpu_map.get(gpu_id)

        if gpu is None:
            errors.append(f"Selected GPU {gpu_id} does not exist.")
            continue

        if gpu_id not in usable_gpu_ids:
            errors.append(f"Selected GPU {gpu_id} is not listed as llama.cpp usable.")

        if not gpu.compute.llama_cpp_usable:
            errors.append(f"Selected GPU {gpu_id} cannot be used by llama.cpp.")

        if gpu.compute.backend is None:
            errors.append(f"Selected GPU {gpu_id} has no llama.cpp backend.")

        if gpu.llama_cpp.device is None:
            errors.append(f"Selected GPU {gpu_id} has no llama.cpp device ID.")

    # ---------------------------------------------------------
    # Multi-GPU consistency
    # ---------------------------------------------------------

    expected_multi_gpu = len(usable_gpu_ids) > 1

    if profile.llama_cpp.multi_gpu_available != expected_multi_gpu:
        warnings.append("multi_gpu_available does not match the number of usable GPUs.")

    # ---------------------------------------------------------
    # Selection consistency
    # ---------------------------------------------------------

    for gpu in profile.gpus:
        selected = gpu.id in selected_gpu_ids

        if gpu.selection.enabled != selected:
            warnings.append(
                f"GPU {gpu.id} selection.enabled does not match llama_cpp.selected_gpus."
            )

    # ---------------------------------------------------------
    # No GPU selected
    # ---------------------------------------------------------

    if profile.llama_cpp.available and not selected_gpu_ids:
        warnings.append("llama.cpp GPUs are available but none are selected.")

    return ValidationResult(
        valid=not errors,
        errors=errors,
        warnings=warnings,
    )


def validate_hardware_file(
    path: Path,
) -> ValidationResult:
    try:
        profile = load_hardware_profile(path)
    except FileNotFoundError as exc:
        return ValidationResult(
            valid=False,
            errors=[str(exc)],
            warnings=[],
        )
    except ValueError as exc:
        return ValidationResult(
            valid=False,
            errors=[f"Invalid hardware profile: {exc}"],
            warnings=[],
        )

    return validate_hardware_profile(profile)

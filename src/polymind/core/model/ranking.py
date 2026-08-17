import math

from polymind.core.hardware.models import HardwareProfile
from polymind.core.model.types import (
    ModelCandidate,
    RankedModel,
)


GPU_SAFETY_MARGIN = 0.80
RAM_SAFETY_MARGIN = 0.75


def rank_models(
    models: list[ModelCandidate],
    hardware: HardwareProfile,
) -> list[RankedModel]:

    available_vram = (
        _selected_gpu_memory(hardware)
        or 0
    )

    available_ram = hardware.memory.total_bytes

    ranked: list[RankedModel] = []

    for model in models:
        for file in model.files:

            if file.size_bytes is None:
                continue

            size = file.size_bytes

            vram_status = classify_vram(
                size,
                available_vram,
            )

            ram_status = classify_ram(
                size,
                available_ram,
            )

            fits_vram = vram_status == "likely-fit"
            fits_ram = ram_status == "fits"

            score = 0.0

            # -------------------------------------------------
            # Hardware compatibility
            # -------------------------------------------------

            if vram_status == "likely-fit":
                score += 100

            elif vram_status == "partial-offload":
                score += 55

            elif vram_status == "too-large":
                score -= 100

            elif vram_status == "unavailable":
                score -= 100

            if ram_status == "fits":
                score += 30
            else:
                score -= 100

            # -------------------------------------------------
            # Model quality
            # -------------------------------------------------

            if file.quantization:
                score += quantization_score(
                    file.quantization
                )
            else:
                score -= 10

            # Prefer larger models when hardware can
            # reasonably support them.
            if model.parameter_count_b is not None:
                score += parameter_score(
                    model.parameter_count_b
                )

            # -------------------------------------------------
            # Popularity
            # -------------------------------------------------

            score += min(
                model.downloads / 100_000,
                10,
            )

            score += min(
                model.likes / 10_000,
                5,
            )

            # -------------------------------------------------
            # Recommendation
            # -------------------------------------------------

            recommended = (
                ram_status == "fits"
                and vram_status in {
                    "likely-fit",
                    "partial-offload",
                }
                and file.quantization is not None
            )

            ranked.append(
                RankedModel(
                    repo_id=model.repo_id,
                    model_name=model.model_name,
                    filename=file.filename,
                    size_bytes=file.size_bytes,
                    quantization=file.quantization,
                    downloads=model.downloads,
                    likes=model.likes,
                    parameter_count_b=model.parameter_count_b,
                    score=score,
                    vram_status=vram_status,
                    ram_status=ram_status,
                    fits_vram=fits_vram,
                    fits_ram=fits_ram,
                    recommended=recommended,
                    shard_count=file.shard_count or 1,
                )
            )

    # Highest quality/hardware score first.
    ranked.sort(
        key=lambda item: (
            item.score,
            item.parameter_count_b or 0,
        ),
        reverse=True,
    )

    return ranked

def classify_vram(
    model_size: int,
    available_vram: int,
) -> str:

    if available_vram <= 0:
        return "unavailable"

    if model_size <= available_vram * GPU_SAFETY_MARGIN:
        return "likely-fit"

    # A model that is larger than VRAM can still be
    # partially offloaded by llama.cpp.
    if model_size <= available_vram * 3:
        return "partial-offload"

    return "too-large"


def classify_ram(
    model_size: int,
    available_ram: int,
) -> str:

    if model_size <= available_ram * RAM_SAFETY_MARGIN:
        return "fits"

    return "too-large"


def _selected_gpu_memory(
    hardware: HardwareProfile,
) -> int | None:

    selected = set(
        hardware.llama_cpp.selected_gpus
    )

    memories: list[int] = []

    for gpu in hardware.gpus:

        if gpu.id not in selected:
            continue

        if not gpu.compute.llama_cpp_usable:
            continue

        if gpu.memory.available_bytes:
            memories.append(
                gpu.memory.available_bytes
            )

        elif gpu.memory.total_bytes:
            memories.append(
                gpu.memory.total_bytes
            )

    if not memories:
        return None

    # Multi-GPU tensor distribution remains a runtime
    # responsibility.
    return max(memories)


def parameter_score(
    parameter_count_b: float,
) -> float:

    # Logarithmic scaling prevents 70B from completely
    # dominating the ranking.
    return min(
        math.log2(parameter_count_b + 1) * 8,
        45,
    )


def quantization_score(
    quantization: str,
) -> float:

    scores = {
        "IQ1_S": 1,
        "IQ1_M": 2,
        "IQ2_XXS": 3,
        "IQ2_XS": 4,
        "IQ2_S": 5,
        "IQ3_XXS": 6,
        "IQ3_XS": 7,
        "IQ3_S": 8,
        "Q2_K": 9,
        "Q3_K_S": 10,
        "Q3_K_M": 12,
        "Q3_K_L": 13,
        "Q4_0": 14,
        "Q4_1": 14,
        "Q4_K_S": 16,
        "Q4_K_M": 18,
        "Q5_0": 15,
        "Q5_1": 15,
        "Q5_K_S": 18,
        "Q5_K_M": 20,
        "Q6_K": 19,
        "Q8_0": 17,
        "F16": 15,
        "BF16": 15,
        "F32": 10,
    }

    return scores.get(
        quantization,
        0,
    )

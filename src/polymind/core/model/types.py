from dataclasses import dataclass
from typing import Literal

Recommendation = Literal[
    "excellent",
    "recommended",
    "usable",
    "marginal",
    "not-recommended",
]


@dataclass
class ModelFile:
    filename: str
    size_bytes: int | None
    quantization: str | None

    # GGUF sharding information.
    shard_index: int | None = None
    shard_count: int | None = None

    # Files belonging to the same logical GGUF model.
    group_key: str | None = None


@dataclass
class ModelCandidate:
    repo_id: str
    model_name: str
    author: str

    downloads: int
    likes: int

    # Approximate parameter count inferred from the
    # repository/model name when available.
    parameter_count_b: float | None

    files: list[ModelFile]


@dataclass
class RankedModel:
    repo_id: str
    model_name: str

    # Logical model filename/group.
    filename: str

    size_bytes: int | None
    quantization: str | None

    downloads: int
    likes: int

    parameter_count_b: float | None

    score: float

    # Hardware compatibility.
    vram_status: str
    ram_status: str

    # Kept for compatibility with existing code.
    fits_vram: bool
    fits_ram: bool

    recommended: bool

    # GGUF shard information.
    shard_count: int = 1

    # Whether this model is already downloaded.
    downloaded: bool = False

from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelSearchOptions:
    query: str

    limit: int = 20

    author: str | None = None

    quantization: str | None = None

    max_size_bytes: int | None = None

    min_size_bytes: int | None = None

    sort: Literal[
        "created_at",
        "downloads",
        "last_modified",
        "likes",
        "trending_score",
    ] = "downloads"

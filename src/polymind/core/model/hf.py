from __future__ import annotations

import re
from typing import Optional

from huggingface_hub import HfApi, hf_hub_download

from polymind.core.model.search import ModelSearchOptions
from polymind.core.model.types import (
    ModelCandidate,
    ModelFile,
)


class HuggingFaceClient:
    def __init__(self) -> None:
        self.api = HfApi()

    def search_gguf(
        self,
        options: ModelSearchOptions,
    ) -> list[ModelCandidate]:

        models = self.api.list_models(
            search=options.query,
            filter="gguf",
            sort=options.sort,
            limit=options.limit,
            author=options.author,
        )

        results: list[ModelCandidate] = []

        for model in models:
            try:
                info = self.api.model_info(
                    model.id,
                    files_metadata=True,
                )
            except Exception:
                continue

            raw_files: list[ModelFile] = []

            for sibling in info.siblings or []:
                filename = sibling.rfilename

                if not filename.lower().endswith(".gguf"):
                    continue

                quantization = detect_quantization(
                    filename
                )

                shard_index, shard_count = detect_shard(
                    filename
                )

                group_key = model_group_key(
                    filename
                )

                size_bytes = getattr(
                    sibling,
                    "size",
                    None,
                )

                raw_files.append(
                    ModelFile(
                        filename=filename,
                        size_bytes=size_bytes,
                        quantization=quantization,
                        shard_index=shard_index,
                        shard_count=shard_count,
                        group_key=group_key,
                    )
                )

            # Merge shard files belonging to the same
            # logical model.
            files = merge_shards(raw_files)

            # Apply filters AFTER shard merging because
            # max-size/min-size should refer to the
            # complete logical model.
            filtered_files: list[ModelFile] = []

            for file in files:
                if (
                    options.quantization
                    and (
                        file.quantization is None
                        or file.quantization.upper()
                        != options.quantization.upper()
                    )
                ):
                    continue

                if (
                    options.max_size_bytes is not None
                    and file.size_bytes is not None
                    and file.size_bytes > options.max_size_bytes
                ):
                    continue

                if (
                    options.min_size_bytes is not None
                    and file.size_bytes is not None
                    and file.size_bytes < options.min_size_bytes
                ):
                    continue

                filtered_files.append(file)

            if not filtered_files:
                continue

            author = model.id.split("/", 1)[0]

            results.append(
                ModelCandidate(
                    repo_id=model.id,
                    model_name=model.id.split("/")[-1],
                    author=author,
                    downloads=getattr(
                        model,
                        "downloads",
                        0,
                    )
                    or 0,
                    likes=getattr(
                        model,
                        "likes",
                        0,
                    )
                    or 0,
                    parameter_count_b=detect_parameter_count(
                        model.id
                    ),
                    files=filtered_files,
                )
            )

        return results

    def download(
        self,
        repo_id: str,
        filename: str,
        local_dir: str,
    ) -> str:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
        )


def detect_quantization(
    filename: str,
) -> Optional[str]:
    """
    Extract common GGUF quantization names.

    The match is performed against filename components
    rather than arbitrary substrings where possible.
    """

    upper = filename.upper()

    quantizations = [
        "IQ1_S",
        "IQ1_M",
        "IQ2_XXS",
        "IQ2_XS",
        "IQ2_S",
        "IQ3_XXS",
        "IQ3_XS",
        "IQ3_S",
        "IQ4_NL",
        "IQ4_XS",
        "Q2_K",
        "Q3_K_S",
        "Q3_K_M",
        "Q3_K_L",
        "Q4_K_S",
        "Q4_K_M",
        "Q4_0",
        "Q4_1",
        "Q5_K_S",
        "Q5_K_M",
        "Q5_0",
        "Q5_1",
        "Q6_K",
        "Q8_0",
        "F16",
        "BF16",
        "F32",
    ]

    quantizations.sort(
        key=len,
        reverse=True,
    )

    for quant in quantizations:
        # Accept common separators around the
        # quantization name.
        pattern = rf"(?<![A-Z0-9]){re.escape(quant)}(?![A-Z0-9])"

        if re.search(pattern, upper):
            return quant

    return None


def detect_shard(
    filename: str,
) -> tuple[int | None, int | None]:
    """
    Detect GGUF split/sharded filenames.

    Examples:

        model-Q4_K_M-00001-of-00003.gguf

        model-Q4_K_M-00002-of-00003.gguf
    """

    match = re.search(
        r"-(\d{5})-of-(\d{5})\.gguf$",
        filename,
        re.IGNORECASE,
    )

    if not match:
        return None, None

    return (
        int(match.group(1)),
        int(match.group(2)),
    )


def model_group_key(
    filename: str,
) -> str:
    """
    Return the logical GGUF model name.

    Removes shard suffixes so all shards become one
    logical model.
    """

    return re.sub(
        r"-\d{5}-of-\d{5}(?=\.gguf$)",
        "",
        filename,
        flags=re.IGNORECASE,
    )


def merge_shards(
    files: list[ModelFile],
) -> list[ModelFile]:
    """
    Merge split GGUF files into logical model entries.

    A logical model gets the combined size of all shards.
    """

    groups: dict[str, list[ModelFile]] = {}

    for file in files:
        key = file.group_key or file.filename

        groups.setdefault(
            key,
            [],
        ).append(file)

    result: list[ModelFile] = []

    for key, group in groups.items():
        first = group[0]

        total_size: int | None

        sizes = [
            file.size_bytes
            for file in group
            if file.size_bytes is not None
        ]

        if sizes:
            total_size = sum(sizes)
        else:
            total_size = None

        shard_count = max(
            (
                file.shard_count
                for file in group
                if file.shard_count is not None
            ),
            default=1,
        )

        result.append(
            ModelFile(
                filename=key,
                size_bytes=total_size,
                quantization=first.quantization,
                shard_index=None,
                shard_count=shard_count,
                group_key=key,
            )
        )

    return result


def detect_parameter_count(
    model_id: str,
) -> float | None:
    """
    Best-effort parameter count extraction from the
    repository name.

    Examples:

        Llama-3.2-1B -> 1
        Llama-3.2-3B -> 3
        Llama-3.1-70B -> 70

    This is intentionally only a discovery heuristic.
    Runtime will eventually use actual GGUF metadata.
    """

    match = re.search(
        r"(?<![A-Za-z0-9])"
        r"(\d+(?:\.\d+)?)"
        r"([BM])"
        r"(?![A-Za-z0-9])",
        model_id.upper(),
    )

    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2)

    if unit == "M":
        return value / 1000

    return value

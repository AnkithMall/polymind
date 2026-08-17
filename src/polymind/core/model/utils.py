from pathlib import Path
import re

_SIZE_UNITS = {
    "B": 1,
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
    "KIB": 1024,
    "MIB": 1024**2,
    "GIB": 1024**3,
    "TIB": 1024**4,
}

def file_size_bytes(path: Path) -> int:
    return path.stat().st_size


def format_size(size_bytes: int) -> str:
    size = float(size_bytes)

    units = [
        "B",
        "KiB",
        "MiB",
        "GiB",
        "TiB",
    ]

    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size_bytes} B"


def parse_size(value: str) -> int:
    value = value.strip().upper()

    match = re.fullmatch(
        r"([0-9]+(?:\.[0-9]+)?)\s*([A-Z]+)",
        value,
    )

    if not match:
        raise ValueError(
            f"Invalid size: {value}. "
            "Examples: 2GiB, 3GB, 500MiB"
        )

    number = float(match.group(1))
    unit = match.group(2)

    multiplier = _SIZE_UNITS.get(unit)

    if multiplier is None:
        raise ValueError(
            f"Unknown size unit: {unit}"
        )

    return int(number * multiplier)

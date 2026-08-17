import psutil

from polymind.core.hardware.models import MemoryInfo


def detect_memory() -> MemoryInfo:
    memory = psutil.virtual_memory()

    return MemoryInfo(
        total_bytes=memory.total,
    )

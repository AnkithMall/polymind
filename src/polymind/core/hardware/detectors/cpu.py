import platform

import psutil

from polymind.core.hardware.models import CPUInfo


def _read_cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as file:
            for line in file:
                if line.lower().startswith("model name"):
                    _, value = line.split(":", 1)
                    return value.strip()
    except OSError:
        pass

    return platform.processor() or "Unknown"


def detect_cpu() -> CPUInfo:
    return CPUInfo(
        model=_read_cpu_model(),
        architecture=platform.machine(),
        physical_cores=psutil.cpu_count(logical=False) or 1,
        logical_cores=psutil.cpu_count(logical=True) or 1,
    )

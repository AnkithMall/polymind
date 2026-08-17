import platform

from polymind.core.hardware.models import SystemInfo


def detect_system() -> SystemInfo:
    return SystemInfo(
        operating_system=platform.system(),
        kernel=platform.release(),
        architecture=platform.machine(),
    )

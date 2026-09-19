from pathlib import Path

import yaml

from polymind.core.hardware.models import HardwareProfile
from polymind.core.paths import hardware_path


def write_hardware_profile(
    profile: HardwareProfile,
    path: Path | None = None,
) -> Path:
    if path is None:
        path = hardware_path()

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            profile.to_dict(),
            file,
            sort_keys=False,
            default_flow_style=False,
        )

    return path

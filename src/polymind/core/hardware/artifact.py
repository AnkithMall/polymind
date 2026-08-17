from pathlib import Path

import yaml

from polymind.core.hardware.models import HardwareProfile


DEFAULT_ARTIFACT_PATH = Path(".polymind/hardware.yaml")


def write_hardware_profile(
    profile: HardwareProfile,
    path: Path = DEFAULT_ARTIFACT_PATH,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            profile.to_dict(),
            file,
            sort_keys=False,
            default_flow_style=False,
        )

    return path

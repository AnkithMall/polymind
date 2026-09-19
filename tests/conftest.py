"""Shared test fixtures for polymind tests."""

from pathlib import Path

import pytest


@pytest.fixture
def tmp_polymind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temporary polymind environment with env vars set."""
    artifact_dir = tmp_path / ".polymind"
    artifact_dir.mkdir()

    model_dir = artifact_dir / "models"
    model_dir.mkdir()

    # Set env vars so all code uses the temp directory
    monkeypatch.setenv("POLYMIND_ARTIFACT_DIR", str(artifact_dir))
    monkeypatch.setenv("POLYMIND_MODEL_DIR", str(model_dir))

    # Create minimal hardware profile
    hardware_yaml = artifact_dir / "hardware.yaml"
    hardware_yaml.write_text(
        """version: 1
system:
  operating_system: Linux
  kernel: 6.0.0
  architecture: x86_64
cpu:
  model: Test CPU
  architecture: x86_64
  physical_cores: 4
  logical_cores: 8
memory:
  total_bytes: 16000000000
gpus:
- id: 0
  vendor: NVIDIA
  model: Test GPU
  pci:
    address: '01:00.0'
    device_id: '1234'
  memory:
    total_bytes: 4000000000
    available_bytes: 3500000000
    used_bytes: 500000000
    shared: false
  driver:
    version: '500.0'
  compute:
    llama_cpp_usable: true
    backend: cuda
  selection:
    enabled: true
    priority: 100
    reason: null
  llama_cpp:
    device: 0
llama_cpp:
  available: true
  backends:
  - cuda
  usable_gpus:
  - 0
  selected_gpus:
  - 0
  multi_gpu_available: false
""",
        encoding="utf-8",
    )

    # Create empty registry
    registry_yaml = artifact_dir / "registry.yaml"
    registry_yaml.write_text(
        "version: 1\nmodels: []\n",
        encoding="utf-8",
    )

    # Create empty runtime config
    runtime_yaml = artifact_dir / "runtime.yaml"
    runtime_yaml.write_text(
        "version: 1\nmodels: {}\n",
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture
def mock_gguf(tmp_path: Path) -> Path:
    """Create a minimal fake GGUF file for testing."""
    gguf_file = tmp_path / "test-model-Q4_K_M.gguf"

    # GGUF magic bytes
    magic = b"GGUF"
    version = (3).to_bytes(4, "little")

    with open(gguf_file, "wb") as f:
        f.write(magic)
        f.write(version)
        # Pad to make it look like a real file
        f.write(b"\x00" * 1024)

    return gguf_file


@pytest.fixture
def env_override(tmp_polymind: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Set environment variables to use temp directory."""
    monkeypatch.setenv("POLYMIND_ARTIFACT_DIR", str(tmp_polymind / ".polymind"))
    monkeypatch.setenv("POLYMIND_MODEL_DIR", str(tmp_polymind / ".polymind" / "models"))

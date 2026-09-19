"""Tests for polymind model commands."""

import pytest
from typer.testing import CliRunner

from polymind.client.cli.app import app

runner = CliRunner()


class TestModelList:
    """Tests for polymind model list."""

    def test_model_list_empty(self, tmp_polymind, env_override):
        """model list should show no models when empty."""
        result = runner.invoke(app, ["model", "list"])
        assert result.exit_code == 0
        assert "No models installed" in result.output

    def test_model_list_with_model(self, tmp_polymind, env_override):
        """model list should show installed models."""
        # Add a model to registry
        registry_yaml = tmp_polymind / ".polymind" / "registry.yaml"
        registry_yaml.write_text(
            """version: 1
models:
- id: 1
  repo_id: test/repo
  filename: test-model.gguf
  local_path: /tmp/test-model.gguf
  size_bytes: 1000000
  quantization: Q4_K_M
""",
            encoding="utf-8",
        )

        result = runner.invoke(app, ["model", "list"])
        assert result.exit_code == 0
        assert "test-model.gguf" in result.output


class TestModelScan:
    """Tests for polymind model scan."""

    def test_model_scan_empty(self, tmp_polymind, env_override):
        """model scan should work on empty directory."""
        result = runner.invoke(app, ["model", "scan"])
        assert result.exit_code == 0
        assert "Registry is up to date" in result.output or "nothing to fix" in result.output

    def test_model_scan_finds_gguf(self, tmp_polymind, env_override, mock_gguf):
        """model scan should find GGUF files."""
        # Copy mock GGUF to model directory
        model_dir = tmp_polymind / ".polymind" / "models"
        import shutil

        shutil.copy(mock_gguf, model_dir / "test-model.gguf")

        result = runner.invoke(app, ["model", "scan"])
        assert result.exit_code == 0
        assert "Registered" in result.output

    def test_model_scan_all(self, tmp_polymind, env_override):
        """model scan --all should scan multiple directories."""
        result = runner.invoke(app, ["model", "scan", "--all"])
        assert result.exit_code == 0

    def test_model_scan_excludes_non_text(self, tmp_polymind, env_override):
        """model scan should skip mmproj files."""
        model_dir = tmp_polymind / ".polymind" / "models"
        mmproj = model_dir / "mmproj-model.gguf"
        mmproj.write_bytes(b"GGUF" + b"\x00" * 100)

        result = runner.invoke(app, ["model", "scan"])
        assert result.exit_code == 0
        # mmproj should NOT be registered
        assert "mmproj" not in result.output or "nothing to fix" in result.output


class TestModelDelete:
    """Tests for polymind model delete."""

    def test_model_delete_not_found(self, tmp_polymind, env_override):
        """model delete should fail for non-existent model."""
        result = runner.invoke(app, ["model", "delete", "999"])
        assert result.exit_code != 0 or "not found" in result.output.lower()

    def test_model_delete_by_id(self, tmp_polymind, env_override):
        """model delete should remove model by ID."""
        # Add a model
        registry_yaml = tmp_polymind / ".polymind" / "registry.yaml"
        registry_yaml.write_text(
            """version: 1
models:
- id: 1
  repo_id: test/repo
  filename: test-model.gguf
  local_path: /tmp/test-model.gguf
  size_bytes: 1000000
  quantization: Q4_K_M
""",
            encoding="utf-8",
        )

        result = runner.invoke(app, ["model", "delete", "1", "--force"])
        assert result.exit_code == 0
        assert "deleted" in result.output.lower()

    def test_model_delete_by_filename(self, tmp_polymind, env_override):
        """model delete should remove model by filename."""
        registry_yaml = tmp_polymind / ".polymind" / "registry.yaml"
        registry_yaml.write_text(
            """version: 1
models:
- id: 1
  repo_id: test/repo
  filename: test-model.gguf
  local_path: /tmp/test-model.gguf
  size_bytes: 1000000
  quantization: Q4_K_M
""",
            encoding="utf-8",
        )

        result = runner.invoke(app, ["model", "delete", "test-model.gguf", "--force"])
        assert result.exit_code == 0


class TestModelMigrate:
    """Tests for polymind model migrate."""

    def test_migrate_no_models(self, tmp_polymind, env_override):
        """migrate should handle no legacy models gracefully."""
        result = runner.invoke(app, ["model", "migrate", "--force"])
        # Should succeed or fail gracefully
        assert result.exit_code in (0, 1)

    def test_migrate_with_models(self, tmp_polymind, env_override):
        """migrate should move models from legacy location."""
        # Create legacy directory with a GGUF file
        legacy_dir = tmp_polymind / "polymind" / "models"
        legacy_dir.mkdir(parents=True)

        import shutil

        mock_gguf = tmp_polymind / "test-model.gguf"
        mock_gguf.write_bytes(b"GGUF" + b"\x00" * 1024)
        shutil.copy(mock_gguf, legacy_dir / "test-model.gguf")

        result = runner.invoke(
            app,
            ["model", "migrate", "--source", str(legacy_dir), "--force"],
        )
        assert result.exit_code == 0
        assert "migrated" in result.output.lower() or "complete" in result.output.lower()


class TestModelSearch:
    """Tests for polymind model search (requires network)."""

    @pytest.mark.skipif(
        not pytest.importorskip("huggingface_hub"),
        reason="huggingface_hub not installed",
    )
    def test_model_search_help(self):
        """model search --help should work."""
        result = runner.invoke(app, ["model", "search", "--help"])
        assert result.exit_code == 0
        assert "search" in result.output.lower() or "query" in result.output.lower()


class TestModelDownload:
    """Tests for polymind model download (requires network)."""

    def test_download_help(self):
        """model download --help should work."""
        result = runner.invoke(app, ["model", "download", "--help"])
        assert result.exit_code == 0

    def test_download_already_registered(self, tmp_polymind, env_override):
        """model download should skip if already registered."""
        # Create a fake model file
        model_dir = tmp_polymind / ".polymind" / "models"
        model_file = model_dir / "Llama-3.2-3B-Instruct.Q4_K_M.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 1024)

        # Register it
        registry_yaml = tmp_polymind / ".polymind" / "registry.yaml"
        registry_yaml.write_text(
            f"""version: 1
models:
- id: 1
  repo_id: MaziyarPanahi/Llama-3.2-3B-Instruct-GGUF
  filename: Llama-3.2-3B-Instruct.Q4_K_M.gguf
  local_path: {model_file}
  size_bytes: 1024
  quantization: Q4_K_M
""",
            encoding="utf-8",
        )

        result = runner.invoke(
            app,
            [
                "model",
                "download",
                "MaziyarPanahi/Llama-3.2-3B-Instruct-GGUF",
                "Llama-3.2-3B-Instruct.Q4_K_M.gguf",
            ],
        )
        # Should detect already downloaded
        assert result.exit_code == 0
        assert "already downloaded" in result.output.lower()

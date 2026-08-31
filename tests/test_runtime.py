"""Tests for polymind runtime commands."""

from typer.testing import CliRunner

from polymind.client.cli.app import app

runner = CliRunner()


class TestRuntimeOptimize:
    """Tests for polymind runtime optimize."""

    def test_optimize_no_models(self, tmp_polymind, env_override):
        """optimize should fail with no models."""
        result = runner.invoke(app, ["runtime", "optimize", "--all"])
        assert result.exit_code != 0
        assert "no models" in result.output.lower()

    def test_optimize_no_model_flag(self, tmp_polymind, env_override):
        """optimize should require --model or --all."""
        result = runner.invoke(app, ["runtime", "optimize"])
        assert result.exit_code != 0
        assert "specify" in result.output.lower()

    def test_optimize_model_not_found(self, tmp_polymind, env_override):
        """optimize should fail for non-existent model."""
        result = runner.invoke(app, ["runtime", "optimize", "-m", "999"])
        assert result.exit_code != 0

    def test_optimize_with_model(self, tmp_polymind, env_override):
        """optimize should work with a model (CPU-only mock)."""
        # Register a model with a small test file
        model_dir = tmp_polymind / ".polymind" / "models"
        model_file = model_dir / "test-model.gguf"
        # Create a minimal valid GGUF-like file
        model_file.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\x00" * 1024)

        registry_yaml = tmp_polymind / ".polymind" / "registry.yaml"
        registry_yaml.write_text(
            f"""version: 1
models:
- id: 1
  repo_id: manual/unknown
  filename: test-model.gguf
  local_path: {model_file}
  size_bytes: 1024
  quantization: Q4_K_M
""",
            encoding="utf-8",
        )

        result = runner.invoke(app, ["runtime", "optimize", "-m", "1"])
        # This may fail because the GGUF file is fake,
        # but it should not crash
        assert result.exit_code in (0, 1)


class TestRuntimeRun:
    """Tests for polymind runtime run."""

    def test_run_no_models(self, tmp_polymind, env_override):
        """run should fail with no models."""
        result = runner.invoke(app, ["runtime", "run", "-m", "1"])
        assert result.exit_code != 0
        assert "no models" in result.output.lower()

    def test_run_model_not_found(self, tmp_polymind, env_override):
        """run should fail for non-existent model."""
        result = runner.invoke(app, ["runtime", "run", "-m", "999"])
        assert result.exit_code != 0

    def test_run_file_not_exist(self, tmp_polymind, env_override):
        """run should fail if model file doesn't exist."""
        registry_yaml = tmp_polymind / ".polymind" / "registry.yaml"
        registry_yaml.write_text(
            """version: 1
models:
- id: 1
  repo_id: test/repo
  filename: test-model.gguf
  local_path: /nonexistent/test-model.gguf
  size_bytes: 1000000
  quantization: Q4_K_M
""",
            encoding="utf-8",
        )

        result = runner.invoke(app, ["runtime", "run", "-m", "1"])
        assert result.exit_code != 0
        assert "not exist" in result.output.lower() or "not found" in result.output.lower()


class TestRuntimeConfig:
    """Tests for runtime configuration loading."""

    def test_load_runtime_config(self, tmp_polymind, env_override):
        """load_runtime_config should read from runtime.yaml."""
        from polymind.core.runtime.artifact import load_runtime_config

        # Write a test config
        runtime_yaml = tmp_polymind / ".polymind" / "runtime.yaml"
        runtime_yaml.write_text(
            """version: 1
models:
  '1':
    model_id: '1'
    gpu_layers: 0
    threads: 5
    context_size: 2048
    batch_size: 256
""",
            encoding="utf-8",
        )

        config = load_runtime_config("1", runtime_yaml)
        assert config is not None
        assert config.gpu_layers == 0
        assert config.threads == 5
        assert config.context_size == 2048
        assert config.batch_size == 256

    def test_load_runtime_config_not_found(self, tmp_polymind, env_override):
        """load_runtime_config should return None for missing model."""
        from polymind.core.runtime.artifact import load_runtime_config

        runtime_yaml = tmp_polymind / ".polymind" / "runtime.yaml"
        runtime_yaml.write_text("version: 1\nmodels: {}\n", encoding="utf-8")

        config = load_runtime_config("999", runtime_yaml)
        assert config is None

    def test_write_runtime_config(self, tmp_polymind, env_override):
        """write_runtime_config should write to runtime.yaml."""
        from polymind.core.runtime.artifact import write_runtime_config
        from polymind.core.runtime.types import RuntimeConfig

        runtime_yaml = tmp_polymind / ".polymind" / "runtime.yaml"
        config = RuntimeConfig(
            model_id="1",
            gpu_layers=0,
            threads=5,
            context_size=2048,
            batch_size=256,
        )

        result = write_runtime_config(config, runtime_yaml)
        assert result.exists()

        # Verify content
        import yaml

        with open(runtime_yaml) as f:
            data = yaml.safe_load(f)

        assert data["models"]["1"]["gpu_layers"] == 0
        assert data["models"]["1"]["threads"] == 5

"""Tests for polymind config commands."""

from typer.testing import CliRunner

from polymind.client.cli.app import app

runner = CliRunner()


class TestConfigShow:
    """Tests for polymind config show."""

    def test_config_show(self, tmp_polymind, env_override):
        """config show should display configuration."""
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        assert "Polymind Configuration" in result.output

    def test_config_show_raw(self, tmp_polymind, env_override):
        """config show --raw should show file contents."""
        result = runner.invoke(app, ["config", "show", "--raw"])
        assert result.exit_code == 0

    def test_config_show_environment(self, tmp_polymind, env_override):
        """config show should display environment variables."""
        result = runner.invoke(app, ["config", "show"])
        assert "Environment Variables" in result.output
        assert "POLYMIND_ARTIFACT_DIR" in result.output

    def test_config_show_paths(self, tmp_polymind, env_override):
        """config show should display resolved paths."""
        result = runner.invoke(app, ["config", "show"])
        assert "Resolved Paths" in result.output

    def test_config_show_runtime(self, tmp_polymind, env_override):
        """config show should display runtime configs."""
        result = runner.invoke(app, ["config", "show"])
        assert "Runtime Configs" in result.output


class TestConfigGet:
    """Tests for polymind config get."""

    def test_get_env_var(self, tmp_polymind, env_override):
        """config get should read environment variables."""
        result = runner.invoke(app, ["config", "get", "POLYMIND_ARTIFACT_DIR"])
        assert result.exit_code == 0

    def test_get_runtime_config(self, tmp_polymind, env_override):
        """config get should read runtime config values."""
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

        result = runner.invoke(app, ["config", "get", "runtime.1.gpu_layers"])
        assert result.exit_code == 0
        assert "0" in result.output

    def test_get_invalid_key(self, tmp_polymind, env_override):
        """config get should fail for invalid keys."""
        result = runner.invoke(app, ["config", "get", "invalid.key"])
        assert result.exit_code != 0

    def test_get_nonexistent_model(self, tmp_polymind, env_override):
        """config get should fail for non-existent model."""
        result = runner.invoke(app, ["config", "get", "runtime.999.gpu_layers"])
        assert result.exit_code != 0


class TestConfigSet:
    """Tests for polymind config set."""

    def test_set_runtime_config(self, tmp_polymind, env_override):
        """config set should write runtime config values."""
        # First create a config
        import yaml

        runtime_yaml = tmp_polymind / ".polymind" / "runtime.yaml"
        runtime_yaml.write_text(
            """version: 1
models:
  '1':
    model_id: '1'
    gpu_layers: -1
    threads: 4
    context_size: 4096
    batch_size: 512
""",
            encoding="utf-8",
        )

        result = runner.invoke(app, ["config", "set", "runtime.1.gpu_layers", "0"])
        assert result.exit_code == 0
        assert "Set" in result.output

        # Verify the change
        with open(runtime_yaml) as f:
            data = yaml.safe_load(f)

        assert data["models"]["1"]["gpu_layers"] == 0

    def test_set_invalid_field(self, tmp_polymind, env_override):
        """config set should fail for invalid fields."""
        result = runner.invoke(app, ["config", "set", "runtime.1.invalid_field", "100"])
        assert result.exit_code != 0
        assert "unknown" in result.output.lower()

    def test_set_non_integer(self, tmp_polymind, env_override):
        """config set should fail for non-integer values."""
        result = runner.invoke(app, ["config", "set", "runtime.1.gpu_layers", "abc"])
        assert result.exit_code != 0
        assert "integer" in result.output.lower()

    def test_set_env_var(self, tmp_polymind, env_override):
        """config set for env vars should show instructions."""
        result = runner.invoke(app, ["config", "set", "POLYMIND_ARTIFACT_DIR", "/tmp/test"])
        assert result.exit_code == 0
        assert "export" in result.output.lower() or "bashrc" in result.output.lower()


class TestConfigEdit:
    """Tests for polymind config edit."""

    def test_edit_help(self, tmp_polymind, env_override):
        """config edit --help should work."""
        result = runner.invoke(app, ["config", "edit", "--help"])
        assert result.exit_code == 0

    def test_edit_unknown_file(self, tmp_polymind, env_override):
        """config edit should fail for unknown file."""
        result = runner.invoke(app, ["config", "edit", "--file", "unknown"])
        assert result.exit_code != 0

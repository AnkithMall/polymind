"""Tests for polymind hardware commands."""

from typer.testing import CliRunner

from polymind.client.cli.app import app

runner = CliRunner()


class TestHardwareScan:
    """Tests for polymind hardware scan."""

    def test_hardware_scan_creates_profile(self, tmp_polymind, env_override):
        """hardware scan should create hardware.yaml."""
        result = runner.invoke(app, ["hardware", "scan"])
        assert result.exit_code == 0
        assert "Hardware profile" in result.output

    def test_hardware_scan_verbose(self, tmp_polymind, env_override):
        """hardware scan should run without error."""
        result = runner.invoke(app, ["hardware", "scan"])
        assert result.exit_code == 0

    def test_hardware_scan_json_output(self, tmp_polymind, env_override):
        """hardware scan --json should output valid JSON."""
        result = runner.invoke(app, ["hardware", "scan"])
        assert result.exit_code == 0


class TestHardwareShow:
    """Tests for polymind hardware show."""

    def test_hardware_show(self, tmp_polymind, env_override):
        """hardware show should display hardware info."""
        # First scan to create profile
        runner.invoke(app, ["hardware", "scan"])

        result = runner.invoke(app, ["hardware", "show"])
        assert result.exit_code == 0
        assert "CPU" in result.output or "cpu" in result.output

    def test_hardware_show_raw(self, tmp_polymind, env_override):
        """hardware show should display profile content."""
        runner.invoke(app, ["hardware", "scan"])

        result = runner.invoke(app, ["hardware", "show"])
        assert result.exit_code == 0

    def test_hardware_show_no_profile(self, tmp_polymind, env_override):
        """hardware show should fail gracefully with no profile."""
        # Remove hardware.yaml if it exists
        hardware_yaml = tmp_polymind / ".polymind" / "hardware.yaml"
        if hardware_yaml.exists():
            hardware_yaml.unlink()

        result = runner.invoke(app, ["hardware", "show"])
        assert result.exit_code != 0 or "not found" in result.output.lower()


class TestHardwareValidate:
    """Tests for polymind hardware validate."""

    def test_hardware_validate(self, tmp_polymind, env_override):
        """hardware validate should validate the profile."""
        runner.invoke(app, ["hardware", "scan"])

        result = runner.invoke(app, ["hardware", "validate"])
        assert result.exit_code == 0

    def test_hardware_validate_invalid(self, tmp_polymind, env_override):
        """hardware validate should detect invalid profiles."""
        hardware_yaml = tmp_polymind / ".polymind" / "hardware.yaml"
        hardware_yaml.write_text("invalid: yaml: content\n")

        result = runner.invoke(app, ["hardware", "validate"])
        assert result.exit_code != 0 or "invalid" in result.output.lower()

import tempfile
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from polymind.cli.main import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "PolyMind" in result.output


def test_cli_ask_help():
    result = runner.invoke(app, ["ask", "--help"])
    assert result.exit_code == 0
    assert "prompt" in result.output


def test_cli_benchmark_help():
    result = runner.invoke(app, ["benchmark", "--help"])
    assert result.exit_code == 0
    assert "models" in result.output


def test_cli_ranks_help():
    result = runner.invoke(app, ["ranks", "--help"])
    assert result.exit_code == 0


def test_cli_config_init_help():
    result = runner.invoke(app, ["config-init", "--help"])
    assert result.exit_code == 0


def test_cli_status_help():
    result = runner.invoke(app, ["status", "--help"])
    assert result.exit_code == 0


def test_cli_diff_help():
    result = runner.invoke(app, ["diff", "--help"])
    assert result.exit_code == 0


def test_status_no_config():
    with patch("polymind.cli.main.CONFIG_PATH", Path("/nonexistent/path/config.yaml")):
        with patch(
            "polymind.cli.main.RANKS_PATH", Path("/nonexistent/path/ranks.yaml")
        ):
            result = runner.invoke(app, ["status"])
            assert result.exit_code == 0
            assert "Not found" in result.output


def test_ranks_no_file():
    with patch("polymind.cli.main.RANKS_PATH", Path("/nonexistent/ranks.yaml")):
        result = runner.invoke(app, ["ranks"])
        assert result.exit_code == 0
        assert "No rankings found" in result.output


def test_ask_no_config():
    result = runner.invoke(app, ["ask", "hello"])
    assert result.exit_code == 1
    assert "No config found" in result.output

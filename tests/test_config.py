import os
import tempfile
from pathlib import Path

import pytest
import yaml

from polymind.core.config import Config, resolve_env_vars
from polymind.core.types import ExecutionStrategy


def test_default_config():
    config = Config()
    assert config.scheduler.strategy == ExecutionStrategy.model_aware
    assert config.router_model == "ollama/llama3.2:1b"
    assert config.models == []


def test_from_yaml(tmp_path: Path):
    data = {
        "models": [{"name": "llama3.2:1b", "provider": "ollama"}],
        "router_model": "ollama/llama3.2:1b",
        "verbose": True,
    }
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(data, f)

    config = Config.from_yaml(cfg_path)
    assert len(config.models) == 1
    assert config.models[0].name == "llama3.2:1b"
    assert config.verbose is True


def test_from_yaml_non_existent():
    config = Config.from_yaml("/nonexistent/path.yaml")
    assert isinstance(config, Config)


def test_to_yaml(tmp_path: Path):
    config = Config(verbose=True)
    cfg_path = tmp_path / "output.yaml"
    config.to_yaml(cfg_path)
    assert cfg_path.exists()

    loaded = Config.from_yaml(cfg_path)
    assert loaded.verbose is True


def test_resolve_env_vars():
    os.environ["POLY_TEST_MODEL"] = "llama3.2:1b"
    result = resolve_env_vars("ollama/${POLY_TEST_MODEL}")
    assert result == "ollama/llama3.2:1b"
    del os.environ["POLY_TEST_MODEL"]


def test_resolve_env_vars_unknown():
    result = resolve_env_vars("${MISSING_VAR}")
    assert result == ""


def test_resolve_env_vars_nested():
    data = {
        "model": "${MODEL_NAME}",
        "nested": {"key": "${NESTED_KEY}"},
    }
    os.environ["MODEL_NAME"] = "test-model"
    os.environ["NESTED_KEY"] = "nested-value"
    result = resolve_env_vars(data)
    assert result["model"] == "test-model"
    assert result["nested"]["key"] == "nested-value"
    del os.environ["MODEL_NAME"]
    del os.environ["NESTED_KEY"]


def test_default_yaml_output():
    yaml_str = Config.default_yaml()
    assert "models" in yaml_str
    assert "router_model" in yaml_str

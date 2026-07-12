from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from polymind.core.types import ExecutionStrategy, ModelSource, RankingMode

logger = logging.getLogger(__name__)


ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def resolve_env_vars(value: Any) -> Any:
    if isinstance(value, str):

        def _replace(m: re.Match) -> str:
            var = m.group(1)
            resolved = os.environ.get(var, "")
            if resolved:
                logger.debug("Resolved env var ${%s}", var)
            return resolved

        return ENV_VAR_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_env_vars(v) for v in value]
    return value


class ProviderConfig(BaseModel):
    provider: str = "ollama"
    model: str = ""
    base_url: str | None = None
    api_key: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout_s: int = 60


class SchedulerConfig(BaseModel):
    strategy: ExecutionStrategy = ExecutionStrategy.model_aware
    pass_context: bool = True
    max_concurrent: int = 1


class ModelConfig(BaseModel):
    name: str
    provider: str = "ollama"
    base_url: str | None = None
    api_key: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout_s: int = 60


class Config(BaseModel):
    models: list[ModelConfig] = Field(default_factory=list)
    router_model: str = "ollama/llama3.2:1b"
    synthesizer_model: str | None = None
    judge_model: str = "ollama/llama3.2:1b"
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    data_dir: str = "~/.polymind"
    verbose: bool = False
    profile: str | None = None
    keep_alive: str | None = None
    litellm_proxy: str | None = None
    ranking_mode: RankingMode = RankingMode.accuracy
    model_source: ModelSource = ModelSource.all

    def get_resolved_profile(self) -> dict[str, Any]:
        profiles: dict[str, dict[str, Any]] = {
            "quality": {
                "scheduler": {"strategy": "model_aware", "pass_context": True},
            },
            "fast": {
                "scheduler": {"strategy": "sequential", "pass_context": False},
                "router_model": "ollama/llama3.2:1b",
            },
            "private": {
                "scheduler": {"strategy": "sequential", "pass_context": True},
            },
        }
        base = {
            "models": self.models,
            "router_model": self.router_model,
            "synthesizer_model": self.synthesizer_model,
            "judge_model": self.judge_model,
            "scheduler": self.scheduler.model_dump(),
            "data_dir": self.data_dir,
            "verbose": self.verbose,
            "keep_alive": self.keep_alive,
            "litellm_proxy": self.litellm_proxy,
            "ranking_mode": self.ranking_mode.value,
            "model_source": self.model_source.value,
        }
        if self.profile and self.profile in profiles:
            merged = {**base}
            for k, v in profiles[self.profile].items():
                if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
                    merged[k] = {**merged[k], **v}
                else:
                    merged[k] = v
            return merged
        return base

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path).expanduser()
        logger.debug("Loading config from %s", path)
        if not path.exists():
            logger.debug("Config file %s not found, returning defaults", path)
            return cls()

        with open(path) as f:
            raw = yaml.safe_load(f)

        if raw is None:
            logger.debug("Config file %s is empty, returning defaults", path)
            return cls()

        resolved = resolve_env_vars(raw)
        logger.debug("Environment variables resolved in config")
        config = cls.model_validate(resolved)
        logger.debug("Config loaded: %d models defined", len(config.models))
        return config

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path).expanduser()
        logger.debug("Saving config to %s", path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(
                self.model_dump(mode="json"),
                f,
                default_flow_style=False,
                sort_keys=False,
            )
        logger.debug("Config saved to %s", path)

    @classmethod
    def default_yaml(cls) -> str:
        return yaml.dump(
            {
                "models": [
                    {"name": "llama3.2:1b", "provider": "ollama"},
                ],
                "router_model": "ollama/llama3.2:1b",
                "synthesizer_model": None,
                "judge_model": "ollama/llama3.2:1b",
                "scheduler": {"strategy": "model_aware", "pass_context": True},
                "data_dir": "~/.polymind",
                "verbose": False,
                "profile": None,
                "keep_alive": None,
                "litellm_proxy": None,
                "ranking_mode": "accuracy",
                "model_source": "all",
            },
            default_flow_style=False,
            sort_keys=False,
        )

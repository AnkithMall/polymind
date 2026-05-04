import os
import yaml
from pydantic import BaseModel, field_validator
from typing import Optional, Dict


class SpecialistConfig(BaseModel):
    model: str
    provider: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    fallback: Optional[str] = None  # openrouter model string used as fallback


class ModelConfig(BaseModel):
    model: str
    provider: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None


class ExecutionConfig(BaseModel):
    mode: str = "sequential"       # "parallel" | "sequential"
    pass_context: bool = False
    timeout_per_task: int = 120
    max_retries: int = 2

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v):
        if v not in ("parallel", "sequential"):
            raise ValueError("execution.mode must be 'parallel' or 'sequential'")
        return v


class PolyMindConfig(BaseModel):
    version: str = "1.0"
    execution: ExecutionConfig = ExecutionConfig()
    analyzer: ModelConfig
    synthesizer: ModelConfig
    specialists: Dict[str, SpecialistConfig]


def _resolve_env(value):
    """Recursively resolve ${ENV_VAR} placeholders in config values."""
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(i) for i in value]
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        resolved = os.environ.get(env_var)
        if not resolved:
            print(f"  Warning: env var {env_var} is not set")
        return resolved or ""
    return value


def load_config(path: str = "config.yaml") -> PolyMindConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    resolved = _resolve_env(raw)
    return PolyMindConfig(**resolved)

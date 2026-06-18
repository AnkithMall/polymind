from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ProviderType(str, Enum):
    ollama = "ollama"
    lm_studio = "lm_studio"
    openrouter = "openrouter"
    openai = "openai"
    anthropic = "anthropic"


@dataclass
class ProviderInfo:
    provider: ProviderType
    model_name: str
    base_url: str | None = None
    api_key: str | None = None

    @property
    def litellm_string(self) -> str:
        if self.provider == ProviderType.ollama:
            return f"ollama/{self.model_name}"
        if self.provider == ProviderType.lm_studio:
            return f"openai/{self.model_name}"
        if self.provider == ProviderType.openrouter:
            return f"openrouter/{self.model_name}"
        if self.provider == ProviderType.openai:
            return f"openai/{self.model_name}"
        if self.provider == ProviderType.anthropic:
            return f"anthropic/{self.model_name}"
        return self.model_name

    def build_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.base_url:
            kwargs["api_base"] = self.base_url
        if self.api_key:
            kwargs["api_key"] = self.api_key
        return kwargs


def resolve_model_string(model_ref: str, provider: str | None = None) -> ProviderInfo:
    if "/" in model_ref:
        prov, name = model_ref.split("/", 1)
        try:
            ptype = ProviderType(prov)
        except ValueError:
            ptype = ProviderType.ollama
        return ProviderInfo(provider=ptype, model_name=name)

    ptype = ProviderType(provider) if provider else ProviderType.ollama
    return ProviderInfo(provider=ptype, model_name=model_ref)

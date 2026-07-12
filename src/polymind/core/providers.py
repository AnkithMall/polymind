from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from polymind.core.types import ModelSource

logger = logging.getLogger(__name__)


class ProviderType(str, Enum):
    ollama = "ollama"
    lm_studio = "lm_studio"
    openrouter = "openrouter"
    openai = "openai"
    anthropic = "anthropic"


LOCAL_PROVIDERS = {ProviderType.ollama, ProviderType.lm_studio}
ONLINE_PROVIDERS = {
    ProviderType.openai,
    ProviderType.anthropic,
    ProviderType.openrouter,
}


def provider_model_source(provider: ProviderType) -> ModelSource:
    if provider in LOCAL_PROVIDERS:
        return ModelSource.local
    return ModelSource.online


@dataclass
class ProviderInfo:
    provider: ProviderType
    model_name: str
    base_url: str | None = None
    api_key: str | None = None

    @property
    def model_source(self) -> ModelSource:
        return provider_model_source(self.provider)

    @property
    def litellm_string(self) -> str:
        result: str
        if self.provider == ProviderType.ollama:
            result = f"ollama/{self.model_name}"
        elif self.provider == ProviderType.lm_studio:
            result = f"openai/{self.model_name}"
        elif self.provider == ProviderType.openrouter:
            result = f"openrouter/{self.model_name}"
        elif self.provider == ProviderType.openai:
            result = f"openai/{self.model_name}"
        elif self.provider == ProviderType.anthropic:
            result = f"anthropic/{self.model_name}"
        else:
            result = self.model_name
        logger.debug(
            "litellm_string for %s/%s -> %s",
            self.provider.value,
            self.model_name,
            result,
        )
        return result

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
        logger.debug(
            "Resolved model ref %s -> provider=%s, model=%s",
            model_ref,
            ptype.value,
            name,
        )
        return ProviderInfo(provider=ptype, model_name=name)

    ptype = ProviderType(provider) if provider else ProviderType.ollama
    logger.debug(
        "Resolved model ref %s -> provider=%s (from provider arg)",
        model_ref,
        ptype.value,
    )
    return ProviderInfo(provider=ptype, model_name=model_ref)

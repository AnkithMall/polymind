from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Rough token estimation: ~4 chars per token for English text
CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str) -> int:
    return len(text) // int(CHARS_PER_TOKEN) + 1


def truncate_to_fit(text: str, max_tokens: int) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text

    max_chars = max_tokens * int(CHARS_PER_TOKEN)
    return text[:max_chars] + "\n\n[truncated...]"


@dataclass
class ContextBudget:
    model: str
    max_context_tokens: int
    reserved_output_tokens: int = 1024

    used_tokens: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def available_tokens(self) -> int:
        return self.max_context_tokens - self.reserved_output_tokens - self.used_tokens

    def can_fit(self, text: str) -> bool:
        return estimate_tokens(text) <= self.available_tokens

    def add_usage(self, text: str) -> None:
        tokens = estimate_tokens(text)
        self.used_tokens += tokens

    def fit_or_truncate(self, text: str, label: str = "content") -> str:
        if self.can_fit(text):
            return text

        max_for_input = self.available_tokens
        truncated = truncate_to_fit(text, max_for_input)
        self.warnings.append(
            f"{label} truncated from {estimate_tokens(text)} to "
            f"{estimate_tokens(truncated)} tokens to fit {self.model} "
            f"context window ({self.max_context_tokens})"
        )
        logger.warning(self.warnings[-1])
        return truncated


DEFAULT_CONTEXT_LIMITS: dict[str, int] = {
    "gpt-4o": 128000,
    "gpt-4": 8192,
    "gpt-3.5": 16384,
    "claude": 100000,
    "mixtral": 32768,
    "mistral": 8192,
    "qwen": 32768,
    "codellama": 16384,
    "llama3": 8192,
    "llama2": 4096,
    "gemma": 8192,
    "phi": 2048,
    "deepseek": 4096,
    "llava": 4096,
    "neural": 8192,
    "solar": 4096,
    "yi": 4096,
}


def get_model_context_limit(model_name: str, default: int = 4096) -> int:
    model_lower = model_name.lower()
    sorted_keys = sorted(DEFAULT_CONTEXT_LIMITS.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if key in model_lower:
            return DEFAULT_CONTEXT_LIMITS[key]
    return default

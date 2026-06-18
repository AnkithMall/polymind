from __future__ import annotations

import logging
import time
from typing import Any

import litellm

from polymind.core.fallback import retry_with_backoff
from polymind.core.providers import ProviderInfo, resolve_model_string
from polymind.core.types import Subtask, SubtaskResult

logger = logging.getLogger(__name__)


async def execute_subtask(
    subtask: Subtask,
    model_ref: str = "ollama/llama3.2:1b",
    provider_info: ProviderInfo | None = None,
    max_retries: int = 2,
    keep_alive: str | None = None,
    **litellm_kwargs: Any,
) -> SubtaskResult:
    if provider_info is None:
        provider_info = resolve_model_string(model_ref)

    model = provider_info.litellm_string
    kwargs = provider_info.build_kwargs()
    if keep_alive is not None:
        kwargs["keep_alive"] = keep_alive
    kwargs.update(litellm_kwargs)

    async def _call() -> str:
        start = time.monotonic()
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": subtask.prompt}],
            **kwargs,
        )
        elapsed = time.monotonic() - start
        content: str = response.choices[0].message.content or ""
        return content

    try:
        output = await retry_with_backoff(
            _call, max_retries=max_retries, base_delay_s=1.0
        )
    except Exception as e:
        logger.error(
            "Subtask %s failed after %d retries: %s", subtask.id, max_retries, e
        )
        return SubtaskResult(
            subtask_id=subtask.id,
            model=model,
            output="",
            error=str(e),
        )

    return SubtaskResult(
        subtask_id=subtask.id,
        model=model,
        output=output,
    )


async def execute_subtask_with_context(
    subtask: Subtask,
    prior_outputs: dict[str, SubtaskResult],
    model_ref: str = "ollama/llama3.2:1b",
    provider_info: ProviderInfo | None = None,
    **litellm_kwargs: Any,
) -> SubtaskResult:
    context_parts: list[str] = []
    for dep_id in subtask.depends_on:
        dep_result = prior_outputs.get(dep_id)
        if dep_result and dep_result.output:
            context_parts.append(
                f"### Output from {dep_id} (using {dep_result.model})\n{dep_result.output}"
            )

    enriched_prompt = subtask.prompt
    if context_parts:
        enriched_prompt = (
            "Here is context from previously completed subtasks:\n\n"
            + "\n\n".join(context_parts)
            + "\n\n---\n\nNow complete this task:\n"
            + subtask.prompt
        )

    enriched = subtask.model_copy(update={"prompt": enriched_prompt})
    return await execute_subtask(
        enriched, model_ref=model_ref, provider_info=provider_info, **litellm_kwargs
    )

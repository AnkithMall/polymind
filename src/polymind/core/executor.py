from __future__ import annotations

import logging
import time
from typing import Any

import litellm

from polymind.core.benchmark import calculate_cost
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

    async def _call() -> tuple[str, int | None, int | None]:
        start = time.monotonic()
        logger.debug(
            "Calling model %s for subtask %s (domain=%s)",
            model, subtask.id, subtask.domain.value,
        )
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": subtask.prompt}],
            **kwargs,
        )
        elapsed = time.monotonic() - start
        content: str = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        in_tokens = usage.prompt_tokens if usage else None
        out_tokens = usage.completion_tokens if usage else None
        logger.debug(
            "Response from %s for subtask %s: tokens=%d/%d latency=%.1fs",
            model, subtask.id, in_tokens or 0, out_tokens or 0, elapsed,
        )
        return content, in_tokens, out_tokens

    try:
        output, in_tokens, out_tokens = await retry_with_backoff(
            _call, max_retries=max_retries, base_delay_s=1.0
        )
        logger.debug(
            "Subtask %s completed successfully on %s", subtask.id, model,
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

    task_cost = None
    if in_tokens is not None and out_tokens is not None:
        task_cost = round(calculate_cost(model, in_tokens, out_tokens), 10)

    return SubtaskResult(
        subtask_id=subtask.id,
        model=model,
        output=output,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        cost=task_cost,
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

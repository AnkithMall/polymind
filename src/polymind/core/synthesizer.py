from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import litellm

from polymind.core.types import PipelineResult, SubtaskResult

logger = logging.getLogger(__name__)

SYNTHESIZER_SYSTEM_PROMPT = """You are a response synthesizer. You will be given:
1. The original user prompt
2. A set of subtask outputs, each produced by a different specialist model

Your job is to merge these outputs into a single coherent, well-structured response.
- Remove redundant information
- Ensure logical flow between sections
- Maintain technical accuracy
- Credit specific subtask outputs where relevant

Provide only the final merged response, no additional commentary."""


def _build_synthesis_messages(
    prompt: str,
    results: list[SubtaskResult],
) -> list[dict[str, str]]:
    subtask_block = ""
    for r in results:
        label = f"[{r.subtask_id}] — model: {r.model}"
        if r.error:
            subtask_block += f"{label}\nERROR: {r.error}\n\n"
        else:
            subtask_block += f"{label}\n{r.output}\n\n"

    user_content = f"Original prompt:\n{prompt}\n\nSubtask outputs:\n{subtask_block}"

    return [
        {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


async def synthesize(
    result: PipelineResult,
    synthesizer_model: str = "ollama/llama3.2:1b",
    **litellm_kwargs: Any,
) -> PipelineResult:
    messages = _build_synthesis_messages(result.original_prompt, result.subtask_results)

    try:
        response = await litellm.acompletion(
            model=synthesizer_model,
            messages=messages,
            **litellm_kwargs,
        )
        content: str = response.choices[0].message.content or ""
        result.synthesis = content
    except Exception as e:
        logger.error("Synthesis failed: %s", e)
        result.synthesis = None

    return result


async def synthesize_streaming(
    result: PipelineResult,
    synthesizer_model: str = "ollama/llama3.2:1b",
    **litellm_kwargs: Any,
) -> AsyncIterator[str]:
    messages = _build_synthesis_messages(result.original_prompt, result.subtask_results)

    try:
        response = await litellm.acompletion(
            model=synthesizer_model,
            messages=messages,
            stream=True,
            **litellm_kwargs,
        )

        full_content: list[str] = []
        async for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full_content.append(delta)
                yield delta

        result.synthesis = "".join(full_content)
    except Exception as e:
        logger.error("Streaming synthesis failed: %s", e)
        yield f"\n\n[Synthesis error: {e}]"


def _build_synthesis_messages_json(
    prompt: str,
    results: list[SubtaskResult],
) -> list[dict[str, str]]:
    subtask_data = []
    for r in results:
        subtask_data.append(
            {
                "id": r.subtask_id,
                "model": r.model,
                "output": r.output,
                "error": r.error,
            }
        )

    user_content = json.dumps(
        {
            "original_prompt": prompt,
            "subtask_outputs": subtask_data,
        },
        indent=2,
    )

    return [
        {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

import litellm
from typing import List, AsyncGenerator
from .config import PolyMindConfig
from .analyzer import _build_litellm_args
from .executor import SubtaskResult

SYNTHESIZER_SYSTEM_PROMPT = """You are a synthesis engine. You receive a user's original prompt and outputs from specialist AI models that each handled a specific subtask.

Your job:
- Combine all specialist outputs into a single, coherent, well-structured response
- Resolve any contradictions or overlaps between specialist outputs
- Present the information naturally as if one expert wrote it
- Format clearly: use markdown headings, code blocks, and bullet points where appropriate
- Do NOT mention that multiple models were used or that there were subtasks"""


def _build_synthesis_input(original_prompt: str, results: List[SubtaskResult]) -> str:
    parts = [f"User's original prompt:\n{original_prompt}\n"]
    for r in results:
        status = " [FALLBACK]" if r.fallback_used else ""
        status += " [ERROR]" if r.error and r.model_used == "none" else ""
        parts.append(
            f"--- {r.domain.upper()} subtask{status} ---\n"
            f"Task: {r.description}\n\n"
            f"Output:\n{r.output}"
        )
    return "\n\n".join(parts)


async def synthesize_stream(
    original_prompt: str,
    results: List[SubtaskResult],
    config: PolyMindConfig,
) -> AsyncGenerator[str, None]:
    """Stream the synthesized response token by token."""

    # If there's only one successful subtask, stream it directly
    if len(results) == 1 and not results[0].fallback_used and results[0].model_used != "none":
        # Stream the single result as chunks
        output = results[0].output
        chunk_size = 4
        for i in range(0, len(output), chunk_size):
            yield output[i:i + chunk_size]
        return

    model_str, kwargs = _build_litellm_args(config.synthesizer)
    synthesis_input = _build_synthesis_input(original_prompt, results)

    try:
        response = await litellm.acompletion(
            model=model_str,
            messages=[
                {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
                {"role": "user", "content": synthesis_input},
            ],
            stream=True,
            **kwargs,
        )

        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    except Exception as e:
        print(f"Synthesizer error: {e}")
        # Fall back to concatenating results
        fallback_output = "\n\n".join(
            f"**{r.domain.capitalize()}:**\n{r.output}"
            for r in results
            if r.model_used != "none"
        )
        yield fallback_output

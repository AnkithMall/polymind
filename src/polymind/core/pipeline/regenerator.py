"""Regenerator — combines task results into a final coherent response.

Takes the outputs of all subtasks and synthesizes them into a single,
well-structured response to the user's original prompt.
"""

from __future__ import annotations

from pathlib import Path

from llama_cpp import Llama

from polymind.core.pipeline.types import Task, TaskStatus

REGENERATE_SYSTEM = """You are a response synthesizer. Your job is to combine the results
of multiple subtasks into a single, coherent, well-structured response.

Rules:
1. Preserve ALL important information from each subtask
2. Organize the response logically — group related information
3. Remove redundancy and contradictions
4. Match the tone and style the user expects
5. If subtasks conflict, prefer the most specific/confident result
6. Add smooth transitions between sections
7. The final response should feel like it was written by one expert

Format:
- Use markdown headers for major sections if the response is long
- Use bullet points for lists
- Use code blocks for code
- Keep paragraphs focused and concise
"""

REGENERATE_USER = """Original user prompt:
{prompt}

Subtask results:
{results}

Synthesize these into a single coherent response. Respond with ONLY the final answer."""


def regenerate_response(
    prompt: str,
    tasks: list[Task],
    model_path: Path,
    model_config: dict | None = None,
) -> str:
    """Combine task results into a final response.

    Args:
        prompt: The original user prompt.
        tasks: Completed tasks with results.
        model_path: Path to the GGUF model to use.
        model_config: Optional runtime config overrides.

    Returns:
        The synthesized final response.
    """
    config = model_config or {}

    # Collect results from completed tasks
    completed = [t for t in tasks if t.status == TaskStatus.COMPLETED and t.result]

    if not completed:
        return "No task results available to synthesize."

    # If only one task, return its result directly
    if len(completed) == 1:
        return completed[0].result

    # Format results for the regenerator
    results_text = _format_results(completed)

    # If results are short enough, return them directly (no LLM needed)
    if len(results_text) < 500:
        return results_text

    # Use LLM to synthesize
    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=config.get("gpu_layers", 0),
        n_threads=config.get("threads", 4),
        n_ctx=config.get("context_size", 4096),
        n_batch=config.get("batch_size", 256),
        verbose=False,
    )

    messages = [
        {"role": "system", "content": REGENERATE_SYSTEM},
        {
            "role": "user",
            "content": REGENERATE_USER.format(prompt=prompt, results=results_text),
        },
    ]

    try:
        output = llm.create_chat_completion(
            messages=messages,
            temperature=0.3,
            max_tokens=2048,
        )
        content = output["choices"][0]["message"]["content"] or ""
        return content.strip() if content.strip() else results_text
    except Exception:
        # Fallback: concatenate results
        return results_text
    finally:
        del llm


def _format_results(tasks: list[Task]) -> str:
    """Format task results for the regenerator."""
    parts: list[str] = []

    for i, task in enumerate(tasks, 1):
        domain = task.domain
        desc = task.metadata.get("description", task.prompt[:100])
        parts.append(f"### Subtask {i} [{domain}]: {desc}")
        parts.append(task.result)
        parts.append("")

    return "\n".join(parts)


def quick_synthesize(tasks: list[Task]) -> str:
    """Quick synthesis without LLM — just concatenates results.

    Used when:
    - No regenerator model is available
    - Results are short
    - Speed is prioritized over quality
    """
    completed = [t for t in tasks if t.status == TaskStatus.COMPLETED and t.result]

    if not completed:
        return "No results available."

    if len(completed) == 1:
        return completed[0].result

    parts: list[str] = []
    for _i, task in enumerate(completed, 1):
        domain = task.domain
        parts.append(f"**[{domain}]** {task.result}")

    return "\n\n".join(parts)

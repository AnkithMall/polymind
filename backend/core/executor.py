import asyncio
import time
import litellm
from pydantic import BaseModel
from typing import List, Dict, Optional
from .config import PolyMindConfig, SpecialistConfig
from .analyzer import Subtask, _build_litellm_args


class SubtaskResult(BaseModel):
    subtask_id: str
    domain: str
    description: str
    output: str
    model_used: str
    provider: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    fallback_used: bool = False
    error: Optional[str] = None


def _specialist_litellm_args(specialist: SpecialistConfig) -> tuple[str, dict]:
    """Build litellm model string and kwargs from a SpecialistConfig."""
    from .config import ModelConfig
    mc = ModelConfig(
        model=specialist.model,
        provider=specialist.provider,
        base_url=specialist.base_url,
        api_key=specialist.api_key,
    )
    from .analyzer import _build_litellm_args
    return _build_litellm_args(mc)


async def _run_single_subtask(
    subtask: Subtask,
    specialist: SpecialistConfig,
    prior_context: Optional[str],
    timeout: int,
) -> SubtaskResult:
    model_str, kwargs = _specialist_litellm_args(specialist)

    messages = []
    if prior_context:
        messages.append({
            "role": "system",
            "content": f"Context from previous subtasks:\n\n{prior_context}"
        })
    messages.append({"role": "user", "content": subtask.description})

    start = time.time()
    fallback_used = False
    error_msg = None

    try:
        response = await asyncio.wait_for(
            litellm.acompletion(model=model_str, messages=messages, **kwargs),
            timeout=timeout,
        )
        output = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        model_used = specialist.model

    except Exception as e:
        error_msg = str(e)
        print(f"  Specialist error ({subtask.domain}): {e}")

        # Try fallback if configured
        if specialist.fallback:
            try:
                fallback_model = (
                    specialist.fallback
                    if specialist.fallback.startswith("openrouter/")
                    else f"openrouter/{specialist.fallback}"
                )
                fb_response = await asyncio.wait_for(
                    litellm.acompletion(
                        model=fallback_model,
                        messages=messages,
                        api_key=specialist.api_key,
                    ),
                    timeout=timeout,
                )
                output = fb_response.choices[0].message.content or ""
                usage = fb_response.usage
                input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(usage, "completion_tokens", 0) or 0
                model_used = specialist.fallback
                fallback_used = True
            except Exception as fe:
                output = f"[Both primary and fallback failed: {fe}]"
                input_tokens = output_tokens = 0
                model_used = "none"
        else:
            output = f"[Error: {error_msg}]"
            input_tokens = output_tokens = 0
            model_used = "none"

    latency_ms = int((time.time() - start) * 1000)

    return SubtaskResult(
        subtask_id=subtask.id,
        domain=subtask.domain,
        description=subtask.description,
        output=output,
        model_used=model_used,
        provider=specialist.provider,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        fallback_used=fallback_used,
        error=error_msg,
    )


def _get_specialist(domain: str, config: PolyMindConfig) -> SpecialistConfig:
    """Return the specialist config for a domain, falling back to 'general'."""
    specialist = config.specialists.get(domain) or config.specialists.get("general")
    if not specialist:
        # Last resort: use synthesizer config wrapped as specialist
        return SpecialistConfig(
            model=config.synthesizer.model,
            provider=config.synthesizer.provider,
            base_url=config.synthesizer.base_url,
            api_key=config.synthesizer.api_key,
        )
    return specialist


async def execute_plan(
    subtasks: List[Subtask],
    config: PolyMindConfig,
) -> List[SubtaskResult]:
    """Execute the subtask plan in sequential or parallel mode."""
    timeout = config.execution.timeout_per_task

    if config.execution.mode == "sequential":
        results: List[SubtaskResult] = []
        context_parts: List[str] = []

        for subtask in subtasks:
            specialist = _get_specialist(subtask.domain, config)
            prior = "\n\n".join(context_parts) if config.execution.pass_context and context_parts else None

            print(f"  → Running [{subtask.domain}] via {specialist.model} ...")
            result = await _run_single_subtask(subtask, specialist, prior, timeout)
            results.append(result)

            if config.execution.pass_context:
                context_parts.append(f"[{subtask.domain.upper()}] {result.output[:500]}")

        return results

    else:
        # Parallel with dependency-aware batching
        result_map: Dict[str, SubtaskResult] = {}
        completed_ids: set = set()
        pending = list(subtasks)

        while pending:
            ready = [t for t in pending if all(d in completed_ids for d in t.depends_on)]
            if not ready:
                print("  Warning: dependency deadlock detected, running remaining tasks anyway")
                ready = pending[:]

            print(f"  → Parallel batch: {[t.domain for t in ready]}")
            coroutines = [
                _run_single_subtask(t, _get_specialist(t.domain, config), None, timeout)
                for t in ready
            ]
            batch = await asyncio.gather(*coroutines)

            for subtask, result in zip(ready, batch):
                result_map[subtask.id] = result
                completed_ids.add(subtask.id)
                pending.remove(subtask)

        return [result_map[s.id] for s in subtasks if s.id in result_map]

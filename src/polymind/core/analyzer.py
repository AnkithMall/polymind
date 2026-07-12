from __future__ import annotations

import json
import logging
from typing import Any

import litellm

from polymind.core.fallback import fallback_chain
from polymind.core.types import ALL_DOMAINS, AnalyzerPlan, DomainType, Subtask

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """You are a task decomposition specialist. Given a user prompt, break it down into subtasks.

For each subtask, provide:
- id: a unique short identifier (e.g. "t1", "t2")
- domain: one of {domains}
- prompt: the specific instruction for this subtask
- depends_on: list of subtask ids that must complete first (empty list for independent tasks)

Rules:
- Keep the original prompt's intent intact across subtasks
- Only use valid domains from the list above
- If the prompt is simple, return a single subtask with domain "general"
- Ensure dependency chains are acyclic

Respond with valid JSON only:
{{"subtasks": [{{"id": "...", "domain": "...", "prompt": "...", "depends_on": [...]}}]}}"""


def _build_router_messages(prompt: str) -> list[dict[str, str]]:
    domains_str = ", ".join(d.value for d in ALL_DOMAINS)
    system = ROUTER_SYSTEM_PROMPT.format(domains=domains_str)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]


def _parse_plan(raw: str, original_prompt: str) -> AnalyzerPlan | None:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    subtasks_data = data.get("subtasks", [])
    if not subtasks_data:
        return None

    subtasks: list[Subtask] = []
    for item in subtasks_data:
        domain_str = item.get("domain", "general")
        try:
            domain = DomainType(domain_str)
        except ValueError:
            domain = DomainType.general

        subtasks.append(
            Subtask(
                id=item.get("id", f"t{len(subtasks) + 1}"),
                domain=domain,
                prompt=item.get("prompt", ""),
                depends_on=item.get("depends_on", []),
            )
        )

    return AnalyzerPlan(original_prompt=original_prompt, subtasks=subtasks)


def _single_task_fallback(prompt: str) -> AnalyzerPlan:
    return AnalyzerPlan(
        original_prompt=prompt,
        subtasks=[
            Subtask(id="t1", domain=DomainType.general, prompt=prompt),
        ],
    )


async def analyze_prompt(
    prompt: str,
    router_model: str = "ollama/llama3.2:1b",
    **litellm_kwargs: Any,
) -> AnalyzerPlan:
    messages = _build_router_messages(prompt)
    logger.debug(
        "Sending router prompt to %s (prompt length=%d chars)",
        router_model,
        len(prompt),
    )

    async def call_router() -> str:
        response = await litellm.acompletion(
            model=router_model,
            messages=messages,
            **litellm_kwargs,
        )
        content: str = response.choices[0].message.content or ""
        return content

    async def call_fallback() -> str:
        response = await litellm.acompletion(
            model=router_model,
            messages=messages,
            max_tokens=512,
            temperature=0.0,
            **litellm_kwargs,
        )
        content: str = response.choices[0].message.content or ""
        return content

    try:
        raw = await call_router()
        logger.debug("Raw router response (%d chars): %s", len(raw), raw[:200])
    except Exception:
        logger.warning("Router LLM call failed, using fallback with lower temperature")
        try:
            raw = await call_fallback()
            logger.debug(
                "Fallback router response (%d chars): %s", len(raw), raw[:200]
            )
        except Exception:
            logger.error(
                "Router LLM completely unavailable, returning single-task plan"
            )
            return _single_task_fallback(prompt)

    plan = _parse_plan(raw, prompt)
    if plan is None:
        logger.warning(
            "Could not parse router response as JSON, using single-task fallback"
        )
        logger.debug("Parse failed for raw response: %s", raw[:300])
        return _single_task_fallback(prompt)

    logger.debug(
        "Router plan parsed successfully: %d subtasks", len(plan.subtasks)
    )
    return plan

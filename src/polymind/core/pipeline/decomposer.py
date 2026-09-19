"""Decomposer — breaks a user prompt into structured tasks.

The decomposer uses a local LLM to analyze the prompt and generate
a task plan with domains and dependencies.

Domain knowledge is loaded dynamically from the category/domain
registry so that custom domains are automatically recognized.
"""

from __future__ import annotations

import re
from pathlib import Path

from llama_cpp import Llama

from polymind.core.pipeline.types import ModelRole, Task, TaskType

# System prompt template for the decomposer — domains are injected dynamically
_DECOMPOSE_SYSTEM_TEMPLATE = """You are a task decomposition engine. Your job is to break down
a user's request into clear, independent subtasks that can be worked on in parallel.

For each subtask, provide:
- A brief description (id will be auto-generated)
- The domain it belongs to (choose from the available domains listed below)
- Whether it depends on other subtasks

Available domains:
{domain_list}

Rules:
1. Keep subtasks focused and self-contained
2. Minimize dependencies between subtasks
3. Each subtask should map to exactly one domain from the list above
4. If the prompt is simple (single question), return just one task
5. Be specific — vague tasks produce poor results
6. Use the most specific domain available for each subtask
7. If a subtask spans multiple domains, pick the primary one

Output format (one task per line):
TASK: <description>
DOMAIN: <domain>
DEPENDS: <comma-separated task numbers, or NONE>
"""

DECOMPOSE_USER = """Decompose this prompt into subtasks:

{prompt}

Respond with ONLY the task definitions, no other text."""


def _build_domain_list() -> str:
    """Build a formatted list of available domains from the registry."""
    from polymind.core.confidence.artifact import load_all_domains

    domains = load_all_domains()
    lines: list[str] = []
    for d in domains:
        aliases_str = ""
        if d.aliases:
            aliases_str = f" (aliases: {', '.join(d.aliases)})"
        lines.append(f"- {d.id}: {d.description}{aliases_str}")
    return "\n".join(lines) if lines else "- general: General purpose tasks"


def decompose_prompt(
    prompt: str,
    model_path: Path,
    model_config: dict | None = None,
) -> list[Task]:
    """Decompose a user prompt into structured tasks.

    Args:
        prompt: The user's input prompt.
        model_path: Path to the GGUF model to use.
        model_config: Optional runtime config overrides.

    Returns:
        List of Task objects with domains and dependencies.
    """
    config = model_config or {}

    # Build domain list dynamically from the registry
    domain_list = _build_domain_list()
    system_prompt = _DECOMPOSE_SYSTEM_TEMPLATE.format(domain_list=domain_list)

    # Try GPU first, fall back to CPU on CUDA errors
    llm = None
    try:
        llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=config.get("gpu_layers", 0),
            n_threads=config.get("threads", 4),
            n_ctx=config.get("context_size", 4096),
            n_batch=config.get("batch_size", 256),
            verbose=False,
        )
    except Exception:
        try:
            llm = Llama(
                model_path=str(model_path),
                n_gpu_layers=0,
                n_threads=config.get("threads", 4),
                n_ctx=config.get("context_size", 4096),
                n_batch=config.get("batch_size", 256),
                verbose=False,
            )
        except Exception:
            # Fallback: treat entire prompt as a single task
            return [_single_task(prompt)]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": DECOMPOSE_USER.format(prompt=prompt)},
    ]

    try:
        output = llm.create_chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
        )
        content = output["choices"][0]["message"]["content"] or ""
    except Exception:
        # Fallback: treat entire prompt as a single task
        return [_single_task(prompt)]
    finally:
        del llm

    return _parse_tasks(content, prompt)


def _single_task(prompt: str) -> Task:
    """Create a single task from a prompt (fallback)."""
    return Task(
        id="task_1",
        prompt=prompt,
        domain=_detect_domain(prompt),
        task_type=TaskType.GENERATE,
        model_role=ModelRole.GENERATOR,
    )


def _detect_domain(prompt: str) -> str:
    """Detect domain from prompt using keyword heuristics.

    First tries to match against registered domain aliases,
    then falls back to built-in keyword matching.
    """
    lower = prompt.lower()

    # Try matching against registered domains and their aliases
    from polymind.core.confidence.artifact import load_all_domains

    domains = load_all_domains()
    for domain in domains:
        # Check domain ID (word boundary match)
        if re.search(r"\b" + re.escape(domain.id.lower()) + r"\b", lower):
            return domain.id
        # Check aliases (word boundary match to avoid "ui" matching "quick")
        for alias in domain.aliases:
            if re.search(r"\b" + re.escape(alias.lower()) + r"\b", lower):
                return domain.id

    # Fallback: use built-in keyword heuristics
    knowledge_keywords = ["what is", "who is", "when did", "where is", "how does", "define "]
    if any(k in lower for k in knowledge_keywords):
        return "knowledge"

    math_keywords = [
        "calculate",
        "math",
        "equation",
        "solve",
        "compute",
        "sum",
        "average",
        "integral",
        "derivative",
    ]
    if any(k in lower for k in math_keywords):
        return "mathematics"

    reason_keywords = [
        "why does",
        "explain why",
        "analyze why",
        "reasoning",
        "logic puzzle",
        "syllogism",
    ]
    if any(k in lower for k in reason_keywords):
        return "reasoning"

    write_keywords = [
        "write a story",
        "write a poem",
        "creative writing",
        "draft an essay",
        "compose",
    ]
    if any(k in lower for k in write_keywords):
        return "writing"

    safety_keywords = ["is it safe", "security risk", "harmful", "dangerous", "ethical"]
    if any(k in lower for k in safety_keywords):
        return "safety"

    code_keywords = [
        "code",
        "function",
        "class",
        "debug",
        "python",
        "javascript",
        "program",
        "implement",
        "api",
        "endpoint",
        "database",
        "sql",
        "algorithm",
        "refactor",
        "test",
        "unit test",
        "integration test",
        "git",
        "docker",
        "deploy",
        "build a",
        "create a",
        "write a function",
        "write code",
        "fix the bug",
        "error in",
        "stack trace",
    ]
    if any(k in lower for k in code_keywords):
        return "coding"

    conv_keywords = ["hello", "hi ", "hey ", "how are you", "tell me a joke", "chat"]
    if any(k in lower for k in conv_keywords):
        return "conversation"

    instruction_keywords = ["list", "give me", "tell me", "show me", "name "]
    if any(k in lower for k in instruction_keywords):
        return "instruction"

    return "general"


def _parse_tasks(content: str, original_prompt: str) -> list[Task]:
    """Parse the decomposer output into Task objects."""
    tasks: list[Task] = []
    current: dict[str, str] = {}

    for line in content.strip().split("\n"):
        line = line.strip()

        if line.startswith("TASK:"):
            if current:
                tasks.append(_make_task(current, len(tasks) + 1, original_prompt))
            current = {"description": line[5:].strip()}
        elif line.startswith("DOMAIN:"):
            current["domain"] = line[7:].strip()
        elif line.startswith("DEPENDS:"):
            current["depends"] = line[8:].strip()

    if current:
        tasks.append(_make_task(current, len(tasks) + 1, original_prompt))

    if not tasks:
        return [_single_task(original_prompt)]

    # Resolve dependencies
    {t.id: t for t in tasks}
    for task in tasks:
        deps_str = task.metadata.get("depends", "NONE")
        if deps_str and deps_str.upper() != "NONE":
            for dep_id in deps_str.split(","):
                dep_id = dep_id.strip()
                if dep_id:
                    # Try to resolve by index or name
                    for t in tasks:
                        if t.id.endswith(dep_id) or t.id == f"task_{dep_id}":
                            task.depends_on.append(t.id)
                            break

    return tasks


def _make_task(data: dict[str, str], index: int, original_prompt: str) -> Task:
    """Create a Task from parsed decomposer output."""
    task_id = f"task_{index}"
    description = data.get("description", "")
    domain = data.get("domain", "general")

    # If description is empty, use original prompt
    if not description:
        description = original_prompt

    # Build the task prompt
    if len(data) > 1:
        # Multiple tasks — each gets a focused prompt
        task_prompt = f"Context: {original_prompt}\n\nTask: {description}"
    else:
        # Single task — use original prompt
        task_prompt = original_prompt

    return Task(
        id=task_id,
        prompt=task_prompt,
        domain=domain,
        task_type=TaskType.GENERATE,
        model_role=ModelRole.GENERATOR,
        metadata={"description": description, "depends": data.get("depends", "NONE")},
    )

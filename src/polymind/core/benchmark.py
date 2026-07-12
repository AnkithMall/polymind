from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import litellm
import yaml

from polymind.core.fallback import retry_with_backoff
from polymind.core.providers import LOCAL_PROVIDERS, ProviderType
from polymind.core.types import ALL_DOMAINS, DomainType, RankEntry, RankStore

logger = logging.getLogger(__name__)

MODEL_PRICING: dict[str, tuple[float, float]] = {
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-3.5-turbo": (0.50, 1.50),
    "anthropic/claude-3-opus-20240229": (15.00, 75.00),
    "anthropic/claude-3-sonnet-20240229": (3.00, 15.00),
    "anthropic/claude-3-haiku-20240307": (0.25, 1.25),
    "anthropic/claude-3-5-sonnet-20241022": (3.00, 15.00),
    "anthropic/claude-3-5-haiku-20241022": (0.80, 4.00),
    "openrouter/mixtral-8x7b-instruct": (0.24, 0.24),
    "openrouter/mistral-7b-instruct": (0.04, 0.04),
    "openrouter/llama-3.1-70b": (0.23, 0.50),
    "openrouter/llama-3.1-8b": (0.02, 0.06),
}


def get_model_pricing(model: str) -> tuple[float, float]:
    if "/" in model:
        provider_prefix = model.split("/", 1)[0]
        try:
            ptype = ProviderType(provider_prefix)
            if ptype in LOCAL_PROVIDERS:
                return (0.0, 0.0)
        except ValueError:
            pass
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    return (0.0, 0.0)


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    input_price, output_price = get_model_pricing(model)
    return (input_tokens / 1000.0 * input_price) + (
        output_tokens / 1000.0 * output_price
    )


@dataclass
class BenchmarkTask:
    domain: DomainType
    prompt: str
    expected_answer: str
    scoring: str = "exact_match"


EXACT_MATCH_DOMAINS = {DomainType.code, DomainType.math, DomainType.qa}
JUDGE_DOMAINS = {
    DomainType.creative,
    DomainType.reasoning,
    DomainType.research,
    DomainType.summarization,
    DomainType.translation,
    DomainType.general,
}

BUILTIN_TASKS: dict[DomainType, list[BenchmarkTask]] = {
    DomainType.code: [
        BenchmarkTask(
            DomainType.code,
            "Write a Python function to check if a number is prime",
            "def is_prime(n):",
            "exact_match",
        ),
        BenchmarkTask(
            DomainType.code,
            "Write a function that reverses a string in Python",
            "def reverse_string(s):",
            "exact_match",
        ),
        BenchmarkTask(
            DomainType.code,
            "Implement binary search in Python",
            "def binary_search(arr, target):",
            "exact_match",
        ),
        BenchmarkTask(
            DomainType.code,
            "Write a Python function to find the Fibonacci number at position n",
            "def fibonacci(n):",
            "exact_match",
        ),
        BenchmarkTask(
            DomainType.code,
            "Write a function to remove duplicates from a list in Python",
            "def remove_duplicates(lst):",
            "exact_match",
        ),
    ],
    DomainType.math: [
        BenchmarkTask(DomainType.math, "What is 25 × 4 + 10?", "110", "exact_match"),
        BenchmarkTask(
            DomainType.math,
            "If a train travels 60 miles per hour, how far will it go in 2.5 hours?",
            "150",
            "exact_match",
        ),
        BenchmarkTask(
            DomainType.math, "What is the square root of 144?", "12", "exact_match"
        ),
        BenchmarkTask(
            DomainType.math, "Solve: 3x + 7 = 22. What is x?", "5", "exact_match"
        ),
        BenchmarkTask(DomainType.math, "What is 15% of 200?", "30", "exact_match"),
    ],
    DomainType.qa: [
        BenchmarkTask(
            DomainType.qa, "What is the capital of France?", "Paris", "exact_match"
        ),
        BenchmarkTask(
            DomainType.qa,
            "Who wrote Romeo and Juliet?",
            "William Shakespeare",
            "exact_match",
        ),
        BenchmarkTask(
            DomainType.qa, "What is the chemical symbol for gold?", "Au", "exact_match"
        ),
        BenchmarkTask(
            DomainType.qa, "In what year did World War II end?", "1945", "exact_match"
        ),
        BenchmarkTask(
            DomainType.qa,
            "What planet is known as the Red Planet?",
            "Mars",
            "exact_match",
        ),
    ],
    DomainType.reasoning: [
        BenchmarkTask(
            DomainType.reasoning,
            "If all humans are mortal and Socrates is human, is Socrates mortal? Explain.",
            "Socrates is mortal",
            "llm_judge",
        ),
        BenchmarkTask(
            DomainType.reasoning,
            "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?",
            "0.05",
            "llm_judge",
        ),
        BenchmarkTask(
            DomainType.reasoning,
            "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
            "5 minutes",
            "llm_judge",
        ),
        BenchmarkTask(
            DomainType.reasoning,
            "You have a 3-gallon jug and a 5-gallon jug. How can you measure exactly 4 gallons?",
            "Fill 5, pour to 3, empty 3, pour remaining 2 to 3, fill 5, pour to 3",
            "llm_judge",
        ),
        BenchmarkTask(
            DomainType.reasoning,
            "A farmer has 17 sheep. All but 9 die. How many are left?",
            "9",
            "llm_judge",
        ),
    ],
    DomainType.creative: [
        BenchmarkTask(
            DomainType.creative,
            "Write a two-line poem about the moon.",
            "poem about moon",
            "llm_judge",
        ),
        BenchmarkTask(
            DomainType.creative,
            "Invent a name for a new social media platform focused on books.",
            "creative name for book social media",
            "llm_judge",
        ),
        BenchmarkTask(
            DomainType.creative,
            "Describe a color to a blind person.",
            "description of color",
            "llm_judge",
        ),
        BenchmarkTask(
            DomainType.creative,
            "Write a short story opening about a robot learning to dream.",
            "robot dream story opening",
            "llm_judge",
        ),
        BenchmarkTask(
            DomainType.creative,
            "Create a metaphor for time passing.",
            "metaphor for time",
            "llm_judge",
        ),
    ],
    DomainType.research: [
        BenchmarkTask(
            DomainType.research,
            "Summarize the key findings of the James Webb Space Telescope's first year.",
            "JWST first year findings",
            "llm_judge",
        ),
        BenchmarkTask(
            DomainType.research,
            "What are the main approaches to quantum error correction?",
            "quantum error correction approaches",
            "llm_judge",
        ),
        BenchmarkTask(
            DomainType.research,
            "Explain the difference between RNA and DNA.",
            "RNA vs DNA differences",
            "llm_judge",
        ),
        BenchmarkTask(
            DomainType.research,
            "What is the current state of fusion energy research?",
            "fusion energy research status",
            "llm_judge",
        ),
        BenchmarkTask(
            DomainType.research,
            "How do mRNA vaccines work?",
            "mRNA vaccine mechanism",
            "llm_judge",
        ),
    ],
    DomainType.summarization: [
        BenchmarkTask(
            DomainType.summarization,
            "Summarize: The Internet is a global network connecting millions of computers. It enables communication, information sharing, and commerce. First developed in the 1960s as ARPANET, it has evolved into a ubiquitous part of modern life.",
            "global computer network evolution",
            "llm_judge",
        ),
        BenchmarkTask(
            DomainType.summarization,
            "Summarize: Photosynthesis is the process by which plants convert sunlight into energy. Chlorophyll absorbs light, which drives a reaction converting CO2 and water into glucose and oxygen.",
            "plant photosynthesis process",
            "llm_judge",
        ),
        BenchmarkTask(
            DomainType.summarization,
            "Summarize: Machine learning is a subset of AI where systems learn from data. Algorithms identify patterns and make decisions with minimal human intervention.",
            "AI machine learning definition",
            "llm_judge",
        ),
        BenchmarkTask(
            DomainType.summarization,
            "Summarize: The water cycle describes how water evaporates, condenses into clouds, and falls as precipitation, continuously circulating through the environment.",
            "water cycle description",
            "llm_judge",
        ),
        BenchmarkTask(
            DomainType.summarization,
            "Summarize: Supply and demand is an economic model where prices are determined by the balance between product availability and consumer desire.",
            "economic supply demand model",
            "llm_judge",
        ),
    ],
    DomainType.translation: [
        BenchmarkTask(
            DomainType.translation,
            'Translate to French: "Hello, how are you?"',
            "Bonjour",
            "llm_judge",
        ),
        BenchmarkTask(
            DomainType.translation,
            'Translate to Spanish: "Thank you very much."',
            "Muchas gracias",
            "llm_judge",
        ),
        BenchmarkTask(
            DomainType.translation,
            'Translate to German: "Good morning, it is a beautiful day."',
            "Guten Morgen",
            "llm_judge",
        ),
        BenchmarkTask(
            DomainType.translation,
            'Translate to Italian: "I love programming."',
            "Amo programmare",
            "llm_judge",
        ),
        BenchmarkTask(
            DomainType.translation,
            'Translate to Japanese: "The weather is nice today." (in romaji)',
            "kyou wa tenki ga ii",
            "llm_judge",
        ),
    ],
    DomainType.general: [
        BenchmarkTask(
            DomainType.general,
            "What are three benefits of regular exercise?",
            "benefits of exercise",
            "llm_judge",
        ),
        BenchmarkTask(
            DomainType.general,
            "Explain why the sky appears blue.",
            "Rayleigh scattering",
            "llm_judge",
        ),
        BenchmarkTask(
            DomainType.general,
            "What is the difference between weather and climate?",
            "weather vs climate",
            "llm_judge",
        ),
        BenchmarkTask(
            DomainType.general,
            "How does a refrigerator work?",
            "refrigeration cycle",
            "llm_judge",
        ),
        BenchmarkTask(
            DomainType.general,
            "What are the three states of matter?",
            "solid liquid gas",
            "llm_judge",
        ),
    ],
}


def get_tasks_for_domain(domain: DomainType) -> list[BenchmarkTask]:
    return BUILTIN_TASKS.get(domain, [])


def exact_match_score(output: str, expected: str) -> float:
    cleaned_output = output.strip().lower()
    cleaned_expected = expected.strip().lower()
    if cleaned_expected in cleaned_output:
        return 1.0
    return 0.0


async def llm_judge_score(
    output: str,
    expected: str,
    prompt: str,
    judge_model: str = "ollama/llama3.2:1b",
    **litellm_kwargs: Any,
) -> float:
    scoring_prompt = (
        f"Rate whether the following answer correctly addresses the task.\n\n"
        f"Task: {prompt}\n\n"
        f"Expected answer contains: {expected}\n\n"
        f"Model output:\n{output}\n\n"
        f"Respond with a single number between 0.0 and 1.0. "
        f"1.0 means perfectly correct, 0.0 means completely wrong. "
        f"Respond with ONLY the number."
    )

    try:
        response = await litellm.acompletion(
            model=judge_model,
            messages=[{"role": "user", "content": scoring_prompt}],
            max_tokens=50,
            temperature=0.0,
            **litellm_kwargs,
        )
        content: str = response.choices[0].message.content or "0.0"
        match = re.search(r"(\d+\.?\d*)", content)
        if match:
            score = float(match.group(1))
        else:
            score = 0.0
        return max(0.0, min(1.0, score))
    except Exception as e:
        logger.warning("LLM judge failed, returning 0.0: %s", e)
        return 0.0


async def score_task(
    output: str,
    task: BenchmarkTask,
    prompt: str,
    judge_model: str = "ollama/llama3.2:1b",
) -> float:
    if task.scoring == "exact_match":
        return exact_match_score(output, task.expected_answer)
    return await llm_judge_score(output, task.expected_answer, prompt, judge_model)


@dataclass
class BenchmarkResult:
    model: str
    domain: DomainType
    score: float
    latency_ms: float
    errors: int = 0
    tasks_completed: int = 0


@dataclass
class BenchmarkTaskDetail:
    model: str
    domain: DomainType
    task: BenchmarkTask
    output: str
    score: float
    latency_ms: float
    error: str | None = None


BenchmarkReport = list[BenchmarkTaskDetail]


async def run_benchmark(
    models: list[str],
    domains: list[DomainType] | None = None,
    judge_model: str = "ollama/llama3.2:1b",
    progress_callback: Any = None,
    collect_details: bool = False,
) -> tuple[RankStore, BenchmarkReport]:
    if domains is None:
        domains = ALL_DOMAINS

    entries: list[RankEntry] = []
    details: BenchmarkReport = []

    total_tasks = len(models) * sum(
        len(get_tasks_for_domain(d)) for d in domains
    )
    completed = 0

    for model in models:
        logger.debug("Benchmark: starting model %s", model)
        for domain in domains:
            tasks = get_tasks_for_domain(domain)
            if not tasks:
                logger.debug("Benchmark: no tasks for domain %s, skipping", domain)
                continue

            logger.debug(
                "Benchmark: model=%s domain=%s tasks=%d",
                model, domain.value, len(tasks),
            )

            domain_scores: list[float] = []
            total_latency = 0.0
            total_cost = 0.0
            errors = 0

            for task_idx, task in enumerate(tasks):
                logger.debug(
                    "Benchmark task %d/%d: model=%s domain=%s",
                    task_idx + 1, len(tasks), model, domain.value,
                )

                task_error: str | None = None
                task_output = ""
                task_score = 0.0
                task_latency = 0.0

                try:
                    start = time.monotonic()

                    async def _call() -> tuple[str, int | None, int | None]:
                        response = await litellm.acompletion(
                            model=model,
                            messages=[{"role": "user", "content": task.prompt}],
                            max_tokens=512,
                            temperature=0.0,
                        )
                        content = response.choices[0].message.content or ""
                        usage = getattr(response, "usage", None)
                        in_tokens = usage.prompt_tokens if usage else None
                        out_tokens = usage.completion_tokens if usage else None
                        return content, in_tokens, out_tokens

                    output, in_tokens, out_tokens = await retry_with_backoff(
                        _call, max_retries=2, base_delay_s=1.0
                    )
                    task_output = output
                    task_latency = (time.monotonic() - start) * 1000

                    task_score = await score_task(task_output, task, task.prompt, judge_model)
                    logger.debug(
                        "Benchmark result: model=%s domain=%s score=%.4f latency=%.0fms",
                        model, domain.value, task_score, task_latency,
                    )
                    domain_scores.append(task_score)
                    total_latency += task_latency
                    if in_tokens is not None and out_tokens is not None:
                        task_cost = calculate_cost(model, in_tokens, out_tokens)
                    else:
                        task_cost = 0.0
                    total_cost += task_cost
                except Exception as e:
                    logger.error("Benchmark task failed: %s", e)
                    task_error = str(e)
                    domain_scores.append(0.0)
                    errors += 1

                if collect_details:
                    details.append(
                        BenchmarkTaskDetail(
                            model=model,
                            domain=domain,
                            task=task,
                            output=task_output,
                            score=task_score,
                            latency_ms=task_latency,
                            error=task_error,
                        )
                    )

                completed += 1

                if progress_callback:
                    progress_callback(
                        completed / max(total_tasks, 1),
                        f"{model} / {domain.value} / task {task_idx + 1}/{len(tasks)}",
                    )

            if domain_scores:
                avg_score = sum(domain_scores) / len(domain_scores)
                avg_latency = total_latency / len(domain_scores)
                avg_cost = total_cost / len(domain_scores) if total_cost > 0 else 0.0
                logger.debug(
                    "Benchmark domain done: model=%s domain=%s avg_score=%.4f avg_latency=%.0fms errors=%d",
                    model, domain.value, avg_score, avg_latency, errors,
                )
                entries.append(
                    RankEntry(
                        model=model,
                        domain=domain,
                        score=round(avg_score, 4),
                        latency_ms=round(avg_latency, 2),
                        cost=round(avg_cost, 8),
                    )
                )

    return RankStore(entries=entries), details


def load_ranks(path: str | Path) -> RankStore:
    path = Path(path).expanduser()
    if not path.exists():
        logger.debug("Ranks file not found at %s, returning empty store", path)
        return RankStore()
    with open(path) as f:
        data = yaml.safe_load(f)
    if not data or "entries" not in data:
        logger.debug("Ranks file %s has no entries, returning empty store", path)
        return RankStore()
    store = RankStore.model_validate(data)
    logger.debug("Loaded %d rank entries from %s", len(store.entries), path)
    return store


def save_ranks(store: RankStore, path: str | Path) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = store.model_dump(mode="json")
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    logger.debug("Saved %d rank entries to %s", len(store.entries), path)

import json
import litellm
from pydantic import BaseModel
from typing import List
from .config import PolyMindConfig, ModelConfig

litellm.set_verbose = False

ANALYZER_SYSTEM_PROMPT = """You are a task decomposition engine. Analyze the user prompt and break it into typed subtasks.

Available domain tags:
  code           → writing, debugging, explaining code
  math           → equations, calculations, proofs
  creative       → writing, storytelling, brainstorming
  research       → facts, comparisons, explanations of concepts
  summarization  → condensing long content
  translation    → language translation
  qa             → simple factual question-answering
  general        → anything that doesn't fit another domain

Rules:
1. Return ONLY valid JSON — no markdown fences, no explanation
2. If the prompt is simple and fits one domain, return exactly ONE subtask
3. Only split into multiple subtasks if the prompt genuinely spans multiple domains
4. depends_on contains IDs of tasks that must finish before this one starts
5. Keep each description self-contained — include enough context for the model to answer it alone

Response format (strict):
{
  "subtasks": [
    {
      "id": "task_1",
      "domain": "code",
      "description": "Write a Python function that...",
      "depends_on": []
    }
  ]
}"""


class Subtask(BaseModel):
    id: str
    domain: str
    description: str
    depends_on: List[str] = []


class AnalyzerPlan(BaseModel):
    subtasks: List[Subtask]


def _build_litellm_args(cfg: ModelConfig) -> tuple[str, dict]:
    """Return (model_string, kwargs) for litellm.acompletion."""
    provider = cfg.provider.lower()
    kwargs = {}

    if provider == "ollama":
        model_str = f"ollama/{cfg.model}"
        kwargs["api_base"] = cfg.base_url or "http://localhost:11434"

    elif provider == "openrouter":
        model_str = cfg.model if cfg.model.startswith("openrouter/") else f"openrouter/{cfg.model}"
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key

    elif provider == "lmstudio":
        model_str = f"openai/{cfg.model}"
        kwargs["api_base"] = cfg.base_url or "http://localhost:1234/v1"
        kwargs["api_key"] = "lm-studio"

    elif provider == "openai":
        model_str = cfg.model
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key

    elif provider == "anthropic":
        model_str = cfg.model
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key

    else:
        # Generic OpenAI-compatible endpoint
        model_str = cfg.model
        if cfg.base_url:
            kwargs["api_base"] = cfg.base_url
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key

    return model_str, kwargs


async def analyze_prompt(prompt: str, config: PolyMindConfig) -> AnalyzerPlan:
    """Call the analyzer LLM to decompose the prompt into a subtask plan."""
    model_str, kwargs = _build_litellm_args(config.analyzer)

    try:
        response = await litellm.acompletion(
            model=model_str,
            messages=[
                {"role": "system", "content": ANALYZER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Decompose this prompt:\n\n{prompt}"},
            ],
            temperature=0.1,
            max_tokens=1024,
            **kwargs,
        )

        content = response.choices[0].message.content.strip()

        # Strip accidental markdown code fences
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                stripped = part.strip()
                if stripped.startswith("json"):
                    stripped = stripped[4:].strip()
                if stripped.startswith("{"):
                    content = stripped
                    break

        data = json.loads(content)
        plan = AnalyzerPlan(**data)

        # Validate all domain tags
        valid_domains = {"code", "math", "creative", "research", "summarization",
                         "translation", "qa", "general", "data-analysis"}
        for task in plan.subtasks:
            if task.domain not in valid_domains and task.domain not in config.specialists:
                task.domain = "general"

        return plan

    except Exception as e:
        print(f"Analyzer error: {e} — falling back to single general task")
        return AnalyzerPlan(
            subtasks=[Subtask(id="task_1", domain="general", description=prompt, depends_on=[])]
        )

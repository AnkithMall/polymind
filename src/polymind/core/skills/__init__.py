"""Skills system — plug-and-play tool framework for local LLMs.

Enables any LLM (with or without native tool calling) to interact with
the filesystem, shell, and external services through a sandboxed,
permission-gated skill system.

Architecture:
    Skill (base) → SkillManifest (metadata) → SkillRegistry (discovery)
    → AgentLoop (ReAct orchestration) → SkillSandbox (security)

Usage:
    from polymind.core.skills import SkillRegistry, AgentLoop

    registry = SkillRegistry()
    registry.discover()  # loads built-in + user skills

    agent = AgentLoop(model, registry, sandbox_config)
    result = agent.run("Read the file config.yaml and summarize it")
"""

from polymind.core.skills.agent import AgentLoop, AgentResult, AgentStep
from polymind.core.skills.base import Skill, SkillContext, SkillResult
from polymind.core.skills.registry import SkillRegistry
from polymind.core.skills.sandbox import SandboxConfig, SandboxPolicy

__all__ = [
    "AgentLoop",
    "AgentResult",
    "AgentStep",
    "SandboxConfig",
    "SandboxPolicy",
    "Skill",
    "SkillContext",
    "SkillRegistry",
    "SkillResult",
]

"""Base Skill class — all skills inherit from this."""

from __future__ import annotations

from abc import ABC, abstractmethod

from polymind.core.skills.types import SkillContext, SkillManifest, SkillResult


class Skill(ABC):
    """Base class for all skills.

    To create a custom skill:
        1. Subclass Skill
        2. Implement manifest() to return metadata
        3. Implement execute() with your logic
        4. Place in ~/.polymind/skills/ or register via CLI

    Example:
        class MySkill(Skill):
            def manifest(self) -> SkillManifest:
                return SkillManifest(
                    name="my_skill",
                    description="Does something useful",
                    params=[SkillParam(name="query", type="string", description="Input")],
                )

            def execute(self, ctx: SkillContext, **kwargs) -> SkillResult:
                query = kwargs.get("query", "")
                return SkillResult(success=True, output=f"Result for {query}")
    """

    @abstractmethod
    def manifest(self) -> SkillManifest:
        """Return the skill's metadata and parameter definitions."""
        ...

    @abstractmethod
    def execute(self, ctx: SkillContext, **kwargs) -> SkillResult:
        """Execute the skill with the given arguments.

        Args:
            ctx: Runtime context (working dir, env, timeout, etc.)
            **kwargs: Skill-specific parameters matching the manifest.

        Returns:
            SkillResult with success/error status and output.
        """
        ...

    def validate_params(self, **kwargs) -> dict:
        """Validate parameters against the manifest. Returns validated dict."""
        manifest = self.manifest()
        validated = {}
        for param in manifest.params:
            value = kwargs.get(param.name)
            validated[param.name] = param.validate(value)
        return validated

    def needs_approval(self, **kwargs) -> bool:
        """Check if this execution requires user approval.

        Override for custom approval logic (e.g., only approve destructive ops).
        """
        return self.manifest().requires_approval

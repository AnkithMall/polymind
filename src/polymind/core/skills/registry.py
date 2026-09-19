"""Skill registry — discovers, loads, and manages skills.

Skills are discovered from:
1. Built-in skills (polymind.core.skills.builtin.*)
2. User skills (~/.polymind/skills/<name>.py or <name>/)
3. Plugin skills (polymind plugin install <name>)

Each skill has a manifest (YAML or code-based) that defines its
name, description, parameters, and permissions.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import yaml

from polymind.core.paths import artifact_dir
from polymind.core.skills.base import Skill
from polymind.core.skills.types import Permission, SkillManifest, SkillParam


class SkillRegistry:
    """Central registry for all available skills."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self._manifests: dict[str, SkillManifest] = {}
        self._discovered = False

    @property
    def skills(self) -> dict[str, Skill]:
        return dict(self._skills)

    @property
    def manifests(self) -> dict[str, SkillManifest]:
        return dict(self._manifests)

    def discover(self) -> None:
        """Discover all available skills (built-in + user)."""
        if self._discovered:
            return

        self._discover_builtin()
        self._discover_user_skills()
        self._discovered = True

    def _discover_builtin(self) -> None:
        """Load all built-in skills."""
        from polymind.core.skills.builtin import BUILTIN_SKILLS

        for skill_cls in BUILTIN_SKILLS:
            skill = skill_cls()
            manifest = skill.manifest()
            self._skills[manifest.name] = skill
            self._manifests[manifest.name] = manifest

    def _discover_user_skills(self) -> None:
        """Discover and load user-defined skills from ~/.polymind/skills/."""
        skills_dir = artifact_dir() / "skills"
        if not skills_dir.exists():
            return

        for item in skills_dir.iterdir():
            if item.is_file() and item.suffix == ".py" and not item.name.startswith("_"):
                self._load_skill_module(item)
            elif item.is_dir() and (item / "__init__.py").exists():
                self._load_skill_package(item)
            elif item.is_file() and item.suffix == ".yaml":
                self._load_skill_yaml(item)

    def _load_skill_module(self, path: Path) -> None:
        """Load a single-file skill from a .py file."""
        try:
            spec = importlib.util.spec_from_file_location(
                f"polymind_user_skill_{path.stem}", str(path)
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "SKILL_CLASS"):
                    skill = module.SKILL_CLASS()
                elif hasattr(module, "Skill"):
                    # Find Skill subclasses in the module
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type) and issubclass(attr, Skill) and attr is not Skill:
                            skill = attr()
                            break
                    else:
                        return
                else:
                    return

                manifest = skill.manifest()
                self._skills[manifest.name] = skill
                self._manifests[manifest.name] = manifest
        except Exception:
            pass  # Skip skills that fail to load

    def _load_skill_package(self, path: Path) -> None:
        """Load a skill package (directory with __init__.py)."""
        self._load_skill_module(path / "__init__.py")

    def _load_skill_yaml(self, path: Path) -> None:
        """Load a skill defined by a YAML manifest + Python handler."""
        try:
            with path.open() as f:
                data = yaml.safe_load(f)
            if not data or "name" not in data:
                return

            # The YAML references a handler module
            handler_path = data.pop("handler", None)
            if handler_path and Path(handler_path).exists():
                spec = importlib.util.spec_from_file_location(
                    f"polymind_yaml_skill_{data['name']}", handler_path
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, "execute"):
                        # Create a wrapper skill
                        manifest = _manifest_from_yaml(data)
                        skill = _YamlSkillWrapper(manifest, module.execute)
                        self._skills[manifest.name] = skill
                        self._manifests[manifest.name] = manifest
        except Exception:
            pass

    def get(self, name: str) -> Skill | None:
        """Get a skill by name."""
        self.discover()
        return self._skills.get(name)

    def get_manifest(self, name: str) -> SkillManifest | None:
        """Get a skill's manifest by name."""
        self.discover()
        return self._manifests.get(name)

    def list_skills(self) -> list[SkillManifest]:
        """List all available skill manifests."""
        self.discover()
        return list(self._manifests.values())

    def list_names(self) -> list[str]:
        """List all available skill names."""
        self.discover()
        return list(self._skills.keys())

    def filter_by_permission(self, permission: Permission) -> list[SkillManifest]:
        """Filter skills that require a specific permission."""
        return [m for m in self.list_skills() if permission in m.permissions]

    def register(self, skill: Skill) -> None:
        """Manually register a skill instance."""
        manifest = skill.manifest()
        self._skills[manifest.name] = skill
        self._manifests[manifest.name] = manifest

    def unregister(self, name: str) -> bool:
        """Remove a skill from the registry."""
        if name in self._skills:
            del self._skills[name]
            del self._manifests[name]
            return True
        return False

    def to_system_prompt_tools(self) -> str:
        """Format all skills as tool descriptions for the LLM system prompt."""
        manifests = self.list_skills()
        if not manifests:
            return "No tools available."

        lines = ["You have access to the following tools:", ""]
        for m in manifests:
            lines.append(m.to_tool_description())
            lines.append("")
        return "\n".join(lines)

    def to_openai_tools(self) -> list[dict]:
        """Convert all skills to OpenAI function-calling format."""
        return [m.to_openai_tool() for m in self.list_skills()]


class _YamlSkillWrapper(Skill):
    """Wrapper for YAML-defined skills with a Python handler function."""

    def __init__(self, manifest: SkillManifest, handler):
        self._manifest = manifest
        self._handler = handler

    def manifest(self) -> SkillManifest:
        return self._manifest

    def execute(self, ctx, **kwargs):
        return self._handler(ctx, **kwargs)


def _manifest_from_yaml(data: dict) -> SkillManifest:
    """Convert a YAML dict to a SkillManifest."""
    params = []
    for p in data.get("params", []):
        params.append(
            SkillParam(
                name=p["name"],
                type=p.get("type", "string"),
                description=p.get("description", ""),
                required=p.get("required", True),
                default=p.get("default"),
                enum=p.get("enum", []),
            )
        )

    permissions = []
    for perm_str in data.get("permissions", []):
        try:
            permissions.append(Permission(perm_str))
        except ValueError:
            pass

    return SkillManifest(
        name=data["name"],
        description=data.get("description", ""),
        version=data.get("version", "1.0.0"),
        author=data.get("author", "user"),
        permissions=permissions,
        params=params,
        examples=data.get("examples", []),
        tags=data.get("tags", []),
        timeout_seconds=data.get("timeout_seconds", 30),
        requires_approval=data.get("requires_approval", False),
    )

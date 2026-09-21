"""Skill registry — discovers, loads, and manages skills.

Skills are discovered from:
1. Built-in skills (polymind.core.skills.builtin.*)
2. User skills (~/.polymind/skills/<name>.py or <name>/)
3. MCP servers (via MCP client)
4. Plugin skills (polymind plugin install <name>)

Each skill has a manifest (YAML or code-based) that defines its
name, description, parameters, and permissions.

Config is persisted to .polymind/skills.yaml for enable/disable state.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import yaml

from polymind.core.paths import artifact_dir
from polymind.core.skills.base import Skill
from polymind.core.skills.types import (
    Permission,
    SkillManifest,
    SkillParam,
    SkillSource,
    SkillStatus,
)


def _skills_config_path() -> Path:
    """Path to skills configuration (enable/disable state)."""
    return artifact_dir() / "skills.yaml"


def _load_skills_config() -> dict:
    """Load skills config (which skills are enabled/disabled)."""
    path = _skills_config_path()
    if path.exists():
        try:
            with path.open() as f:
                data = yaml.safe_load(f) or {}
            return data.get("skills", {})
        except Exception:
            return {}
    return {}


def _save_skills_config(skills_state: dict[str, dict]) -> None:
    """Save skills config (enable/disable state)."""
    path = _skills_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.dump({"skills": skills_state}, f, default_flow_style=False)


class SkillRegistry:
    """Central registry for all available skills."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self._manifests: dict[str, SkillManifest] = {}
        self._sources: dict[str, SkillSource] = {}
        self._disabled: set[str] = set()
        self._discovered = False
        self._mcp_servers: dict[str, dict] = {}  # name -> server config

        # Load persisted enable/disable state
        config = _load_skills_config()
        for name, info in config.items():
            if isinstance(info, dict) and info.get("status") == "disabled":
                self._disabled.add(name)

    @property
    def skills(self) -> dict[str, Skill]:
        return dict(self._skills)

    @property
    def manifests(self) -> dict[str, SkillManifest]:
        return dict(self._manifests)

    def discover(self) -> None:
        """Discover all available skills (built-in + user + MCP)."""
        if self._discovered:
            return

        self._discover_builtin()
        self._discover_user_skills()
        self._discover_mcp_skills()
        self._discovered = True

    def _discover_builtin(self) -> None:
        """Load all built-in skills."""
        from polymind.core.skills.builtin import BUILTIN_SKILLS

        for skill_cls in BUILTIN_SKILLS:
            skill = skill_cls()
            manifest = skill.manifest()
            manifest.source = SkillSource.BUILTIN
            manifest.status = (
                SkillStatus.DISABLED if manifest.name in self._disabled else SkillStatus.ENABLED
            )
            self._skills[manifest.name] = skill
            self._manifests[manifest.name] = manifest
            self._sources[manifest.name] = SkillSource.BUILTIN

    def _discover_user_skills(self) -> None:
        """Discover and load user-defined skills from .polymind/skills/."""
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
                manifest.source = SkillSource.USER
                manifest.source_detail = str(path)
                manifest.status = (
                    SkillStatus.DISABLED if manifest.name in self._disabled else SkillStatus.ENABLED
                )
                self._skills[manifest.name] = skill
                self._manifests[manifest.name] = manifest
                self._sources[manifest.name] = SkillSource.USER
        except Exception:
            pass

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

            handler_path = data.pop("handler", None)
            if handler_path and Path(handler_path).exists():
                spec = importlib.util.spec_from_file_location(
                    f"polymind_yaml_skill_{data['name']}", handler_path
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, "execute"):
                        manifest = _manifest_from_yaml(data)
                        manifest.source = SkillSource.USER
                        manifest.source_detail = str(path)
                        manifest.status = (
                            SkillStatus.DISABLED
                            if manifest.name in self._disabled
                            else SkillStatus.ENABLED
                        )
                        skill = _YamlSkillWrapper(manifest, module.execute)
                        self._skills[manifest.name] = skill
                        self._manifests[manifest.name] = manifest
                        self._sources[manifest.name] = SkillSource.USER
        except Exception:
            pass

    def _discover_mcp_skills(self) -> None:
        """Discover tools from configured MCP servers."""
        mcp_config = _load_mcp_config()
        for server_name, server_cfg in mcp_config.items():
            if not server_cfg.get("enabled", True):
                continue
            try:
                tools = _discover_mcp_tools(server_cfg)
                for tool_info in tools:
                    skill = _McpToolSkill(tool_info, server_name)
                    manifest = skill.manifest()
                    manifest.source = SkillSource.MCP
                    manifest.source_detail = f"mcp:{server_name}"
                    manifest.status = (
                        SkillStatus.DISABLED
                        if manifest.name in self._disabled
                        else SkillStatus.ENABLED
                    )
                    self._skills[manifest.name] = skill
                    self._manifests[manifest.name] = manifest
                    self._sources[manifest.name] = SkillSource.MCP
                    self._mcp_servers.setdefault(server_name, server_cfg)
            except Exception:
                pass  # Skip MCP servers that fail to connect

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

    def list_enabled(self) -> list[SkillManifest]:
        """List only enabled skill manifests."""
        return [m for m in self.list_skills() if m.status == SkillStatus.ENABLED]

    def list_disabled(self) -> list[SkillManifest]:
        """List only disabled skill manifests."""
        return [m for m in self.list_skills() if m.status == SkillStatus.DISABLED]

    def list_by_source(self, source: SkillSource) -> list[SkillManifest]:
        """List skills from a specific source."""
        return [m for m in self.list_skills() if m.source == source]

    def list_names(self) -> list[str]:
        """List all available skill names."""
        self.discover()
        return list(self._skills.keys())

    def list_enabled_names(self) -> list[str]:
        """List names of enabled skills."""
        return [m.name for m in self.list_enabled()]

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
            self._sources.pop(name, None)
            return True
        return False

    def enable(self, name: str) -> bool:
        """Enable a skill. Returns True if the skill exists."""
        self.discover()
        if name not in self._manifests:
            return False
        self._disabled.discard(name)
        self._manifests[name].status = SkillStatus.ENABLED
        self._persist_state()
        return True

    def disable(self, name: str) -> bool:
        """Disable a skill. Returns True if the skill exists."""
        self.discover()
        if name not in self._manifests:
            return False
        self._disabled.add(name)
        self._manifests[name].status = SkillStatus.DISABLED
        self._persist_state()
        return True

    def _persist_state(self) -> None:
        """Save enable/disable state to config file."""
        state = {}
        for name, manifest in self._manifests.items():
            state[name] = {
                "status": manifest.status.value,
                "source": manifest.source.value,
            }
        _save_skills_config(state)

    def to_system_prompt_tools(self) -> str:
        """Format enabled skills as tool descriptions for the LLM system prompt."""
        manifests = self.list_enabled()
        if not manifests:
            return "No tools available."

        lines = ["You have access to the following tools:", ""]
        for m in manifests:
            lines.append(m.to_tool_description())
            lines.append("")
        return "\n".join(lines)

    def to_openai_tools(self) -> list[dict]:
        """Convert enabled skills to OpenAI function-calling format."""
        return [m.to_openai_tool() for m in self.list_enabled()]


class _YamlSkillWrapper(Skill):
    """Wrapper for YAML-defined skills with a Python handler function."""

    def __init__(self, manifest: SkillManifest, handler):
        self._manifest = manifest
        self._handler = handler

    def manifest(self) -> SkillManifest:
        return self._manifest

    def execute(self, ctx, **kwargs):
        return self._handler(ctx, **kwargs)


class _McpToolSkill(Skill):
    """Wrapper that exposes an MCP server tool as a polymind Skill."""

    def __init__(self, tool_info: dict, server_name: str):
        self._tool_info = tool_info
        self._server_name = server_name
        self._manifest = _manifest_from_mcp_tool(tool_info, server_name)

    def manifest(self) -> SkillManifest:
        return self._manifest

    def execute(self, ctx, **kwargs):
        from polymind.core.skills.mcp_client import call_mcp_tool

        server_cfg = _load_mcp_config().get(self._server_name, {})
        return call_mcp_tool(server_cfg, self._tool_info["name"], kwargs)


def _manifest_from_mcp_tool(tool_info: dict, server_name: str) -> SkillManifest:
    """Convert an MCP tool definition to a SkillManifest."""
    input_schema = tool_info.get("inputSchema", {})
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    params = []
    for pname, pdef in properties.items():
        params.append(
            SkillParam(
                name=pname,
                type=pdef.get("type", "string"),
                description=pdef.get("description", ""),
                required=pname in required,
                default=pdef.get("default"),
                enum=pdef.get("enum", []),
            )
        )

    return SkillManifest(
        name=tool_info["name"],
        description=tool_info.get("description", ""),
        version="1.0.0",
        author=server_name,
        params=params,
        tags=["mcp", server_name],
        timeout_seconds=30,
        source=SkillSource.MCP,
        source_detail=f"mcp:{server_name}",
    )


# ── MCP config persistence ─────────────────────────────


def _mcp_config_path() -> Path:
    """Path to MCP servers configuration."""
    return artifact_dir() / "mcp_servers.yaml"


def _load_mcp_config() -> dict:
    """Load MCP server configurations."""
    path = _mcp_config_path()
    if path.exists():
        try:
            with path.open() as f:
                data = yaml.safe_load(f) or {}
            return data.get("servers", {})
        except Exception:
            return {}
    return {}


def _save_mcp_config(servers: dict) -> None:
    """Save MCP server configurations."""
    path = _mcp_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.dump({"servers": servers}, f, default_flow_style=False)


def _discover_mcp_tools(server_cfg: dict) -> list[dict]:
    """Connect to an MCP server and discover its tools via stdio."""
    from polymind.core.skills.mcp_client import list_mcp_tools

    return list_mcp_tools(server_cfg)


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

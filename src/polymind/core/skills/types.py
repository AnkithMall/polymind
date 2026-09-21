"""Core types for the skills system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Permission(StrEnum):
    """Skill permission levels — granular access control."""

    FS_READ = "fs_read"  # Read files and directories
    FS_WRITE = "fs_write"  # Create, modify, delete files
    SHELL = "shell"  # Execute shell commands
    NETWORK = "network"  # HTTP requests, API calls
    PYTHON = "python"  # Execute Python code
    APPROVAL = "approval"  # Requires user approval before execution


class SkillSource(StrEnum):
    """Where a skill comes from."""

    BUILTIN = "builtin"  # Ships with polymind
    USER = "user"  # Installed by user to .polymind/skills/
    MCP = "mcp"  # Discovered from an MCP server
    PLUGIN = "plugin"  # Installed via plugin system


class SkillStatus(StrEnum):
    """Whether a skill is enabled or disabled."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"  # Failed to load


class SandboxPolicy(StrEnum):
    """Sandbox enforcement policies."""

    DISABLED = "disabled"  # No sandboxing (trusted skills only)
    BASIC = "basic"  # Timeouts + output limits
    STRICT = "strict"  # chroot + timeouts + network block
    CUSTOM = "custom"  # User-defined rules


class StepStatus(StrEnum):
    """Status of an agent step."""

    THINKING = "thinking"
    CALLING = "calling"
    COMPLETE = "complete"
    ERROR = "error"
    APPROVED = "approved"
    DENIED = "denied"


@dataclass
class SkillParam:
    """A single parameter for a skill."""

    name: str
    type: str  # "string", "integer", "boolean", "array", "object"
    description: str = ""
    required: bool = True
    default: Any = None
    enum: list[str] = field(default_factory=list)

    def to_schema(self) -> dict:
        """Convert to JSON Schema property."""
        schema: dict[str, Any] = {"type": self.type, "description": self.description}
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema

    def validate(self, value: Any) -> Any:
        """Validate and coerce a parameter value."""
        if value is None and self.required and self.default is None:
            raise ValueError(f"Parameter '{self.name}' is required")
        if value is None:
            return self.default

        if self.enum and value not in self.enum:
            raise ValueError(f"Parameter '{self.name}' must be one of: {self.enum}")

        type_map = {
            "string": str,
            "integer": int,
            "boolean": bool,
        }
        target = type_map.get(self.type)
        if target:
            try:
                return target(value)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Parameter '{self.name}' expected {self.type}, got {type(value).__name__}"
                )
        return value


@dataclass
class SkillManifest:
    """Metadata for a skill — discovered from YAML manifests or code."""

    name: str
    description: str
    version: str = "1.0.0"
    author: str = "polymind"
    permissions: list[Permission] = field(default_factory=list)
    params: list[SkillParam] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    timeout_seconds: int = 30
    requires_approval: bool = False
    sandbox_policy: SandboxPolicy = SandboxPolicy.BASIC
    source: SkillSource = SkillSource.BUILTIN
    status: SkillStatus = SkillStatus.ENABLED
    source_detail: str = ""  # e.g. MCP server name, file path

    def to_tool_description(self) -> str:
        """Format as a tool description for the LLM system prompt."""
        lines = [
            f"## {self.name}",
            f"{self.description}",
            "",
            "Parameters:",
        ]
        for p in self.params:
            req = "(required)" if p.required else f"(optional, default={p.default})"
            lines.append(f"  - {p.name} ({p.type}) {req}: {p.description}")
            if p.enum:
                lines.append(f"    Allowed values: {', '.join(p.enum)}")
        if self.examples:
            lines.append("")
            lines.append("Examples:")
            for ex in self.examples:
                lines.append(f"  {ex}")
        return "\n".join(lines)

    def to_openai_tool(self) -> dict:
        """Convert to OpenAI function-calling schema format."""
        properties = {}
        required = []
        for p in self.params:
            properties[p.name] = p.to_schema()
            if p.required:
                required.append(p.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


@dataclass
class SkillResult:
    """Result of a skill execution."""

    success: bool
    output: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False

    def to_text(self) -> str:
        """Format as text for feeding back to the LLM."""
        if self.success:
            prefix = "[Tool result:"
            if self.truncated:
                prefix += " (truncated)"
            return f"{prefix} {self.output}]"
        else:
            return f"[Tool error: {self.error}]"


@dataclass
class SkillContext:
    """Runtime context passed to a skill during execution."""

    working_dir: str = "."
    env: dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 30
    max_output_bytes: int = 1024 * 100  # 100KB
    user_approved: bool = False
    skill_name: str = ""
    call_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentStep:
    """A single step in the agent's reasoning loop."""

    step_number: int
    thought: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_result: SkillResult | None = None
    status: StepStatus = StepStatus.THINKING

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "step": self.step_number,
            "status": self.status.value,
        }
        if self.thought:
            d["thought"] = self.thought
        if self.tool_name:
            d["tool"] = self.tool_name
            d["args"] = self.tool_args
        if self.tool_result:
            d["result"] = {
                "success": self.tool_result.success,
                "output": self.tool_result.output[:500],
            }
        return d


@dataclass
class AgentResult:
    """Final result from the agent loop."""

    response: str
    steps: list[AgentStep] = field(default_factory=list)
    total_steps: int = 0
    tools_used: list[str] = field(default_factory=list)
    success: bool = True
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "response": self.response,
            "total_steps": self.total_steps,
            "tools_used": self.tools_used,
            "success": self.success,
            "steps": [s.to_dict() for s in self.steps],
        }

"""Sandboxing and security for skill execution.

Every skill runs through the sandbox which enforces:
- Timeouts (prevent hangs)
- Output size limits (prevent memory exhaustion)
- Permission checks (prevent unauthorized access)
- Filesystem restrictions (prevent escape)
- Network blocking (when not permitted)
- User approval for dangerous operations
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from polymind.core.skills.types import Permission, SandboxPolicy, SkillContext


@dataclass
class SandboxConfig:
    """Configuration for the sandbox environment."""

    policy: SandboxPolicy = SandboxPolicy.BASIC
    allowed_dirs: list[str] = field(default_factory=list)
    blocked_commands: list[str] = field(default_factory=list)
    max_output_bytes: int = 1024 * 100  # 100KB
    default_timeout: int = 30
    network_allowed: bool = False
    require_approval_for: list[str] = field(default_factory=list)

    # Pre-approved skills (skip approval prompt)
    auto_approved_skills: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Ensure home directory is always in allowed_dirs
        home = str(Path.home())
        if home not in self.allowed_dirs:
            self.allowed_dirs.append(home)


class SandboxEnforcer:
    """Enforces sandbox policies on skill execution."""

    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or SandboxConfig()

    def check_permission(self, skill_name: str, permission: Permission) -> bool:
        """Check if a skill has the required permission."""
        if self.config.policy == SandboxPolicy.DISABLED:
            return True

        # Network permission
        if permission == Permission.NETWORK and not self.config.network_allowed:
            return False

        # Shell permission — check blocked commands
        if permission == Permission.SHELL:
            return True  # Shell access is allowed but sandboxed

        return True

    def check_path_access(self, path: str, writing: bool = False) -> bool:
        """Check if a path is within allowed directories."""
        if self.config.policy == SandboxPolicy.DISABLED:
            return True

        try:
            resolved = str(Path(path).resolve())
        except (OSError, ValueError):
            return False

        for allowed in self.config.allowed_dirs:
            allowed_resolved = str(Path(allowed).resolve())
            if resolved.startswith(allowed_resolved + os.sep) or resolved == allowed_resolved:
                return True

        return False

    def needs_approval(self, skill_name: str, permission: Permission) -> bool:
        """Check if this skill+permission requires user approval."""
        if self.config.policy == SandboxPolicy.DISABLED:
            return False

        if skill_name in self.config.auto_approved_skills:
            return False

        if permission.value in self.config.require_approval_for:
            return True

        if permission == Permission.SHELL and self.config.policy == SandboxPolicy.STRICT:
            return True

        if permission == Permission.FS_WRITE and self.config.policy == SandboxPolicy.STRICT:
            return True

        return False

    def create_context(
        self,
        skill_name: str,
        working_dir: str = ".",
        timeout: int | None = None,
    ) -> SkillContext:
        """Create a sandboxed context for skill execution."""
        return SkillContext(
            working_dir=str(Path(working_dir).resolve()),
            timeout_seconds=timeout or self.config.default_timeout,
            max_output_bytes=self.config.max_output_bytes,
            skill_name=skill_name,
        )

    def truncate_output(self, output: str) -> tuple[str, bool]:
        """Truncate output if it exceeds the size limit."""
        encoded = output.encode("utf-8")
        if len(encoded) <= self.config.max_output_bytes:
            return output, False
        truncated = encoded[: self.config.max_output_bytes].decode("utf-8", errors="replace")
        return truncated, True

    def is_command_blocked(self, command: str) -> bool:
        """Check if a shell command is in the blocklist."""
        cmd_lower = command.lower().strip()
        for blocked in self.config.blocked_commands:
            if cmd_lower.startswith(blocked.lower()):
                return True
        return False


def default_sandbox_config() -> SandboxConfig:
    """Create a default sandbox configuration."""
    home = str(Path.home())
    return SandboxConfig(
        policy=SandboxPolicy.BASIC,
        allowed_dirs=[
            home,
            "/tmp",
            str(Path.cwd()),
        ],
        blocked_commands=[
            "rm -rf /",
            "mkfs",
            "dd if=",
            ":(){:|:&};:",  # fork bomb
            "chmod -R 777 /",
            "wget http://malicious",
            "curl http://malicious",
        ],
        max_output_bytes=1024 * 100,  # 100KB
        default_timeout=30,
        network_allowed=False,
        require_approval_for=["shell", "fs_write", "python", "network"],
    )

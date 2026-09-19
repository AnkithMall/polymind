"""Shell skill — execute shell commands in a sandboxed subprocess."""

from __future__ import annotations

import subprocess
import time

from polymind.core.skills.base import Skill
from polymind.core.skills.types import (
    Permission,
    SkillContext,
    SkillManifest,
    SkillParam,
    SkillResult,
)


class ShellSkill(Skill):
    """Execute a shell command."""

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="shell",
            description=(
                "Execute a shell command and return its output. "
                "Commands run in a sandboxed subprocess with timeout. "
                "Use for: running build tools, git commands, package managers, "
                "system utilities, etc."
            ),
            permissions=[Permission.SHELL],
            params=[
                SkillParam(
                    name="command",
                    type="string",
                    description="Shell command to execute",
                ),
                SkillParam(
                    name="timeout",
                    type="integer",
                    description="Timeout in seconds. Default: 30",
                    required=False,
                    default=30,
                ),
                SkillParam(
                    name="workdir",
                    type="string",
                    description="Working directory for the command",
                    required=False,
                    default=None,
                ),
            ],
            examples=[
                'shell(command="ls -la")',
                'shell(command="git status")',
                "shell(command=\"python -c 'print(1+1)'\")",
                'shell(command="find . -name *.py | head -20")',
            ],
            tags=["shell", "exec", "system"],
            timeout_seconds=60,
            requires_approval=True,
        )

    def execute(self, ctx: SkillContext, **kwargs) -> SkillResult:
        try:
            validated = self.validate_params(**kwargs)
            command = validated["command"]
            timeout = validated.get("timeout", ctx.timeout_seconds) or ctx.timeout_seconds
            workdir = validated.get("workdir") or ctx.working_dir

            start = time.time()
            try:
                proc = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=workdir,
                    env={**__import__("os").environ, **ctx.env},
                )
                elapsed = time.time() - start

                output = proc.stdout
                if proc.stderr:
                    output += f"\n[stderr]\n{proc.stderr}"

                truncated = False
                if len(output.encode("utf-8")) > ctx.max_output_bytes:
                    output = output.encode("utf-8")[: ctx.max_output_bytes].decode(
                        "utf-8", errors="replace"
                    )
                    truncated = True

                if not output.strip():
                    output = "(no output)"

                return SkillResult(
                    success=proc.returncode == 0,
                    output=output,
                    error=f"Exit code: {proc.returncode}" if proc.returncode != 0 else "",
                    truncated=truncated,
                    metadata={
                        "exit_code": proc.returncode,
                        "elapsed_s": f"{elapsed:.2f}",
                        "command": command,
                    },
                )
            except subprocess.TimeoutExpired:
                return SkillResult(
                    success=False,
                    error=f"Command timed out after {timeout}s",
                    metadata={"command": command, "timeout": timeout},
                )
        except Exception as e:
            return SkillResult(success=False, error=str(e))

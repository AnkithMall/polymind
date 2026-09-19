"""Python execution skill — run Python code in a sandboxed subprocess."""

from __future__ import annotations

import subprocess
import tempfile
import time

from polymind.core.skills.base import Skill
from polymind.core.skills.types import (
    Permission,
    SkillContext,
    SkillManifest,
    SkillParam,
    SkillResult,
)


class PythonExecSkill(Skill):
    """Execute Python code."""

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="python_exec",
            description=(
                "Execute Python code in an isolated subprocess. "
                "Use for: calculations, data processing, testing ideas, "
                "generating content. The code runs with access to standard library "
                "only (no network by default)."
            ),
            permissions=[Permission.PYTHON],
            params=[
                SkillParam(
                    name="code",
                    type="string",
                    description="Python code to execute",
                ),
                SkillParam(
                    name="timeout",
                    type="integer",
                    description="Timeout in seconds. Default: 30",
                    required=False,
                    default=30,
                ),
            ],
            examples=[
                'python_exec(code="print(sum(range(100)))")',
                "python_exec(code=\"import json; print(json.dumps({'a': 1}))\")",
                'python_exec(code="result = [i**2 for i in range(10)]\\nprint(result)")',
            ],
            tags=["python", "exec", "code"],
            timeout_seconds=30,
            requires_approval=True,
        )

    def execute(self, ctx: SkillContext, **kwargs) -> SkillResult:
        try:
            validated = self.validate_params(**kwargs)
            code = validated["code"]
            timeout = validated.get("timeout", ctx.timeout_seconds) or ctx.timeout_seconds

            # Write code to a temp file and execute it
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir="/tmp") as f:
                f.write(code)
                temp_path = f.name

            start = time.time()
            try:
                proc = subprocess.run(
                    ["python3", temp_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=ctx.working_dir,
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
                    },
                )
            except subprocess.TimeoutExpired:
                return SkillResult(
                    success=False,
                    error=f"Code execution timed out after {timeout}s",
                )
            finally:
                import os

                os.unlink(temp_path)
        except Exception as e:
            return SkillResult(success=False, error=str(e))

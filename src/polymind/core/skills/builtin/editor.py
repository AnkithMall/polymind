"""Editor skills — view and edit files with line-level precision."""

from __future__ import annotations

from pathlib import Path

from polymind.core.skills.base import Skill
from polymind.core.skills.types import (
    Permission,
    SkillContext,
    SkillManifest,
    SkillParam,
    SkillResult,
)


class ViewFileSkill(Skill):
    """View file contents with line numbers (like cat -n)."""

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="view_file",
            description="View file contents with line numbers. Supports viewing specific line ranges.",
            permissions=[Permission.FS_READ],
            params=[
                SkillParam(
                    name="path",
                    type="string",
                    description="Path to the file to view",
                ),
                SkillParam(
                    name="start_line",
                    type="integer",
                    description="Start line number (1-based). Default: 1",
                    required=False,
                    default=1,
                ),
                SkillParam(
                    name="end_line",
                    type="integer",
                    description="End line number (1-based, inclusive). Default: all",
                    required=False,
                    default=None,
                ),
            ],
            examples=[
                'view_file(path="src/main.py")',
                'view_file(path="README.md", start_line=1, end_line=50)',
            ],
            tags=["filesystem", "read", "view"],
            timeout_seconds=10,
        )

    def execute(self, ctx: SkillContext, **kwargs) -> SkillResult:
        try:
            validated = self.validate_params(**kwargs)
            path = Path(validated["path"])
            if not path.is_absolute():
                path = Path(ctx.working_dir) / path

            if not path.exists():
                return SkillResult(success=False, error=f"File not found: {path}")

            content = path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines(keepends=True)

            start = max(1, validated.get("start_line", 1) or 1)
            end = validated.get("end_line") or len(lines)

            selected = lines[start - 1 : end]
            numbered = [f"{i + start:4d} │ {line.rstrip()}" for i, line in enumerate(selected)]

            output = "\n".join(numbered)
            truncated = False
            if len(output.encode("utf-8")) > ctx.max_output_bytes:
                output = output.encode("utf-8")[: ctx.max_output_bytes].decode(
                    "utf-8", errors="replace"
                )
                truncated = True

            return SkillResult(
                success=True,
                output=output,
                truncated=truncated,
                metadata={"path": str(path), "total_lines": len(lines)},
            )
        except Exception as e:
            return SkillResult(success=False, error=str(e))


class EditFileSkill(Skill):
    """Edit a file by replacing exact text."""

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="edit_file",
            description=(
                "Edit a file by finding and replacing exact text. "
                "The old_text must match exactly (including whitespace and indentation). "
                "Prefer this over write_file for modifying existing files."
            ),
            permissions=[Permission.FS_WRITE],
            params=[
                SkillParam(
                    name="path",
                    type="string",
                    description="Path to the file to edit",
                ),
                SkillParam(
                    name="old_text",
                    type="string",
                    description="Exact text to find and replace (must match exactly)",
                ),
                SkillParam(
                    name="new_text",
                    type="string",
                    description="Text to replace with",
                ),
            ],
            examples=[
                'edit_file(path="config.yaml", old_text="debug: false", new_text="debug: true")',
                'edit_file(path="main.py", old_text="def old_name(", new_text="def new_name(")',
            ],
            tags=["filesystem", "write", "edit"],
            timeout_seconds=10,
            requires_approval=True,
        )

    def execute(self, ctx: SkillContext, **kwargs) -> SkillResult:
        try:
            validated = self.validate_params(**kwargs)
            path = Path(validated["path"])
            if not path.is_absolute():
                path = Path(ctx.working_dir) / path

            if not path.exists():
                return SkillResult(success=False, error=f"File not found: {path}")

            content = path.read_text(encoding="utf-8")
            old_text = validated["old_text"]
            new_text = validated["new_text"]

            if old_text not in content:
                return SkillResult(
                    success=False,
                    error=f"Text not found in {path}. Make sure the old_text matches exactly.",
                )

            count = content.count(old_text)
            new_content = content.replace(old_text, new_text, 1)

            path.write_text(new_content, encoding="utf-8")

            msg = f"Edited {path}"
            if count > 1:
                msg += f" ({count} occurrences found, replaced first one)"

            return SkillResult(
                success=True,
                output=msg,
                metadata={"path": str(path), "occurrences": count},
            )
        except Exception as e:
            return SkillResult(success=False, error=str(e))

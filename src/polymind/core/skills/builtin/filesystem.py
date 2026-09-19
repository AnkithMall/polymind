"""Filesystem skills — read, write, list, search files."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path

from polymind.core.skills.base import Skill
from polymind.core.skills.types import (
    Permission,
    SkillContext,
    SkillManifest,
    SkillParam,
    SkillResult,
)


class ReadFileSkill(Skill):
    """Read the contents of a file."""

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="read_file",
            description="Read the contents of a text file. Returns the full text content.",
            permissions=[Permission.FS_READ],
            params=[
                SkillParam(
                    name="path",
                    type="string",
                    description="Path to the file to read (absolute or relative to working dir)",
                ),
                SkillParam(
                    name="offset",
                    type="integer",
                    description="Line number to start reading from (0-based). Default: 0",
                    required=False,
                    default=0,
                ),
                SkillParam(
                    name="limit",
                    type="integer",
                    description="Maximum number of lines to read. Default: all",
                    required=False,
                    default=None,
                ),
            ],
            examples=[
                'read_file(path="README.md")',
                'read_file(path="src/main.py", offset=0, limit=50)',
            ],
            tags=["filesystem", "read"],
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
            if not path.is_file():
                return SkillResult(success=False, error=f"Not a file: {path}")

            content = path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines(keepends=True)

            offset = validated.get("offset", 0) or 0
            limit = validated.get("limit")

            if offset > 0:
                lines = lines[offset:]
            if limit is not None:
                lines = lines[:limit]

            output = "".join(lines)
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
                metadata={"path": str(path), "lines": len(lines)},
            )
        except Exception as e:
            return SkillResult(success=False, error=str(e))


class WriteFileSkill(Skill):
    """Write content to a file."""

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="write_file",
            description="Write text content to a file. Creates parent directories if needed. Overwrites existing files.",
            permissions=[Permission.FS_WRITE],
            params=[
                SkillParam(
                    name="path",
                    type="string",
                    description="Path to the file to write",
                ),
                SkillParam(
                    name="content",
                    type="string",
                    description="Text content to write to the file",
                ),
            ],
            examples=[
                'write_file(path="output.txt", content="Hello world")',
                'write_file(path="src/new.py", content="def main():\\n    print(\'hi\')")',
            ],
            tags=["filesystem", "write"],
            timeout_seconds=10,
            requires_approval=True,
        )

    def execute(self, ctx: SkillContext, **kwargs) -> SkillResult:
        try:
            validated = self.validate_params(**kwargs)
            path = Path(validated["path"])
            if not path.is_absolute():
                path = Path(ctx.working_dir) / path

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(validated["content"], encoding="utf-8")

            return SkillResult(
                success=True,
                output=f"File written: {path} ({len(validated['content'])} bytes)",
                metadata={"path": str(path), "bytes_written": len(validated["content"])},
            )
        except Exception as e:
            return SkillResult(success=False, error=str(e))


class ListDirSkill(Skill):
    """List directory contents."""

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="list_dir",
            description="List files and subdirectories in a directory. Shows file type, size, and name.",
            permissions=[Permission.FS_READ],
            params=[
                SkillParam(
                    name="path",
                    type="string",
                    description="Directory path to list. Default: working directory",
                    required=False,
                    default=".",
                ),
                SkillParam(
                    name="pattern",
                    type="string",
                    description="Glob pattern to filter files (e.g., '*.py')",
                    required=False,
                    default=None,
                ),
                SkillParam(
                    name="recursive",
                    type="boolean",
                    description="List recursively. Default: false",
                    required=False,
                    default=False,
                ),
            ],
            examples=[
                'list_dir(path="src/")',
                'list_dir(path=".", pattern="*.py")',
                'list_dir(path="docs/", recursive=true)',
            ],
            tags=["filesystem", "list"],
            timeout_seconds=10,
        )

    def execute(self, ctx: SkillContext, **kwargs) -> SkillResult:
        try:
            validated = self.validate_params(**kwargs)
            path = Path(validated["path"])
            if not path.is_absolute():
                path = Path(ctx.working_dir) / path

            if not path.exists():
                return SkillResult(success=False, error=f"Directory not found: {path}")
            if not path.is_dir():
                return SkillResult(success=False, error=f"Not a directory: {path}")

            pattern = validated.get("pattern")
            recursive = validated.get("recursive", False)

            entries = []
            if recursive:
                for item in sorted(path.rglob(pattern or "*")):
                    entries.append(_format_entry(item, path))
            else:
                for item in sorted(path.iterdir()):
                    if pattern and not fnmatch.fnmatch(item.name, pattern):
                        continue
                    entries.append(_format_entry(item, path))

            output = "\n".join(entries) if entries else "(empty directory)"
            return SkillResult(
                success=True,
                output=output,
                metadata={"path": str(path), "count": len(entries)},
            )
        except Exception as e:
            return SkillResult(success=False, error=str(e))


class SearchFilesSkill(Skill):
    """Search for files by name pattern or content."""

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="search_files",
            description="Search for files matching a pattern, or search file contents for text.",
            permissions=[Permission.FS_READ],
            params=[
                SkillParam(
                    name="path",
                    type="string",
                    description="Directory to search in",
                    required=False,
                    default=".",
                ),
                SkillParam(
                    name="pattern",
                    type="string",
                    description="Glob pattern for filenames (e.g., '*.py')",
                    required=False,
                    default=None,
                ),
                SkillParam(
                    name="content",
                    type="string",
                    description="Text to search for within files",
                    required=False,
                    default=None,
                ),
                SkillParam(
                    name="max_results",
                    type="integer",
                    description="Maximum results to return. Default: 50",
                    required=False,
                    default=50,
                ),
            ],
            examples=[
                'search_files(path="src/", pattern="*.py", content="def main")',
                'search_files(path=".", pattern="README*")',
            ],
            tags=["filesystem", "search"],
            timeout_seconds=30,
        )

    def execute(self, ctx: SkillContext, **kwargs) -> SkillResult:
        try:
            validated = self.validate_params(**kwargs)
            search_path = Path(validated["path"])
            if not search_path.is_absolute():
                search_path = Path(ctx.working_dir) / search_path

            pattern = validated.get("pattern")
            content_search = validated.get("content")
            max_results = validated.get("max_results", 50)

            results = []
            files_searched = 0

            for root, _dirs, files in os.walk(search_path):
                for fname in files:
                    if len(results) >= max_results:
                        break
                    if pattern and not fnmatch.fnmatch(fname, pattern):
                        continue

                    fpath = Path(root) / fname
                    files_searched += 1

                    if content_search:
                        try:
                            text = fpath.read_text(encoding="utf-8", errors="replace")
                            if content_search.lower() in text.lower():
                                # Find matching lines
                                for i, line in enumerate(text.splitlines(), 1):
                                    if content_search.lower() in line.lower():
                                        results.append(f"{fpath}:{i}: {line.strip()[:200]}")
                                        if len(results) >= max_results:
                                            break
                        except (OSError, UnicodeDecodeError):
                            continue
                    else:
                        results.append(str(fpath.relative_to(search_path)))

                if len(results) >= max_results:
                    break

            output = "\n".join(results) if results else "No matches found"
            return SkillResult(
                success=True,
                output=output,
                metadata={"files_searched": files_searched, "matches": len(results)},
            )
        except Exception as e:
            return SkillResult(success=False, error=str(e))


def _format_entry(item: Path, base: Path) -> str:
    """Format a directory entry for display."""
    try:
        rel = item.relative_to(base)
    except ValueError:
        rel = item

    if item.is_dir():
        return f"  📁 {rel}/"
    else:
        size = item.stat().st_size
        if size < 1024:
            size_str = f"{size}B"
        elif size < 1024 * 1024:
            size_str = f"{size / 1024:.1f}KB"
        else:
            size_str = f"{size / (1024 * 1024):.1f}MB"
        return f"  📄 {rel} ({size_str})"

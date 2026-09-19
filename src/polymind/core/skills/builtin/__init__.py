"""Built-in skills — filesystem, shell, python, web, editor."""

from polymind.core.skills.builtin.editor import EditFileSkill, ViewFileSkill
from polymind.core.skills.builtin.filesystem import (
    ListDirSkill,
    ReadFileSkill,
    SearchFilesSkill,
    WriteFileSkill,
)
from polymind.core.skills.builtin.python_exec import PythonExecSkill
from polymind.core.skills.builtin.shell import ShellSkill
from polymind.core.skills.builtin.web import WebFetchSkill

BUILTIN_SKILLS = [
    ReadFileSkill,
    WriteFileSkill,
    ListDirSkill,
    SearchFilesSkill,
    ShellSkill,
    PythonExecSkill,
    WebFetchSkill,
    ViewFileSkill,
    EditFileSkill,
]

__all__ = [
    "BUILTIN_SKILLS",
    "ReadFileSkill",
    "WriteFileSkill",
    "ListDirSkill",
    "SearchFilesSkill",
    "ShellSkill",
    "PythonExecSkill",
    "WebFetchSkill",
    "ViewFileSkill",
    "EditFileSkill",
]

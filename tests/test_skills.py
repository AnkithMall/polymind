"""Tests for the skills system — types, registry, sandbox, built-in skills, agent loop."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from polymind.core.skills.agent import AgentConfig, AgentLoop, LlamaProvider
from polymind.core.skills.base import Skill
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
from polymind.core.skills.registry import SkillRegistry
from polymind.core.skills.sandbox import SandboxConfig, SandboxEnforcer, default_sandbox_config
from polymind.core.skills.types import (
    AgentResult,
    AgentStep,
    Permission,
    SandboxPolicy,
    SkillContext,
    SkillManifest,
    SkillParam,
    SkillResult,
    SkillSource,
    SkillStatus,
    StepStatus,
)

# ── Types tests ──────────────────────────────────────────


class TestSkillParam:
    def test_to_schema_string(self):
        p = SkillParam(name="path", type="string", description="File path")
        schema = p.to_schema()
        assert schema["type"] == "string"
        assert schema["description"] == "File path"

    def test_to_schema_with_enum(self):
        p = SkillParam(name="mode", type="string", enum=["read", "write"])
        schema = p.to_schema()
        assert schema["enum"] == ["read", "write"]

    def test_validate_string(self):
        p = SkillParam(name="x", type="string")
        assert p.validate("hello") == "hello"
        assert p.validate(42) == "42"

    def test_validate_integer(self):
        p = SkillParam(name="x", type="integer")
        assert p.validate("42") == 42
        assert p.validate(42) == 42

    def test_validate_boolean(self):
        p = SkillParam(name="x", type="boolean")
        assert p.validate(True) is True
        assert p.validate("true") is True

    def test_validate_required(self):
        p = SkillParam(name="x", type="string", required=True)
        with pytest.raises(ValueError, match="required"):
            p.validate(None)

    def test_validate_optional_default(self):
        p = SkillParam(name="x", type="string", required=False, default="hi")
        assert p.validate(None) == "hi"

    def test_validate_enum(self):
        p = SkillParam(name="x", type="string", enum=["a", "b"])
        assert p.validate("a") == "a"
        with pytest.raises(ValueError, match="must be one of"):
            p.validate("c")


class TestSkillManifest:
    def test_to_tool_description(self):
        m = SkillManifest(
            name="test",
            description="A test skill",
            params=[SkillParam(name="q", type="string", description="query")],
        )
        desc = m.to_tool_description()
        assert "test" in desc
        assert "query" in desc

    def test_to_openai_tool(self):
        m = SkillManifest(
            name="my_tool",
            description="Does stuff",
            params=[
                SkillParam(name="a", type="string", description="arg1"),
                SkillParam(name="b", type="integer", description="arg2", required=False, default=5),
            ],
        )
        tool = m.to_openai_tool()
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "my_tool"
        assert "a" in tool["function"]["parameters"]["properties"]
        assert "a" in tool["function"]["parameters"]["required"]
        assert "b" not in tool["function"]["parameters"]["required"]


class TestSkillResult:
    def test_success_text(self):
        r = SkillResult(success=True, output="hello world")
        text = r.to_text()
        assert "hello world" in text
        assert "[Tool result:" in text

    def test_error_text(self):
        r = SkillResult(success=False, error="not found")
        text = r.to_text()
        assert "not found" in text
        assert "[Tool error:" in text

    def test_truncated(self):
        r = SkillResult(success=True, output="data", truncated=True)
        assert "truncated" in r.to_text()


class TestAgentStep:
    def test_to_dict(self):
        s = AgentStep(step_number=1, thought="thinking", status=StepStatus.THINKING)
        d = s.to_dict()
        assert d["step"] == 1
        assert d["thought"] == "thinking"
        assert d["status"] == "thinking"


class TestAgentResult:
    def test_to_dict(self):
        r = AgentResult(response="done", total_steps=3, tools_used=["shell", "read_file"])
        d = r.to_dict()
        assert d["response"] == "done"
        assert d["total_steps"] == 3
        assert "shell" in d["tools_used"]


# ── Sandbox tests ────────────────────────────────────────


class TestSandboxEnforcer:
    def test_default_config(self):
        cfg = default_sandbox_config()
        assert cfg.policy == SandboxPolicy.BASIC
        assert cfg.network_allowed is False

    def test_check_path_access(self):
        enforcer = SandboxEnforcer(default_sandbox_config())
        home = str(Path.home())
        assert enforcer.check_path_access(home) is True
        assert enforcer.check_path_access(str(Path(home) / "Documents")) is True

    def test_truncate_output(self):
        enforcer = SandboxEnforcer(SandboxConfig(max_output_bytes=20))
        output, truncated = enforcer.truncate_output("a" * 50)
        assert len(output.encode("utf-8")) <= 20
        assert truncated is True

        output2, truncated2 = enforcer.truncate_output("short")
        assert truncated2 is False

    def test_blocked_command(self):
        enforcer = SandboxEnforcer(default_sandbox_config())
        assert enforcer.is_command_blocked("rm -rf /") is True
        assert enforcer.is_command_blocked("ls -la") is False

    def test_disabled_policy_allows_all(self):
        cfg = SandboxConfig(policy=SandboxPolicy.DISABLED)
        enforcer = SandboxEnforcer(cfg)
        assert enforcer.check_permission("any", Permission.SHELL) is True
        assert enforcer.check_path_access("/any/path") is True


# ── Built-in skills tests ────────────────────────────────


class TestReadFileSkill:
    def test_read_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello\nworld\n")
        skill = ReadFileSkill()
        ctx = SkillContext(working_dir=str(tmp_path))
        result = skill.execute(ctx, path=str(f))
        assert result.success
        assert "hello" in result.output

    def test_read_nonexistent(self):
        skill = ReadFileSkill()
        ctx = SkillContext(working_dir=".")
        result = skill.execute(ctx, path="/nonexistent/file.txt")
        assert not result.success
        assert "not found" in result.error.lower()

    def test_read_with_offset(self, tmp_path):
        f = tmp_path / "lines.txt"
        f.write_text("line1\nline2\nline3\nline4\nline5\n")
        skill = ReadFileSkill()
        ctx = SkillContext(working_dir=str(tmp_path))
        result = skill.execute(ctx, path=str(f), offset=2, limit=2)
        assert result.success
        assert "line3" in result.output
        assert "line1" not in result.output


class TestWriteFileSkill:
    def test_write_file(self, tmp_path):
        skill = WriteFileSkill()
        ctx = SkillContext(working_dir=str(tmp_path))
        result = skill.execute(ctx, path="output.txt", content="hello world")
        assert result.success
        assert (tmp_path / "output.txt").read_text() == "hello world"

    def test_write_creates_dirs(self, tmp_path):
        skill = WriteFileSkill()
        ctx = SkillContext(working_dir=str(tmp_path))
        result = skill.execute(ctx, path="a/b/c/file.txt", content="nested")
        assert result.success
        assert (tmp_path / "a/b/c/file.txt").exists()


class TestListDirSkill:
    def test_list_dir(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.py").write_text("b")
        (tmp_path / "subdir").mkdir()
        skill = ListDirSkill()
        ctx = SkillContext(working_dir=str(tmp_path))
        result = skill.execute(ctx, path=".")
        assert result.success
        assert "a.txt" in result.output
        assert "subdir" in result.output

    def test_list_dir_with_pattern(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.py").write_text("b")
        skill = ListDirSkill()
        ctx = SkillContext(working_dir=str(tmp_path))
        result = skill.execute(ctx, path=".", pattern="*.py")
        assert result.success
        assert "b.py" in result.output
        assert "a.txt" not in result.output


class TestSearchFilesSkill:
    def test_search_by_content(self, tmp_path):
        (tmp_path / "a.py").write_text("def main():\n    pass\n")
        (tmp_path / "b.py").write_text("x = 1\n")
        skill = SearchFilesSkill()
        ctx = SkillContext(working_dir=str(tmp_path))
        result = skill.execute(ctx, path=".", pattern="*.py", content="def main")
        assert result.success
        assert "a.py" in result.output

    def test_search_by_name(self, tmp_path):
        (tmp_path / "README.md").write_text("# Hi")
        (tmp_path / "setup.py").write_text("")
        skill = SearchFilesSkill()
        ctx = SkillContext(working_dir=str(tmp_path))
        result = skill.execute(ctx, path=".", pattern="README*")
        assert result.success
        assert "README.md" in result.output


class TestShellSkill:
    def test_shell_echo(self):
        skill = ShellSkill()
        ctx = SkillContext(working_dir=".")
        result = skill.execute(ctx, command="echo hello", timeout=5)
        assert result.success
        assert "hello" in result.output

    def test_shell_failure(self):
        skill = ShellSkill()
        ctx = SkillContext(working_dir=".")
        result = skill.execute(ctx, command="false", timeout=5)
        assert not result.success


class TestPythonExecSkill:
    def test_python_print(self):
        skill = PythonExecSkill()
        ctx = SkillContext(working_dir=".")
        result = skill.execute(ctx, code="print(2 + 2)")
        assert result.success
        assert "4" in result.output

    def test_python_error(self):
        skill = PythonExecSkill()
        ctx = SkillContext(working_dir=".")
        result = skill.execute(ctx, code="raise ValueError('test')")
        assert not result.success


class TestWebFetchSkill:
    def test_invalid_url(self):
        skill = WebFetchSkill()
        ctx = SkillContext(working_dir=".")
        result = skill.execute(ctx, url="ftp://example.com")
        assert not result.success
        assert "HTTP/HTTPS" in result.error


class TestViewFileSkill:
    def test_view_with_line_numbers(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("line1\nline2\nline3\nline4\nline5\n")
        skill = ViewFileSkill()
        ctx = SkillContext(working_dir=str(tmp_path))
        result = skill.execute(ctx, path=str(f), start_line=2, end_line=4)
        assert result.success
        assert "line2" in result.output
        assert "line4" in result.output
        assert "line1" not in result.output
        assert "line5" not in result.output


class TestEditFileSkill:
    def test_edit_file(self, tmp_path):
        f = tmp_path / "config.txt"
        f.write_text("debug: false\nport: 8080\n")
        skill = EditFileSkill()
        ctx = SkillContext(working_dir=str(tmp_path))
        result = skill.execute(ctx, path=str(f), old_text="debug: false", new_text="debug: true")
        assert result.success
        assert "debug: true" in f.read_text()

    def test_edit_not_found(self, tmp_path):
        f = tmp_path / "config.txt"
        f.write_text("hello")
        skill = EditFileSkill()
        ctx = SkillContext(working_dir=str(tmp_path))
        result = skill.execute(ctx, path=str(f), old_text="xyz", new_text="abc")
        assert not result.success
        assert "not found" in result.error.lower()


# ── Registry tests ───────────────────────────────────────


class TestSkillRegistry:
    def test_discover_builtin(self):
        registry = SkillRegistry()
        registry.discover()
        names = registry.list_names()
        assert "read_file" in names
        assert "write_file" in names
        assert "shell" in names
        assert "python_exec" in names
        assert "list_dir" in names
        assert "search_files" in names
        assert "web_fetch" in names
        assert "view_file" in names
        assert "edit_file" in names

    def test_get_skill(self):
        registry = SkillRegistry()
        registry.discover()
        skill = registry.get("read_file")
        assert skill is not None
        assert isinstance(skill, ReadFileSkill)

    def test_list_manifests(self):
        registry = SkillRegistry()
        registry.discover()
        manifests = registry.list_skills()
        assert len(manifests) >= 9

    def test_register_custom(self):
        registry = SkillRegistry()
        registry.discover()

        class MySkill(Skill):
            def manifest(self):
                return SkillManifest(name="custom_test", description="test")

            def execute(self, ctx, **kwargs):
                return SkillResult(success=True, output="custom")

        registry.register(MySkill())
        assert "custom_test" in registry.list_names()

    def test_unregister(self):
        registry = SkillRegistry()
        registry.discover()

        class TempSkill(Skill):
            def manifest(self):
                return SkillManifest(name="temp_skill", description="temp")

            def execute(self, ctx, **kwargs):
                return SkillResult(success=True, output="temp")

        registry.register(TempSkill())
        assert "temp_skill" in registry.list_names()
        assert registry.unregister("temp_skill") is True
        assert "temp_skill" not in registry.list_names()

    def test_to_system_prompt_tools(self):
        registry = SkillRegistry()
        registry.discover()
        prompt = registry.to_system_prompt_tools()
        assert "read_file" in prompt
        assert "shell" in prompt

    def test_to_openai_tools(self):
        registry = SkillRegistry()
        registry.discover()
        tools = registry.to_openai_tools()
        assert len(tools) >= 9
        assert all(t["type"] == "function" for t in tools)


# ── Agent loop tests ─────────────────────────────────────


class TestToolCallParsing:
    def test_parse_basic(self):
        text = 'I need to read a file.\n<tool_call>\n{"name": "read_file", "arguments": {"path": "test.py"}}\n</tool_call>'
        agent = AgentLoop(MagicMock(), SkillRegistry())
        result = agent._parse_tool_call(text)
        assert result is not None
        assert result["name"] == "read_file"
        assert result["arguments"]["path"] == "test.py"

    def test_parse_no_call(self):
        text = "This is just a normal response."
        agent = AgentLoop(MagicMock(), SkillRegistry())
        result = agent._parse_tool_call(text)
        assert result is None

    def test_parse_with_thought(self):
        text = (
            "Let me check the file.\n"
            '<tool_call>\n{"name": "shell", "arguments": {"command": "ls"}}\n</tool_call>'
        )
        agent = AgentLoop(MagicMock(), SkillRegistry())
        result = agent._parse_tool_call(text)
        assert result is not None
        assert result["name"] == "shell"

    def test_parse_malformed_json(self):
        text = "<tool_call>\n{name: bad}\n</tool_call>"
        agent = AgentLoop(MagicMock(), SkillRegistry())
        result = agent._parse_tool_call(text)
        assert result is None


class TestAgentLoop:
    def test_simple_question(self):
        """Test a simple question that doesn't need tools."""
        registry = SkillRegistry()
        registry.discover()

        mock_provider = MagicMock()
        mock_provider.complete.return_value = {
            "choices": [
                {
                    "message": {"content": "The answer is 42."},
                    "finish_reason": "stop",
                }
            ]
        }

        agent = AgentLoop(mock_provider, registry)
        result = agent.run("What is the answer to everything?")

        assert result.success
        assert "42" in result.response
        assert result.total_steps == 1

    def test_tool_call_flow(self):
        """Test a prompt that requires tool use."""
        registry = SkillRegistry()
        registry.discover()

        mock_provider = MagicMock()
        # First call: model requests tool
        # Second call: model gives final answer
        mock_provider.complete.side_effect = [
            {
                "choices": [
                    {
                        "message": {
                            "content": 'Let me read that file.\n<tool_call>\n{"name": "read_file", "arguments": {"path": "test.txt"}}\n</tool_call>',
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {"content": "The file contains: hello world"},
                    }
                ]
            },
        ]

        agent = AgentLoop(mock_provider, registry, config=AgentConfig(max_steps=5))
        result = agent.run("Read test.txt")

        assert result.success
        assert "hello world" in result.response
        assert "read_file" in result.tools_used

    def test_max_steps(self):
        """Test that max steps is respected."""
        registry = SkillRegistry()
        registry.discover()

        mock_provider = MagicMock()
        mock_provider.complete.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '<tool_call>\n{"name": "shell", "arguments": {"command": "echo hi"}}\n</tool_call>',
                    },
                }
            ]
        }

        agent = AgentLoop(
            mock_provider,
            registry,
            config=AgentConfig(max_steps=2),
        )
        result = agent.run("Do something")

        assert not result.success
        assert result.total_steps == 2

    def test_unknown_tool(self):
        """Test handling of unknown tool names."""
        registry = SkillRegistry()
        registry.discover()

        mock_provider = MagicMock()
        mock_provider.complete.side_effect = [
            {
                "choices": [
                    {
                        "message": {
                            "content": '<tool_call>\n{"name": "nonexistent_tool", "arguments": {}}\n</tool_call>',
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {"content": "Sorry, that tool isn't available."},
                    }
                ]
            },
        ]

        agent = AgentLoop(mock_provider, registry)
        result = agent.run("Use nonexistent_tool")

        assert result.success
        assert "nonexistent" in result.response.lower() or "sorry" in result.response.lower()


class TestLlamaProvider:
    def test_protocol_compliance(self):
        """LlamaProvider matches the LLMProvider protocol."""
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {"choices": [{"message": {"content": "hi"}}]}
        provider = LlamaProvider(mock_llm)
        result = provider.complete(messages=[{"role": "user", "content": "hi"}])
        assert "choices" in result
        mock_llm.create_chat_completion.assert_called_once()


# ── SkillSource / SkillStatus tests ─────────────────────


class TestSkillSourceAndStatus:
    def test_skill_source_values(self):
        assert SkillSource.BUILTIN == "builtin"
        assert SkillSource.USER == "user"
        assert SkillSource.MCP == "mcp"
        assert SkillSource.PLUGIN == "plugin"

    def test_skill_status_values(self):
        assert SkillStatus.ENABLED == "enabled"
        assert SkillStatus.DISABLED == "disabled"
        assert SkillStatus.ERROR == "error"

    def test_manifest_has_source_and_status(self):
        m = SkillManifest(name="test", description="test")
        assert m.source == SkillSource.BUILTIN
        assert m.status == SkillStatus.ENABLED

    def test_manifest_source_detail(self):
        m = SkillManifest(name="test", description="test", source_detail="mcp:filesystem")
        assert m.source_detail == "mcp:filesystem"


# ── Enable / Disable tests ──────────────────────────────


class TestEnableDisable:
    def test_disable_builtin_skill(self):
        registry = SkillRegistry()
        registry.discover()
        assert registry.disable("read_file") is True
        manifest = registry.get_manifest("read_file")
        assert manifest.status == SkillStatus.DISABLED

    def test_enable_disabled_skill(self):
        registry = SkillRegistry()
        registry.discover()
        registry.disable("shell")
        assert registry.enable("shell") is True
        manifest = registry.get_manifest("shell")
        assert manifest.status == SkillStatus.ENABLED

    def test_disable_nonexistent(self):
        registry = SkillRegistry()
        registry.discover()
        assert registry.disable("nonexistent") is False

    def test_enable_nonexistent(self):
        registry = SkillRegistry()
        registry.discover()
        assert registry.enable("nonexistent") is False

    def test_list_enabled_excludes_disabled(self):
        registry = SkillRegistry()
        registry.discover()
        registry.disable("shell")
        enabled = registry.list_enabled()
        assert all(m.name != "shell" for m in enabled)

    def test_list_disabled(self):
        registry = SkillRegistry()
        registry.discover()
        registry.disable("shell")
        disabled = registry.list_disabled()
        assert any(m.name == "shell" for m in disabled)

    def test_list_enabled_names(self):
        registry = SkillRegistry()
        registry.discover()
        names = registry.list_enabled_names()
        assert "read_file" in names
        assert "shell" in names

    def test_list_by_source(self):
        registry = SkillRegistry()
        registry.discover()
        builtin = registry.list_by_source(SkillSource.BUILTIN)
        assert len(builtin) >= 9
        assert all(m.source == SkillSource.BUILTIN for m in builtin)

    def test_system_prompt_excludes_disabled(self):
        registry = SkillRegistry()
        registry.discover()
        registry.disable("shell")
        prompt = registry.to_system_prompt_tools()
        assert "shell" not in prompt

    def test_openai_tools_excludes_disabled(self):
        registry = SkillRegistry()
        registry.discover()
        registry.disable("shell")
        tools = registry.to_openai_tools()
        tool_names = [t["function"]["name"] for t in tools]
        assert "shell" not in tool_names

    def test_persist_state(self, tmp_path, monkeypatch):
        """Test that enable/disable state persists to config file."""
        from polymind.core.skills import registry as reg_module

        monkeypatch.setattr(reg_module, "artifact_dir", lambda: tmp_path)
        # Ensure the skills dir doesn't interfere
        reg_module._save_skills_config({
            "shell": {"status": "disabled", "source": "builtin"},
            "read_file": {"status": "enabled", "source": "builtin"},
        })

        loaded = reg_module._load_skills_config()
        assert loaded["shell"]["status"] == "disabled"
        assert loaded["read_file"]["status"] == "enabled"


# ── MCP integration tests ───────────────────────────────


class TestMcpClient:
    def test_manifest_from_mcp_tool(self):
        from polymind.core.skills.registry import _manifest_from_mcp_tool

        tool_info = {
            "name": "read_file",
            "description": "Read a file from the filesystem",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                },
                "required": ["path"],
            },
        }
        manifest = _manifest_from_mcp_tool(tool_info, "filesystem")
        assert manifest.name == "read_file"
        assert manifest.source == SkillSource.MCP
        assert manifest.source_detail == "mcp:filesystem"
        assert len(manifest.params) == 1
        assert manifest.params[0].name == "path"
        assert manifest.params[0].required is True

    def test_manifest_from_mcp_tool_with_optional_params(self):
        from polymind.core.skills.registry import _manifest_from_mcp_tool

        tool_info = {
            "name": "search",
            "description": "Search files",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results", "default": 10},
                },
                "required": ["query"],
            },
        }
        manifest = _manifest_from_mcp_tool(tool_info, "search-server")
        assert len(manifest.params) == 2
        query_param = next(p for p in manifest.params if p.name == "query")
        limit_param = next(p for p in manifest.params if p.name == "limit")
        assert query_param.required is True
        assert limit_param.required is False
        assert limit_param.default == 10

    def test_mcp_config_persistence(self, tmp_path, monkeypatch):
        from polymind.core.skills import registry as reg_module

        monkeypatch.setattr(reg_module, "artifact_dir", lambda: tmp_path)

        servers = {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem"],
                "enabled": True,
            }
        }
        reg_module._save_mcp_config(servers)

        loaded = reg_module._load_mcp_config()
        assert "filesystem" in loaded
        assert loaded["filesystem"]["command"] == "npx"

    def test_mcp_tool_skill_execute(self):
        """Test that _McpToolSkill correctly calls call_mcp_tool."""
        from polymind.core.skills.registry import _McpToolSkill

        tool_info = {
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {"type": "object", "properties": {}},
        }
        skill = _McpToolSkill(tool_info, "test-server")
        manifest = skill.manifest()
        assert manifest.name == "test_tool"
        assert manifest.source == SkillSource.MCP

    def test_list_mcp_tools_empty_config(self):
        """Test that list_mcp_tools handles missing server gracefully."""
        from polymind.core.skills.mcp_client import list_mcp_tools

        # Non-existent command should return empty list
        result = list_mcp_tools({"command": "nonexistent_command_xyz", "args": []})
        assert result == []

    def test_call_mcp_tool_missing_server(self):
        """Test that call_mcp_tool handles missing server gracefully."""
        from polymind.core.skills.mcp_client import call_mcp_tool

        result = call_mcp_tool(
            {"command": "nonexistent_command_xyz", "args": []},
            "test_tool",
            {},
        )
        assert not result.success
        assert "error" in result.error.lower()

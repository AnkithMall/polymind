"""MCP client — connects to MCP servers and discovers/calls tools.

Supports the Model Context Protocol (MCP) over stdio transport.
This lets polymind use any third-party MCP server (like Claude's servers)
as skills for local LLMs.

MCP servers are separate processes that communicate via JSON-RPC 2.0
over stdin/stdout. This client:
1. Spawns the server process
2. Sends tools/list to discover available tools
3. Sends tools/call to execute tools
4. Returns results as polymind SkillResult objects

Usage:
    from polymind.core.skills.mcp_client import list_mcp_tools, call_mcp_tools

    server = {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem"]}
    tools = list_mcp_tools(server)  # Discover tools
    result = call_mcp_tool(server, "read_file", {"path": "/tmp/test.txt"})
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

from polymind.core.skills.types import SkillResult

# MCP protocol version we support
MCP_PROTOCOL_VERSION = "2024-11-05"


def list_mcp_tools(server_cfg: dict) -> list[dict]:
    """Connect to an MCP server and list its tools.

    Args:
        server_cfg: Server configuration with keys:
            - command: str (e.g., "npx", "python", "node")
            - args: list[str] (e.g., ["-y", "@modelcontextprotocol/server-filesystem"])
            - env: dict[str, str] (optional environment variables)
            - cwd: str (optional working directory)

    Returns:
        List of tool dicts with keys: name, description, inputSchema
    """
    try:
        proc = _start_server(server_cfg)
        try:
            # Initialize
            _send_json(proc, {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "polymind", "version": "0.1.2"},
                },
            })
            _read_json(proc)  # Read initialize response

            # Send initialized notification
            _send_json(proc, {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            })

            # List tools
            _send_json(proc, {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {},
            })
            response = _read_json(proc)
            tools = response.get("result", {}).get("tools", [])
            return tools
        finally:
            _stop_server(proc)
    except Exception:
        return []


def call_mcp_tool(server_cfg: dict, tool_name: str, arguments: dict[str, Any]) -> SkillResult:
    """Call a tool on an MCP server.

    Args:
        server_cfg: Server configuration (same format as list_mcp_tools)
        tool_name: Name of the tool to call
        arguments: Tool arguments

    Returns:
        SkillResult with the tool's output
    """
    try:
        proc = _start_server(server_cfg)
        try:
            # Initialize
            _send_json(proc, {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "polymind", "version": "0.1.2"},
                },
            })
            _read_json(proc)

            _send_json(proc, {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            })

            # Call the tool
            _send_json(proc, {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments,
                },
            })
            response = _read_json(proc)

            # Parse result
            result_data = response.get("result", {})
            content = result_data.get("content", [])

            # Extract text from content array
            texts = []
            for item in content:
                if item.get("type") == "text":
                    texts.append(item.get("text", ""))

            output = "\n".join(texts) if texts else json.dumps(result_data)

            # Check for errors
            is_error = result_data.get("isError", False)
            if is_error:
                return SkillResult(success=False, error=output)

            return SkillResult(
                success=True,
                output=output,
                metadata={"mcp_server": server_cfg.get("command", ""), "tool": tool_name},
            )
        finally:
            _stop_server(proc)
    except subprocess.TimeoutExpired:
        return SkillResult(success=False, error="MCP server timed out")
    except Exception as e:
        return SkillResult(success=False, error=f"MCP error: {e}")


def _start_server(server_cfg: dict) -> subprocess.Popen:
    """Start an MCP server process."""
    command = server_cfg.get("command", "")
    args = server_cfg.get("args", [])
    env = server_cfg.get("env")
    cwd = server_cfg.get("cwd")

    import os

    full_env = dict(os.environ) if env else None
    if full_env and env:
        full_env.update(env)

    proc = subprocess.Popen(
        [command, *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=full_env,
        cwd=cwd,
    )
    return proc


def _stop_server(proc: subprocess.Popen) -> None:
    """Stop an MCP server process."""
    try:
        proc.stdin.close()  # type: ignore
    except Exception:
        pass
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _send_json(proc: subprocess.Popen, data: dict) -> None:
    """Send a JSON-RPC message to the server's stdin."""
    message = json.dumps(data)
    proc.stdin.write(f"{message}\n".encode())  # type: ignore
    proc.stdin.flush()  # type: ignore


def _read_json(proc: subprocess.Popen) -> dict:
    """Read a JSON-RPC response from the server's stdout."""
    line = proc.stdout.readline()  # type: ignore
    if not line:
        raise ConnectionError("MCP server closed connection")
    return json.loads(line.decode())

"""Agent loop — ReAct-style orchestration for any LLM.

This is the heart of the skills system. It implements a reasoning loop
that works with ANY local LLM, regardless of whether it supports
native tool calling.

Strategy:
1. Inject tool descriptions into the system prompt
2. Instruct the model to output structured tool calls
3. Parse the model's output for tool invocations
4. Execute tools via the sandbox
5. Feed results back and repeat

For models that DO support function calling (via llama-cpp-python),
the loop can use native tool calling as an optimization.

The loop supports:
- Multi-step reasoning (ReAct pattern)
- Parallel tool calls
- User approval for dangerous operations
- Max step limits (prevent infinite loops)
- Streaming support (optional)
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from polymind.core.skills.registry import SkillRegistry
from polymind.core.skills.sandbox import SandboxConfig, SandboxEnforcer
from polymind.core.skills.types import (
    AgentResult,
    AgentStep,
    SkillResult,
    StepStatus,
)


class LLMProvider(Protocol):
    """Protocol for LLM interaction — any object implementing this works."""

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> dict:
        """Send a chat completion request. Returns OpenAI-compatible response dict."""
        ...


class LlamaProvider:
    """Adapter for llama-cpp-python Llama objects."""

    def __init__(self, llm):
        self.llm = llm

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> dict:
        kwargs: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        output = self.llm.create_chat_completion(**kwargs)
        return output


# ── System prompt templates ─────────────────────────────

AGENT_SYSTEM_PROMPT = """You are an intelligent assistant with access to tools. You can use these tools to help answer the user's question.

{tool_descriptions}

## How to use tools

When you need to use a tool, output your response in this EXACT format:

<tool_call>
{{"name": "tool_name", "arguments": {{"param1": "value1", "param2": "value2"}}}}
</tool_call>

## Rules
1. Think step by step before using a tool
2. You can use multiple tools in sequence
3. After receiving a tool result, analyze it and decide if you need more tools
4. When you have enough information, provide a final answer WITHOUT tool tags
5. Always explain what you're doing and why
6. If a tool fails, try a different approach
7. For file operations, always confirm the path is correct first

## Example

User: Read the file config.yaml and tell me the debug setting

Assistant: I'll read the config file to find the debug setting.

<tool_call>
{{"name": "read_file", "arguments": {{"path": "config.yaml"}}}}
</tool_call>

[After receiving result]

The debug setting in config.yaml is set to `true`. This means...

## Important
- ONLY use tools that are listed above
- Arguments must be valid JSON
- Do NOT output tool calls in any other format
- When you're done using tools, give a clear final answer
"""


@dataclass
class AgentConfig:
    """Configuration for the agent loop."""

    max_steps: int = 15
    max_tool_calls_per_step: int = 3
    temperature: float = 0.2
    max_tokens: int = 2048
    verbose: bool = False
    approval_callback: Any = None  # Callable[[str, str, dict], bool]


class AgentLoop:
    """ReAct-style agent loop that works with any LLM.

    Usage:
        provider = LlamaProvider(llm)
        agent = AgentLoop(provider, registry, config)
        result = agent.run("Read config.yaml and summarize it")
    """

    def __init__(
        self,
        provider: LLMProvider,
        registry: SkillRegistry,
        sandbox_config: SandboxConfig | None = None,
        config: AgentConfig | None = None,
    ):
        self.provider = provider
        self.registry = registry
        self.sandbox = SandboxEnforcer(sandbox_config or SandboxConfig())
        self.config = config or AgentConfig()

    def run(self, prompt: str, context: dict | None = None) -> AgentResult:
        """Run the agent loop on a user prompt.

        Args:
            prompt: The user's input.
            context: Optional context (working dir, env vars, etc.)

        Returns:
            AgentResult with the final response and step history.
        """
        self.registry.discover()

        steps: list[AgentStep] = []
        tools_used: list[str] = []
        step_number = 0

        # Build system prompt with tool descriptions
        tool_descriptions = self.registry.to_system_prompt_tools()
        system_content = AGENT_SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)

        # Build initial messages
        messages: list[dict] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]

        # Convert tools to OpenAI format (for models that support it)
        openai_tools = self.registry.to_openai_tools()

        while step_number < self.config.max_steps:
            step_number += 1
            step = AgentStep(step_number=step_number, status=StepStatus.THINKING)
            steps.append(step)

            try:
                # Call the LLM
                response = self.provider.complete(
                    messages=messages,
                    tools=openai_tools if openai_tools else None,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )

                # Extract the assistant message
                choice = response.get("choices", [{}])[0]
                message = choice.get("message", {})
                content = message.get("content", "")

                # Check for native tool calls (function calling)
                tool_calls = message.get("tool_calls") or choice.get("tool_calls")

                if tool_calls:
                    # Native function calling path
                    result = self._handle_native_tool_calls(
                        tool_calls, messages, step, tools_used, context
                    )
                    if result:
                        return result
                    continue

                # Check for ReAct-style tool calls in text output
                tool_call = self._parse_tool_call(content)

                if tool_call:
                    step.thought = content.split("<tool_call>")[0].strip()
                    step.tool_name = tool_call["name"]
                    step.tool_args = tool_call.get("arguments", {})

                    # Execute the tool
                    tool_result = self._execute_tool(
                        tool_call["name"],
                        tool_call.get("arguments", {}),
                        context,
                    )
                    step.tool_result = tool_result
                    step.status = StepStatus.COMPLETE

                    if tool_call["name"] not in tools_used:
                        tools_used.append(tool_call["name"])

                    # Add assistant message and tool result to conversation
                    messages.append({"role": "assistant", "content": content})
                    messages.append(
                        {
                            "role": "user",
                            "content": tool_result.to_text(),
                        }
                    )
                    continue

                # No tool call — this is the final answer
                step.thought = content
                step.status = StepStatus.COMPLETE

                return AgentResult(
                    response=content,
                    steps=steps,
                    total_steps=step_number,
                    tools_used=tools_used,
                    success=True,
                )

            except Exception as e:
                step.status = StepStatus.ERROR
                step.thought = f"Error: {e}"
                return AgentResult(
                    response=f"Agent error at step {step_number}: {e}",
                    steps=steps,
                    total_steps=step_number,
                    tools_used=tools_used,
                    success=False,
                    error=str(e),
                )

        # Max steps reached — return what we have
        return AgentResult(
            response=messages[-1].get("content", "Max steps reached without a final answer."),
            steps=steps,
            total_steps=step_number,
            tools_used=tools_used,
            success=False,
            error="Max steps reached",
        )

    def _parse_tool_call(self, text: str) -> dict | None:
        """Parse a ReAct-style tool call from the model's output.

        Looks for: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        """
        pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            # Also try without closing tag (some models omit it)
            pattern2 = r"<tool_call>\s*(\{.*?\})\s*$"
            match = re.search(pattern2, text, re.DOTALL | re.MULTILINE)

        if not match:
            return None

        try:
            data = json.loads(match.group(1))
            if "name" in data:
                return {
                    "name": data["name"],
                    "arguments": data.get("arguments", {}),
                }
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            raw = match.group(1)
            raw = raw.replace("'", '"')
            try:
                data = json.loads(raw)
                if "name" in data:
                    return {
                        "name": data["name"],
                        "arguments": data.get("arguments", {}),
                    }
            except json.JSONDecodeError:
                pass

        return None

    def _handle_native_tool_calls(
        self,
        tool_calls: list,
        messages: list[dict],
        step: AgentStep,
        tools_used: list[str],
        context: dict | None,
    ) -> AgentResult | None:
        """Handle native function-calling tool calls."""
        # Add the assistant message with tool calls
        messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})

        for tc in tool_calls[: self.config.max_tool_calls_per_step]:
            func = tc.get("function", {})
            name = func.get("name", "")
            args_str = func.get("arguments", "{}")

            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {}

            step.tool_name = name
            step.tool_args = args

            tool_result = self._execute_tool(name, args, context)
            step.tool_result = tool_result

            if name not in tools_used:
                tools_used.append(name)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": tool_result.to_text(),
                }
            )

        step.status = StepStatus.COMPLETE
        return None

    def _execute_tool(
        self,
        name: str,
        args: dict,
        context: dict | None,
    ) -> SkillResult:
        """Execute a tool through the sandbox."""
        skill = self.registry.get(name)
        if skill is None:
            return SkillResult(success=False, error=f"Unknown tool: {name}")

        manifest = self.registry.get_manifest(name)

        # Check permissions
        if manifest:
            for perm in manifest.permissions:
                if not self.sandbox.check_permission(name, perm):
                    return SkillResult(
                        success=False,
                        error=f"Permission denied: {perm.value} not allowed for {name}",
                    )

        # Check approval
        if manifest:
            for perm in manifest.permissions:
                if self.sandbox.needs_approval(name, perm):
                    if self.config.approval_callback:
                        approved = self.config.approval_callback(name, perm.value, args)
                        if not approved:
                            return SkillResult(
                                success=False,
                                error=f"User denied {perm.value} permission for {name}",
                            )
                    else:
                        # No approval callback — check if auto-approved
                        if name not in self.sandbox.config.auto_approved_skills:
                            return SkillResult(
                                success=False,
                                error=f"Tool '{name}' requires user approval. Run with --approve to allow.",
                            )

        # Create sandboxed context
        working_dir = (context or {}).get("working_dir", ".")
        skill_ctx = self.sandbox.create_context(
            skill_name=name,
            working_dir=working_dir,
            timeout=manifest.timeout_seconds if manifest else 30,
        )

        # Execute with timeout
        start = time.time()
        try:
            result = skill.execute(skill_ctx, **args)
            elapsed = time.time() - start

            # Truncate output if needed
            if result.output:
                result.output, was_truncated = self.sandbox.truncate_output(result.output)
                if was_truncated:
                    result.truncated = True

            result.metadata["execution_time_s"] = f"{elapsed:.2f}"
            return result
        except Exception as e:
            return SkillResult(
                success=False,
                error=f"Tool execution failed: {e}",
                metadata={"execution_time_s": f"{time.time() - start:.2f}"},
            )

    def stream(self, prompt: str, context: dict | None = None) -> Iterator[AgentStep]:
        """Stream agent steps (for real-time UI updates).

        Yields AgentStep objects as they are processed.
        """
        # For now, use the non-streaming loop and yield steps
        # TODO: Implement true streaming with llama-cpp-python's stream=True
        result = self.run(prompt, context)
        yield from result.steps

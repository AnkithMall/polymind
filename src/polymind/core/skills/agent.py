"""Agent loop - ReAct-style orchestration for any LLM."""

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
    AgentResult, AgentStep, SkillResult, StepStatus,
)


class LLMProvider(Protocol):
    def complete(self, messages, tools=None, temperature=0.2, max_tokens=2048) -> dict: ...


class LlamaProvider:
    def __init__(self, llm): self.llm = llm
    def complete(self, messages, tools=None, temperature=0.2, max_tokens=2048):
        kwargs = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if tools: kwargs["tools"] = tools; kwargs["tool_choice"] = "auto"
        return self.llm.create_chat_completion(**kwargs)


_TC_TAG = '<tool_call>'
_TC_CLOSE = '</tool_call>'
_TC_PATTERN = '<tool_call></tool_call>'
_XML_FUNC = '<function=(\\w+)>(.*?)</function>'
_XML_PARAM = '<parameter=(\\w+)>\\s*(.*?)\\s*</parameter>'
_XML_PARAM_FB = '<parameter=(\\w+)>\\s*(.*?)(?=<parameter=|</function|$)'

AGENT_SYSTEM_PROMPT = 'You are an intelligent assistant with access to tools.\n\n{tool_descriptions}\n\n## How to use tools\n\nWhen you need to use a tool, output EXACTLY this format:\n\n<tool_call>\n{{"name": "tool_name", "arguments": {{"param1": "value1"}}}}\n</tool_call>\n\n## Rules\n1. Think step by step before using a tool\n2. Call ONLY ONE tool at a time\n3. After receiving a tool result, decide: call another tool or give final answer\n4. For your final answer, respond normally WITHOUT tool tags\n5. Use double quotes in JSON, not single quotes\n'


@dataclass
class AgentConfig:
    max_steps: int = 15
    max_tool_calls_per_step: int = 3
    temperature: float = 0.2
    max_tokens: int = 2048
    verbose: bool = False
    approval_callback: Any = None


class AgentLoop:
    def __init__(self, provider, registry, sandbox_config=None, config=None):
        self.provider = provider
        self.registry = registry
        self.sandbox = SandboxEnforcer(sandbox_config or SandboxConfig())
        self.config = config or AgentConfig()

    def run(self, prompt, context=None):
        self.registry.discover()
        steps, tools_used, step_number = [], [], 0
        td = self.registry.to_system_prompt_tools()
        sc = AGENT_SYSTEM_PROMPT.format(tool_descriptions=td)
        messages = [{"role": "system", "content": sc}, {"role": "user", "content": prompt}]

        while step_number < self.config.max_steps:
            step_number += 1
            step = AgentStep(step_number=step_number, status=StepStatus.THINKING)
            steps.append(step)
            try:
                response = self.provider.complete(messages=messages, temperature=self.config.temperature, max_tokens=self.config.max_tokens)
                choice = response.get("choices", [{}])[0]
                content = choice.get("message", {}).get("content", "")
                tc = choice.get("message", {}).get("tool_calls") or choice.get("tool_calls")
                if tc:
                    r = self._handle_native(tc, messages, step, tools_used, context)
                    if r: return r
                    continue
                tool_call = self._parse_tool_call(content)
                if tool_call:
                    thought = content
                    for marker in [_TC_TAG, "<function="]:
                        idx = content.find(marker)
                        if idx >= 0: thought = content[:idx].strip(); break
                    step.thought, step.tool_name, step.tool_args = thought, tool_call["name"], tool_call.get("arguments", {})
                    tr = self._exec(tool_call["name"], tool_call.get("arguments", {}), context)
                    step.tool_result, step.status = tr, StepStatus.COMPLETE
                    if tool_call["name"] not in tools_used: tools_used.append(tool_call["name"])
                    messages.extend([{"role": "assistant", "content": content}, {"role": "user", "content": tr.to_text()}])
                    continue
                step.thought, step.status = content, StepStatus.COMPLETE
                return AgentResult(response=content, steps=steps, total_steps=step_number, tools_used=tools_used, success=True)
            except Exception as e:
                step.status = StepStatus.ERROR
                return AgentResult(response=f"Agent error at step {step_number}: {e}", steps=steps, total_steps=step_number, tools_used=tools_used, success=False, error=str(e))
        return AgentResult(response=messages[-1].get("content", "Max steps reached."), steps=steps, total_steps=step_number, tools_used=tools_used, success=False, error="Max steps reached")

    def _parse_tool_call(self, text):
        r = self._parse_json(text)
        return r if r else self._parse_xml(text)

    def _parse_json(self, text):
        match = re.search(_TC_PATTERN, text, re.DOTALL)
        if not match: return None
        try:
            data = json.loads(match.group(1))
            if "name" in data: return {"name": data["name"], "arguments": data.get("arguments", {})}
        except json.JSONDecodeError:
            raw = match.group(1).replace(chr(39), chr(34))
            try:
                data = json.loads(raw)
                if "name" in data: return {"name": data["name"], "arguments": data.get("arguments", {})}
            except json.JSONDecodeError: pass
        return None

    def _parse_xml(self, text):
        """Parse XML function/parameter format."""
        match = re.search(_XML_FUNC, text, re.DOTALL)
        if not match: return None
        func_name, block, params = match.group(1), match.group(2), {}
        for m in re.finditer(_XML_PARAM, block, re.DOTALL):
            params[m.group(1)] = m.group(2).strip()
        if not params:
            for m in re.finditer(_XML_PARAM_FB, block, re.DOTALL):
                params[m.group(1)] = m.group(2).strip()
        return {"name": func_name, "arguments": params}

    def _handle_native(self, tool_calls, messages, step, tools_used, context):
        messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
        for tc in tool_calls[:self.config.max_tool_calls_per_step]:
            func = tc.get("function", {})
            name = func.get("name", "")
            try: args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError: args = {}
            step.tool_name, step.tool_args = name, args
            tr = self._exec(name, args, context)
            step.tool_result = tr
            if name not in tools_used: tools_used.append(name)
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": tr.to_text()})
        step.status = StepStatus.COMPLETE
        return None

    def _exec(self, name, args, context):
        skill = self.registry.get(name)
        if skill is None: return SkillResult(success=False, error=f"Unknown tool: {name}")
        manifest = self.registry.get_manifest(name)
        if manifest:
            for perm in manifest.permissions:
                if not self.sandbox.check_permission(name, perm):
                    return SkillResult(success=False, error=f"Permission denied: {perm.value}")
            for perm in manifest.permissions:
                if self.sandbox.needs_approval(name, perm):
                    if self.config.approval_callback:
                        if not self.config.approval_callback(name, perm.value, args):
                            return SkillResult(success=False, error=f"User denied {perm.value}")
                    elif name not in self.sandbox.config.auto_approved_skills:
                        return SkillResult(success=False, error=f"Tool requires approval: {name}")
        wd = (context or {}).get("working_dir", ".")
        ctx = self.sandbox.create_context(skill_name=name, working_dir=wd, timeout=manifest.timeout_seconds if manifest else 30)
        start = time.time()
        try:
            result = skill.execute(ctx, **args)
            elapsed = time.time() - start
            if result.output:
                result.output, wt = self.sandbox.truncate_output(result.output)
                if wt: result.truncated = True
            result.metadata["execution_time_s"] = f"{elapsed:.2f}"
            return result
        except Exception as e:
            return SkillResult(success=False, error=f"Tool execution failed: {e}")

    def stream(self, prompt, context=None):
        result = self.run(prompt, context)
        yield from result.steps

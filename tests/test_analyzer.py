import json

import pytest

from polymind.core.analyzer import (
    _build_router_messages,
    _parse_plan,
    _single_task_fallback,
)
from polymind.core.types import AnalyzerPlan, DomainType


def test_build_router_messages():
    messages = _build_router_messages("write a fibonacci function")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "write a fibonacci function"
    assert "code, math" in messages[0]["content"]


def test_parse_plan_valid_json():
    raw = json.dumps(
        {
            "subtasks": [
                {"id": "t1", "domain": "code", "prompt": "write fib", "depends_on": []},
                {
                    "id": "t2",
                    "domain": "math",
                    "prompt": "test fib",
                    "depends_on": ["t1"],
                },
            ]
        }
    )
    plan = _parse_plan(raw, "original prompt")
    assert plan is not None
    assert len(plan.subtasks) == 2
    assert plan.subtasks[0].domain == DomainType.code
    assert plan.subtasks[1].depends_on == ["t1"]


def test_parse_plan_with_code_block():
    raw = f"```json\n{json.dumps({'subtasks': [{'id': 't1', 'domain': 'general', 'prompt': 'do thing', 'depends_on': []}]})}\n```"
    plan = _parse_plan(raw, "test")
    assert plan is not None
    assert len(plan.subtasks) == 1


def test_parse_plan_invalid_json():
    plan = _parse_plan("not json at all", "test")
    assert plan is None


def test_parse_plan_missing_subtasks():
    plan = _parse_plan(json.dumps({"other": "data"}), "test")
    assert plan is None


def test_parse_plan_invalid_domain_falls_back_to_general():
    raw = json.dumps(
        {
            "subtasks": [
                {
                    "id": "t1",
                    "domain": "invalid_domain",
                    "prompt": "do thing",
                    "depends_on": [],
                },
            ]
        }
    )
    plan = _parse_plan(raw, "test")
    assert plan is not None
    assert plan.subtasks[0].domain == DomainType.general


def test_single_task_fallback():
    plan = _single_task_fallback("hello world")
    assert isinstance(plan, AnalyzerPlan)
    assert len(plan.subtasks) == 1
    assert plan.subtasks[0].domain == DomainType.general
    assert plan.subtasks[0].prompt == "hello world"
    assert plan.original_prompt == "hello world"


@pytest.mark.asyncio
async def test_analyze_prompt_parse_fallback(monkeypatch):
    async def mock_acompletion(**kwargs):
        class MockResponse:
            class Choice:
                class Message:
                    content: str = "invalid json"

                def __init__(self):
                    self.message = self.Message()

            choices = [Choice()]

        return MockResponse()

    monkeypatch.setattr("litellm.acompletion", mock_acompletion)

    from polymind.core.analyzer import analyze_prompt

    plan = await analyze_prompt("hello")
    assert len(plan.subtasks) == 1
    assert plan.subtasks[0].domain == DomainType.general


@pytest.mark.asyncio
async def test_analyze_prompt_valid(monkeypatch):
    valid_json = json.dumps(
        {
            "subtasks": [
                {
                    "id": "t1",
                    "domain": "code",
                    "prompt": "write code",
                    "depends_on": [],
                },
            ]
        }
    )

    async def mock_acompletion(**kwargs):
        class MockResponse:
            class Choice:
                class Message:
                    content: str = valid_json

                def __init__(self):
                    self.message = self.Message()

            choices = [Choice()]

        return MockResponse()

    monkeypatch.setattr("litellm.acompletion", mock_acompletion)

    from polymind.core.analyzer import analyze_prompt

    plan = await analyze_prompt("write code")
    assert len(plan.subtasks) == 1
    assert plan.subtasks[0].domain == DomainType.code

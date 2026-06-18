import pytest

from polymind.core.executor import execute_subtask, execute_subtask_with_context
from polymind.core.types import DomainType, Subtask, SubtaskResult


@pytest.mark.asyncio
async def test_execute_subtask_success(monkeypatch):
    async def mock_acompletion(**kwargs):
        class MockResponse:
            class Choice:
                class Message:
                    content: str = "def fib(): pass"

                def __init__(self):
                    self.message = self.Message()

            choices = [Choice()]

        return MockResponse()

    monkeypatch.setattr("litellm.acompletion", mock_acompletion)

    subtask = Subtask(id="t1", domain=DomainType.code, prompt="write fib")
    result = await execute_subtask(subtask, max_retries=1)
    assert result.subtask_id == "t1"
    assert result.output == "def fib(): pass"
    assert result.error is None


@pytest.mark.asyncio
async def test_execute_subtask_failure(monkeypatch):
    async def mock_fail(**kwargs):
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr("litellm.acompletion", mock_fail)

    subtask = Subtask(id="t1", domain=DomainType.code, prompt="write fib")
    result = await execute_subtask(subtask, max_retries=1)
    assert result.error is not None
    assert "LLM unavailable" in result.error


@pytest.mark.asyncio
async def test_execute_subtask_with_context(monkeypatch):
    call_count = 0

    async def mock_acompletion(**kwargs):
        nonlocal call_count
        call_count += 1
        msgs = kwargs.get("messages", [])
        txt = msgs[-1]["content"] if msgs else ""

        class MockMessage:
            pass

        class MockChoice:
            pass

        class MockResponse:
            pass

        m = MockMessage()
        m.content = txt
        c = MockChoice()
        c.message = m
        r = MockResponse()
        r.choices = [c]
        return r

    monkeypatch.setattr("litellm.acompletion", mock_acompletion)

    prior_results = {
        "t1": SubtaskResult(subtask_id="t1", model="m1", output="base class"),
    }
    subtask = Subtask(
        id="t2",
        domain=DomainType.code,
        prompt="extend class",
        depends_on=["t1"],
    )
    result = await execute_subtask_with_context(subtask, prior_results)
    assert "base class" in result.output
    assert "extend class" in result.output

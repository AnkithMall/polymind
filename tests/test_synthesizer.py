import pytest

from polymind.core.synthesizer import (
    _build_synthesis_messages,
    _build_synthesis_messages_json,
    synthesize,
)
from polymind.core.types import PipelineResult, SubtaskResult


def test_build_synthesis_messages():
    results = [
        SubtaskResult(subtask_id="t1", model="m1", output="code output"),
        SubtaskResult(subtask_id="t2", model="m2", output="math output"),
    ]
    messages = _build_synthesis_messages("solve problem", results)
    assert len(messages) == 2
    assert "code output" in messages[1]["content"]
    assert "math output" in messages[1]["content"]
    assert "solve problem" in messages[1]["content"]


def test_build_synthesis_messages_with_error():
    results = [
        SubtaskResult(subtask_id="t1", model="m1", output="", error="timeout"),
    ]
    messages = _build_synthesis_messages("test", results)
    assert "ERROR: timeout" in messages[1]["content"]


def test_build_synthesis_messages_json():
    results = [
        SubtaskResult(subtask_id="t1", model="m1", output="hello"),
    ]
    messages = _build_synthesis_messages_json("test", results)
    assert messages[1]["content"]  # is valid json string


@pytest.mark.asyncio
async def test_synthesize_success(monkeypatch):
    async def mock_acompletion(**kwargs):
        class MockResponse:
            class Choice:
                class Message:
                    content: str = "synthesized response"

                def __init__(self):
                    self.message = self.Message()

            choices = [Choice()]

        return MockResponse()

    monkeypatch.setattr("litellm.acompletion", mock_acompletion)

    result = PipelineResult(
        original_prompt="test",
        subtask_results=[
            SubtaskResult(subtask_id="t1", model="m1", output="part 1"),
        ],
    )
    output = await synthesize(result, synthesizer_model="ollama/test")
    assert output.synthesis == "synthesized response"


@pytest.mark.asyncio
async def test_synthesize_failure(monkeypatch):
    async def mock_fail(**kwargs):
        raise RuntimeError("synthesis failed")

    monkeypatch.setattr("litellm.acompletion", mock_fail)

    result = PipelineResult(
        original_prompt="test",
        subtask_results=[
            SubtaskResult(subtask_id="t1", model="m1", output="part 1"),
        ],
    )
    output = await synthesize(result, synthesizer_model="ollama/test")
    assert output.synthesis is None


@pytest.mark.asyncio
async def test_synthesize_streaming(monkeypatch):
    async def mock_acompletion(**kwargs):
        async def _chunks():
            for text in ("hello ", "world"):

                class Chunk:
                    pass

                class Choice:
                    pass

                class Delta:
                    pass

                d = Delta()
                d.content = text
                c = Choice()
                c.delta = d
                ch = Chunk()
                ch.choices = [c]
                yield ch

        return _chunks()

    monkeypatch.setattr("litellm.acompletion", mock_acompletion)

    from polymind.core.synthesizer import synthesize_streaming

    result = PipelineResult(
        original_prompt="test",
        subtask_results=[
            SubtaskResult(subtask_id="t1", model="m1", output="part 1"),
        ],
    )
    chunks = []
    async for chunk in synthesize_streaming(result):
        chunks.append(chunk)
    assert "".join(chunks) == "hello world"
    assert result.synthesis == "hello world"

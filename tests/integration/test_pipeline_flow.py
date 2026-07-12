"""Integration: analyzer → scheduler → executor → synthesizer flow."""

import pytest

from polymind.core.types import (
    ALL_DOMAINS,
    AnalyzerPlan,
    DomainType,
    ExecutionStrategy,
    PipelineResult,
    RankEntry,
    RankStore,
    Subtask,
    SubtaskResult,
)


def test_plan_to_schedule_to_result():
    """Build a plan, schedule it, execute (mocked), and produce a PipelineResult."""
    plan = AnalyzerPlan(
        original_prompt="Solve math and write code",
        subtasks=[
            Subtask(id="t1", domain=DomainType.math, prompt="2+2", depends_on=[]),
            Subtask(id="t2", domain=DomainType.code, prompt="write a function", depends_on=["t1"]),
        ],
    )
    store = RankStore(entries=[
        RankEntry(model="math-model", domain=DomainType.math, score=0.9),
        RankEntry(model="code-model", domain=DomainType.code, score=0.8),
    ])

    from polymind.core.scheduler import build_schedule
    schedule = build_schedule(plan, rank_store=store, strategy=ExecutionStrategy.sequential)

    assert len(schedule.batches) == 2
    assert schedule.batches[0].model == "math-model"
    assert schedule.batches[1].model == "code-model"

    results = [
        SubtaskResult(subtask_id="t1", model="math-model", output="4", latency_ms=10),
        SubtaskResult(subtask_id="t2", model="code-model", output="def f(): pass", latency_ms=20),
    ]

    from polymind.core.synthesizer import _build_synthesis_messages
    messages = _build_synthesis_messages(plan.original_prompt, results)
    assert len(messages) > 0
    assert "4" in str(messages)
    assert "def f(): pass" in str(messages)

    result = PipelineResult(original_prompt=plan.original_prompt, subtask_results=results)
    assert result.subtask_results[0].output == "4"
    assert result.subtask_results[1].output == "def f(): pass"


@pytest.mark.asyncio
async def test_execute_subtask_and_collect(monkeypatch):
    """Execute a subtask via mocked LLM and verify the result fields."""
    async def mock_acompletion(*_args, **_kwargs):
        class M: content = "mocked output"
        class C: message = M()
        class R: choices = [C()]; usage = None
        return R()

    monkeypatch.setattr("litellm.acompletion", mock_acompletion)

    from polymind.core.executor import execute_subtask
    subtask = Subtask(id="t1", domain=DomainType.code, prompt="write code", depends_on=[])
    result = await execute_subtask(subtask, model_ref="ollama/test", max_retries=1)
    assert result.output == "mocked output"
    assert result.model == "ollama/test"
    assert result.latency_ms > 0


def test_schedule_from_plan_integrity():
    """Total task count in schedule matches plan."""
    plan = AnalyzerPlan(
        original_prompt="multi step",
        subtasks=[
            Subtask(id="t1", domain=DomainType.code, prompt="a", depends_on=[]),
            Subtask(id="t2", domain=DomainType.math, prompt="b", depends_on=["t1"]),
            Subtask(id="t3", domain=DomainType.code, prompt="c", depends_on=[]),
        ],
    )
    from polymind.core.scheduler import build_schedule
    schedule = build_schedule(plan)
    total = sum(len(b.subtask_ids) for b in schedule.batches)
    assert total == len(plan.subtasks)

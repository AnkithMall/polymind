import json
import tempfile
from pathlib import Path

import pytest
import yaml

from polymind.core.benchmark import (
    BUILTIN_TASKS,
    BenchmarkTask,
    exact_match_score,
    get_tasks_for_domain,
    load_ranks,
    llm_judge_score,
    run_benchmark,
    save_ranks,
    score_task,
)
from polymind.core.types import ALL_DOMAINS, DomainType, RankEntry, RankStore


def test_builtin_tasks_all_domains():
    for domain in ALL_DOMAINS:
        tasks = BUILTIN_TASKS.get(domain, [])
        assert len(tasks) == 5, f"Domain {domain} has {len(tasks)} tasks, expected 5"


def test_get_tasks_for_domain():
    tasks = get_tasks_for_domain(DomainType.code)
    assert len(tasks) == 5
    assert all(t.domain == DomainType.code for t in tasks)


def test_get_tasks_for_domain_unknown():
    from polymind.core.types import DomainType as DT

    tasks = get_tasks_for_domain(DT.creative)
    assert len(tasks) >= 5


def test_exact_match_score_positive():
    assert exact_match_score("The answer is 42", "42") == 1.0


def test_exact_match_score_case_insensitive():
    assert exact_match_score("paris is beautiful", "Paris") == 1.0


def test_exact_match_score_negative():
    assert exact_match_score("The answer is 43", "42") == 0.0


def test_exact_match_score_substring():
    assert exact_match_score("def fibonacci(n): pass", "def fibonacci") == 1.0


@pytest.mark.asyncio
async def test_llm_judge_score(monkeypatch):
    async def mock_acompletion(**kwargs):
        class MockMessage:
            pass

        class MockChoice:
            pass

        class MockResponse:
            pass

        m = MockMessage()
        m.content = "0.85"
        c = MockChoice()
        c.message = m
        r = MockResponse()
        r.choices = [c]
        return r

    monkeypatch.setattr("litellm.acompletion", mock_acompletion)

    score = await llm_judge_score(
        "good output",
        "expected phrase",
        "test prompt",
        judge_model="ollama/test",
    )
    assert score == 0.85


@pytest.mark.asyncio
async def test_llm_judge_score_fallback(monkeypatch):
    async def mock_fail(**kwargs):
        raise RuntimeError("judge unavailable")

    monkeypatch.setattr("litellm.acompletion", mock_fail)

    score = await llm_judge_score(
        "output",
        "expected",
        "prompt",
        judge_model="ollama/test",
    )
    assert score == 0.0


@pytest.mark.asyncio
async def test_score_task_exact_match():
    task = BenchmarkTask(DomainType.math, "2+2?", "4", "exact_match")
    score = await score_task("the answer is 4", task, "2+2?")
    assert score == 1.0


@pytest.mark.asyncio
async def test_score_task_llm_judge(monkeypatch):
    async def mock_acompletion(**kwargs):
        class MockMessage:
            pass

        class MockChoice:
            pass

        class MockResponse:
            pass

        m = MockMessage()
        m.content = "0.9"
        c = MockChoice()
        c.message = m
        r = MockResponse()
        r.choices = [c]
        return r

    monkeypatch.setattr("litellm.acompletion", mock_acompletion)

    task = BenchmarkTask(DomainType.creative, "write a poem", "moon", "llm_judge")
    score = await score_task("moonlight sonata", task, "write a poem")
    assert score == 0.9


def test_save_and_load_ranks(tmp_path: Path):
    store = RankStore(
        entries=[
            RankEntry(model="m1", domain=DomainType.code, score=0.95),
        ]
    )
    path = tmp_path / "ranks.yaml"
    save_ranks(store, path)
    assert path.exists()

    loaded = load_ranks(path)
    assert len(loaded.entries) == 1
    assert loaded.entries[0].model == "m1"
    assert loaded.entries[0].score == 0.95


def test_load_ranks_non_existent(tmp_path: Path):
    store = load_ranks(tmp_path / "nonexistent.yaml")
    assert len(store.entries) == 0


def test_load_ranks_empty_file(tmp_path: Path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    store = load_ranks(path)
    assert len(store.entries) == 0


def test_rankstore_staleness():
    from datetime import datetime, timedelta

    fresh = RankStore(
        entries=[
            RankEntry(
                model="m1", domain=DomainType.code, score=0.9, timestamp=datetime.now()
            ),
        ]
    )
    assert not fresh.is_stale(max_age_days=30)

    stale = RankStore(
        entries=[
            RankEntry(
                model="m1",
                domain=DomainType.code,
                score=0.9,
                timestamp=datetime.now() - timedelta(days=31),
            ),
        ]
    )
    assert stale.is_stale(max_age_days=30)


@pytest.mark.asyncio
async def test_run_benchmark(monkeypatch):
    call_count = [0]
    mock_responses = [
        "110",
        "150",
        "12",
        "5",
        "30",
        "0.85",
        "0.85",
        "0.85",
        "0.85",
        "0.85",
    ]

    async def mock_acompletion(**kwargs):
        resp = mock_responses[call_count[0] % len(mock_responses)]
        call_count[0] += 1

        class MockMessage:
            pass

        class MockChoice:
            pass

        class MockResponse:
            pass

        m = MockMessage()
        m.content = resp
        c = MockChoice()
        c.message = m
        r = MockResponse()
        r.choices = [c]
        return r

    monkeypatch.setattr("litellm.acompletion", mock_acompletion)

    store = await run_benchmark(
        models=["ollama/test-model"],
        domains=[DomainType.math],
        judge_model="ollama/test-judge",
    )
    assert len(store.entries) > 0
    assert store.entries[0].score > 0
    assert store.entries[0].domain == DomainType.math
    assert store.entries[0].model == "ollama/test-model"

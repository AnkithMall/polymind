from polymind.core.types import (
    ALL_DOMAINS,
    AnalyzerPlan,
    DomainType,
    ExecutionSchedule,
    ExecutionStrategy,
    ModelBatch,
    PipelineResult,
    RankEntry,
    RankStore,
    Subtask,
    SubtaskResult,
)


def test_subtask_creation():
    s = Subtask(id="t1", domain=DomainType.code, prompt="write a function")
    assert s.id == "t1"
    assert s.domain == DomainType.code
    assert s.depends_on == []


def test_analyzer_plan():
    s1 = Subtask(id="t1", domain=DomainType.code, prompt="write a function")
    s2 = Subtask(
        id="t2", domain=DomainType.math, prompt="solve equation", depends_on=["t1"]
    )
    plan = AnalyzerPlan(original_prompt="do math and code", subtasks=[s1, s2])
    assert len(plan.subtasks) == 2


def test_rank_store_top_for_domain():
    store = RankStore(
        entries=[
            RankEntry(model="model-a", domain=DomainType.code, score=0.9),
            RankEntry(model="model-b", domain=DomainType.code, score=0.8),
            RankEntry(model="model-c", domain=DomainType.math, score=0.95),
        ]
    )
    top = store.top_for_domain(DomainType.code)
    assert top is not None
    assert top.model == "model-a"
    assert top.score == 0.9

    top_math = store.top_for_domain(DomainType.math)
    assert top_math is not None
    assert top_math.model == "model-c"


def test_rank_store_empty():
    store = RankStore()
    assert store.top_for_domain(DomainType.code) is None
    assert store.is_stale() is True


def test_rank_store_top_n():
    store = RankStore(
        entries=[
            RankEntry(model="a", domain=DomainType.code, score=0.9),
            RankEntry(model="b", domain=DomainType.code, score=0.8),
            RankEntry(model="c", domain=DomainType.code, score=0.7),
        ]
    )
    top2 = store.top_n_for_domain(DomainType.code, 2)
    assert len(top2) == 2
    assert [e.model for e in top2] == ["a", "b"]


def test_execution_schedule():
    batches = [
        ModelBatch(model="m1", subtask_ids=["t1", "t3"]),
        ModelBatch(model="m2", subtask_ids=["t2"]),
    ]
    sched = ExecutionSchedule(strategy=ExecutionStrategy.model_aware, batches=batches)
    assert len(sched.batches) == 2


def test_pipeline_result():
    result = PipelineResult(
        original_prompt="test",
        subtask_results=[
            SubtaskResult(subtask_id="t1", model="m1", output="hello"),
        ],
    )
    assert result.synthesis is None
    assert result.subtask_results[0].output == "hello"


def test_all_domains():
    assert len(ALL_DOMAINS) == 9
    assert DomainType.code in ALL_DOMAINS


# ── Ranking mode helpers ─────────────────────────────────────────────

def test_cost_effective_ranking():
    from polymind.core.types import RankingMode, _rank_key
    cheap = RankEntry(model="cheap", domain=DomainType.code, score=0.6, cost=0.0001)
    expensive = RankEntry(model="expensive", domain=DomainType.code, score=0.9, cost=0.01)
    assert _rank_key(cheap, RankingMode.cost_effective) > _rank_key(expensive, RankingMode.cost_effective)


def test_cost_ranking():
    from polymind.core.types import RankingMode, _rank_key
    cheap = RankEntry(model="cheap", domain=DomainType.code, score=0.6, cost=0.0001)
    expensive = RankEntry(model="expensive", domain=DomainType.code, score=0.9, cost=0.01)
    assert _rank_key(cheap, RankingMode.cost) > _rank_key(expensive, RankingMode.cost)

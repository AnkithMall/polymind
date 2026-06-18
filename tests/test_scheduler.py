from polymind.core.scheduler import (
    _assign_models,
    _model_aware_batches,
    _parallel_batches,
    _sequential_batches,
    _topological_batches,
    build_schedule,
    count_model_loads,
)
from polymind.core.types import (
    AnalyzerPlan,
    DomainType,
    ExecutionStrategy,
    RankEntry,
    RankStore,
    Subtask,
)


def _make_plan(subtasks: list[tuple[str, DomainType, list[str]]]) -> AnalyzerPlan:
    return AnalyzerPlan(
        original_prompt="test",
        subtasks=[
            Subtask(id=id_, domain=domain, prompt=f"do {domain.value}", depends_on=deps)
            for id_, domain, deps in subtasks
        ],
    )


def test_assign_models_from_rank_store():
    plan = _make_plan([("t1", DomainType.code, [])])
    store = RankStore(
        entries=[
            RankEntry(model="code-expert", domain=DomainType.code, score=0.95),
        ]
    )
    _assign_models(
        plan, store, default_model="ollama/llama3.2:1b", fallback_models=None
    )
    assert plan.subtasks[0].assigned_model == "code-expert"


def test_assign_models_fallback_to_default():
    plan = _make_plan([("t1", DomainType.code, [])])
    store = RankStore(entries=[])
    _assign_models(plan, store, default_model="ollama/default", fallback_models=None)
    assert plan.subtasks[0].assigned_model == "ollama/default"


def test_assign_models_without_rank_store():
    plan = _make_plan([("t1", DomainType.code, [])])
    _assign_models(plan, None, default_model="ollama/default", fallback_models=None)
    assert plan.subtasks[0].assigned_model == "ollama/default"


def test_topological_batches_simple():
    plan = _make_plan(
        [
            ("t1", DomainType.code, []),
            ("t2", DomainType.math, ["t1"]),
        ]
    )
    batches = _topological_batches(plan)
    assert len(batches) == 2
    assert batches[0][0].id == "t1"
    assert batches[1][0].id == "t2"


def test_topological_batches_complex():
    plan = _make_plan(
        [
            ("t1", DomainType.code, []),
            ("t2", DomainType.math, ["t1"]),
            ("t3", DomainType.reasoning, []),
            ("t4", DomainType.code, ["t1"]),
            ("t5", DomainType.code, ["t2", "t3"]),
        ]
    )
    batches = _topological_batches(plan)
    ordered = [b[0].id for b in batches]
    assert ordered.index("t1") < ordered.index("t2")
    assert ordered.index("t1") < ordered.index("t4")
    assert ordered.index("t2") < ordered.index("t5")
    assert ordered.index("t3") < ordered.index("t5")


def test_topological_batches_cycle_does_not_crash():
    plan = _make_plan(
        [
            ("t1", DomainType.code, ["t2"]),
            ("t2", DomainType.math, ["t1"]),
        ]
    )
    batches = _topological_batches(plan)
    assert len(batches) == 2


def test_sequential_batches():
    plan = _make_plan(
        [
            ("t1", DomainType.code, []),
            ("t2", DomainType.math, ["t1"]),
        ]
    )
    store = RankStore(
        entries=[
            RankEntry(model="code-m", domain=DomainType.code, score=0.9),
            RankEntry(model="math-m", domain=DomainType.math, score=0.8),
        ]
    )
    _assign_models(plan, store, default_model="default", fallback_models=None)
    batches = _sequential_batches(plan)
    assert len(batches) == 2
    assert batches[0].model == "code-m"
    assert batches[1].model == "math-m"


def test_model_aware_batching():
    plan = _make_plan(
        [
            ("t1", DomainType.code, []),
            ("t2", DomainType.math, []),
            ("t3", DomainType.code, []),
            ("t4", DomainType.code, ["t3"]),
        ]
    )
    store = RankStore(
        entries=[
            RankEntry(model="model-a", domain=DomainType.code, score=0.9),
            RankEntry(model="model-b", domain=DomainType.math, score=0.8),
        ]
    )
    _assign_models(plan, store, default_model="default", fallback_models=None)
    batches = _model_aware_batches(plan)
    assert len(batches) >= 2

    for batch in batches:
        assert batch.model in ("model-a", "model-b")


def test_model_aware_reduces_loads():
    plan = _make_plan(
        [
            ("t1", DomainType.code, []),
            ("t2", DomainType.math, []),
            ("t3", DomainType.code, []),
            ("t4", DomainType.code, []),
        ]
    )
    store = RankStore(
        entries=[
            RankEntry(model="model-a", domain=DomainType.code, score=0.9),
            RankEntry(model="model-b", domain=DomainType.math, score=0.8),
        ]
    )
    _assign_models(plan, store, default_model="default", fallback_models=None)

    model_batches = _model_aware_batches(plan)
    seq_batches = _sequential_batches(plan)

    model_loads = count_model_loads(type("Schedule", (), {"batches": model_batches})())
    seq_loads = count_model_loads(type("Schedule", (), {"batches": seq_batches})())

    assert model_loads <= seq_loads


def test_worked_example_from_docs():
    plan = _make_plan(
        [
            ("t1", DomainType.code, []),
            ("t2", DomainType.math, ["t1"]),
            ("t3", DomainType.reasoning, []),
            ("t4", DomainType.code, []),
            ("t5", DomainType.code, ["t3"]),
            ("t6", DomainType.math, []),
        ]
    )
    store = RankStore(
        entries=[
            RankEntry(model="model-a", domain=DomainType.code, score=0.9),
            RankEntry(model="model-b", domain=DomainType.math, score=0.8),
            RankEntry(model="model-c", domain=DomainType.reasoning, score=0.85),
        ]
    )
    schedule = build_schedule(plan, store, ExecutionStrategy.model_aware)

    task_count = sum(len(b.subtask_ids) for b in schedule.batches)
    assert task_count == 6

    all_ids = []
    for b in schedule.batches:
        all_ids.extend(b.subtask_ids)
    assert sorted(all_ids) == ["t1", "t2", "t3", "t4", "t5", "t6"]

    assert schedule.batches[0].model == "model-a"


def test_parallel_batches():
    plan = _make_plan(
        [
            ("t1", DomainType.code, []),
            ("t2", DomainType.math, []),
            ("t3", DomainType.code, ["t1"]),
        ]
    )
    store = RankStore(
        entries=[
            RankEntry(model="model-a", domain=DomainType.code, score=0.9),
            RankEntry(model="model-b", domain=DomainType.math, score=0.8),
        ]
    )
    _assign_models(plan, store, default_model="default", fallback_models=None)
    batches = _parallel_batches(plan)

    first_batch_models = {b.model for b in batches[:2]}
    assert "model-a" in first_batch_models
    assert "model-b" in first_batch_models


def test_build_schedule_strategies():
    plan = _make_plan(
        [
            ("t1", DomainType.code, []),
            ("t2", DomainType.math, ["t1"]),
        ]
    )
    for strategy in ExecutionStrategy:
        schedule = build_schedule(plan, strategy=strategy)
        assert schedule.strategy == strategy
        assert len(schedule.batches) > 0


def test_count_model_loads():
    from polymind.core.types import ExecutionSchedule, ModelBatch

    schedule = ExecutionSchedule(
        strategy=ExecutionStrategy.model_aware,
        batches=[
            ModelBatch(model="a", subtask_ids=["t1", "t3"]),
            ModelBatch(model="b", subtask_ids=["t2"]),
            ModelBatch(model="a", subtask_ids=["t4"]),
        ],
    )
    assert count_model_loads(schedule) == 3


def test_count_model_loads_same_model():
    from polymind.core.types import ExecutionSchedule, ModelBatch

    schedule = ExecutionSchedule(
        strategy=ExecutionStrategy.model_aware,
        batches=[
            ModelBatch(model="a", subtask_ids=["t1"]),
            ModelBatch(model="a", subtask_ids=["t3"]),
            ModelBatch(model="a", subtask_ids=["t4"]),
        ],
    )
    assert count_model_loads(schedule) == 1

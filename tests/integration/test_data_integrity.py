"""Data-structure invariants across modules."""

from polymind.core.types import ALL_DOMAINS, AnalyzerPlan, DomainType, Subtask


def test_subtask_depends_on_references():
    """All depends_on values reference valid subtask IDs."""
    plan = AnalyzerPlan(
        original_prompt="test",
        subtasks=[
            Subtask(id="T1", domain=DomainType.code, prompt="a", depends_on=[]),
            Subtask(id="T2", domain=DomainType.math, prompt="b", depends_on=["T1"]),
            Subtask(id="T3", domain=DomainType.code, prompt="c", depends_on=["T1"]),
        ],
    )
    ids = {s.id for s in plan.subtasks}
    for s in plan.subtasks:
        for dep in s.depends_on:
            assert dep in ids, f"{s.id} depends on unknown {dep}"


def test_execution_schedule_total_tasks():
    """Subtask IDs in batches equal plan subtask count."""
    plan = AnalyzerPlan(
        original_prompt="test",
        subtasks=[
            Subtask(id="T1", domain=DomainType.code, prompt="a", depends_on=[]),
            Subtask(id="T2", domain=DomainType.math, prompt="b", depends_on=["T1"]),
            Subtask(id="T3", domain=DomainType.code, prompt="c", depends_on=[]),
        ],
    )
    from polymind.core.scheduler import build_schedule
    schedule = build_schedule(plan)
    total = sum(len(b.subtask_ids) for b in schedule.batches)
    assert total == len(plan.subtasks)


def test_all_domains_have_tasks():
    """Every domain has at least one benchmark task defined."""
    from polymind.core.benchmark import get_tasks_for_domain
    for d in ALL_DOMAINS:
        tasks = get_tasks_for_domain(d)
        assert len(tasks) > 0, f"Domain {d} has no benchmark tasks"

"""Integration: benchmark → ranks save/load → RankStore queries."""

import pytest

from polymind.core.types import ALL_DOMAINS, DomainType, RankEntry, RankStore


def test_benchmark_ranks_round_trip(tmp_path):
    """Simulate benchmark output → save → load → query."""
    entries = [
        RankEntry(model="m1", domain=DomainType.code, score=0.95, latency_ms=100, cost=0.0),
        RankEntry(model="m1", domain=DomainType.math, score=0.80, latency_ms=150, cost=0.0),
        RankEntry(model="m2", domain=DomainType.code, score=0.90, latency_ms=200, cost=0.0),
    ]
    store = RankStore(entries=entries)

    from polymind.core.benchmark import save_ranks, load_ranks
    path = tmp_path / "ranks.yaml"
    save_ranks(store, path)

    loaded = load_ranks(path)
    assert len(loaded.entries) == 3

    top_code = loaded.top_for_domain(DomainType.code)
    assert top_code is not None
    assert top_code.model == "m1"
    assert top_code.score == 0.95

    top2 = loaded.top_n_for_domain(DomainType.code, 2)
    assert len(top2) == 2
    assert top2[0].model == "m1"
    assert top2[1].model == "m2"


def test_benchmark_ranks_score_range():
    """All rank entries maintain 0.0 <= score <= 1.0."""
    entries = [
        RankEntry(model="m1", domain=DomainType.code, score=0.5),
        RankEntry(model="m2", domain=DomainType.code, score=1.0),
        RankEntry(model="m3", domain=DomainType.code, score=0.0),
    ]
    for e in entries:
        assert 0.0 <= e.score <= 1.0


def test_benchmark_ranks_top_for_domain_empty():
    store = RankStore()
    assert store.top_for_domain(DomainType.code) is None

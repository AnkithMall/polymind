from polymind.core.context import (
    DEFAULT_CONTEXT_LIMITS,
    ContextBudget,
    estimate_tokens,
    get_model_context_limit,
    truncate_to_fit,
)


def test_estimate_tokens():
    tokens = estimate_tokens("hello world")
    assert tokens > 0
    assert isinstance(tokens, int)


def test_truncate_to_fit_within_limit():
    text = "short text"
    result = truncate_to_fit(text, 1000)
    assert result == text


def test_truncate_to_fit_exceeds():
    text = "hello world " * 500
    result = truncate_to_fit(text, 10)
    assert result.endswith("[truncated...]")
    assert len(result) < len(text)


def test_context_budget_can_fit():
    budget = ContextBudget(model="test", max_context_tokens=4096)
    assert budget.can_fit("short text")
    assert not budget.can_fit("x" * 50000)


def test_context_budget_add_usage():
    budget = ContextBudget(model="test", max_context_tokens=4096)
    initial = budget.used_tokens
    budget.add_usage("some text here")
    assert budget.used_tokens > initial


def test_context_budget_fit_or_truncate():
    budget = ContextBudget(
        model="test", max_context_tokens=100, reserved_output_tokens=10
    )
    long_text = "hello " * 500
    result = budget.fit_or_truncate(long_text, label="long content")
    assert len(budget.warnings) == 1
    assert "truncated" in budget.warnings[0]
    assert result.endswith("[truncated...]")


def test_get_model_context_limit():
    limit = get_model_context_limit("ollama/llama3.2:1b")
    assert limit == 8192


def test_get_model_context_limit_gpt4():
    limit = get_model_context_limit("gpt-4o")
    assert limit == 128000


def test_get_model_context_limit_unknown():
    limit = get_model_context_limit("unknown-model-xyz")
    assert limit == 4096


def test_get_model_context_limit_claude():
    limit = get_model_context_limit("claude-3-opus")
    assert limit == 100000


def test_context_budget_available():
    budget = ContextBudget(
        model="test", max_context_tokens=4096, reserved_output_tokens=1024
    )
    assert budget.available_tokens == 4096 - 1024

import asyncio

import pytest

from polymind.core.fallback import (
    FallbackError,
    fallback_chain,
    retry_with_backoff,
    sync_retry_with_backoff,
)


@pytest.mark.asyncio
async def test_retry_success():
    call_count = 0

    async def flaky_fn(x: int) -> int:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("not yet")
        return x * 2

    result = await retry_with_backoff(flaky_fn, 5, max_retries=3, base_delay_s=0.01)
    assert result == 10
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_exhausted():
    async def always_fails() -> str:
        raise RuntimeError("always fails")

    with pytest.raises(RuntimeError):
        await retry_with_backoff(always_fails, max_retries=2, base_delay_s=0.01)


@pytest.mark.asyncio
async def test_fallback_chain_first_succeeds():
    async def fn1() -> str:
        return "from fn1"

    async def fn2() -> str:
        return "from fn2"

    result = await fallback_chain([fn1, fn2])
    assert result == "from fn1"


@pytest.mark.asyncio
async def test_fallback_chain_second_succeeds():
    async def fn1() -> str:
        raise ValueError("fn1 failed")

    async def fn2() -> str:
        return "from fn2"

    result = await fallback_chain([fn1, fn2])
    assert result == "from fn2"


@pytest.mark.asyncio
async def test_fallback_chain_all_fail():
    async def fn1() -> str:
        raise ValueError("fn1 failed")

    async def fn2() -> str:
        raise RuntimeError("fn2 failed")

    with pytest.raises(FallbackError) as exc_info:
        await fallback_chain([fn1, fn2])
    assert len(exc_info.value.errors) == 2


def test_sync_retry():
    call_count = 0

    def flaky_fn(x: int) -> int:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("not yet")
        return x * 2

    result = sync_retry_with_backoff(flaky_fn, 5, max_retries=3, base_delay_s=0.01)
    assert result == 10


@pytest.mark.asyncio
async def test_fallback_timeout():
    async def slow_fn() -> str:
        await asyncio.sleep(10)
        return "too late"

    async def fast_fn() -> str:
        return "fast"

    result = await fallback_chain([slow_fn, fast_fn], timeout_s=0.1)
    assert result == "fast"

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class FallbackError(Exception):
    def __init__(self, message: str, errors: list[Exception]) -> None:
        self.errors = errors
        super().__init__(f"{message} (attempts: {len(errors)})")


async def retry_with_backoff(
    fn: Callable[..., Coroutine[Any, Any, T]],
    *args: Any,
    max_retries: int = 3,
    base_delay_s: float = 1.0,
    max_delay_s: float = 30.0,
    **kwargs: Any,
) -> T:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = min(base_delay_s * (2**attempt), max_delay_s)
                logger.warning(
                    "Attempt %d/%d failed: %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)
    raise last_error  # type: ignore[misc]


async def fallback_chain(
    callables: list[Callable[[], Coroutine[Any, Any, T]]],
    timeout_s: float | None = None,
) -> T:
    errors: list[Exception] = []
    for i, call in enumerate(callables):
        try:
            if timeout_s is not None:
                return await asyncio.wait_for(call(), timeout=timeout_s)
            return await call()
        except Exception as e:
            logger.warning("Fallback %d/%d failed: %s", i + 1, len(callables), e)
            errors.append(e)

    raise FallbackError(
        f"All {len(callables)} fallback attempts failed",
        errors=errors,
    )


def sync_retry_with_backoff(
    fn: Callable[..., T],
    *args: Any,
    max_retries: int = 3,
    base_delay_s: float = 1.0,
    max_delay_s: float = 30.0,
    **kwargs: Any,
) -> T:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = min(base_delay_s * (2**attempt), max_delay_s)
                logger.warning(
                    "Attempt %d/%d failed: %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    e,
                    delay,
                )
                time.sleep(delay)
    raise last_error  # type: ignore[misc]

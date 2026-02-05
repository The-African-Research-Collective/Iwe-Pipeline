"""
Retry and backoff utilities for handling transient failures.
"""

import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def exponential_backoff(
    func: Callable[..., T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,),
) -> Callable[..., T]:
    """
    Decorator for exponential backoff retry logic.

    Args:
        func: Function to wrap
        max_retries: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exceptions: Tuple of exceptions to catch

    Returns:
        Wrapped function with retry logic
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if attempt == max_retries:
                    logger.error(f"Failed after {max_retries} retries: {e}")
                    raise

                delay = min(base_delay * (2**attempt), max_delay)
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. " f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

    return wrapper

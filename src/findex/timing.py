"""Small logging-based timing helper."""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import wraps
from time import perf_counter
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")
logger = logging.getLogger(__name__)


def timed[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            logger.debug(
                "%s completed in %.6f s", func.__name__, perf_counter() - start
            )

    return wrapper

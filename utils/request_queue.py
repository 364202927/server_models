"""单用户 FIFO 请求队列。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class RequestQueue:
    """用一个异步锁保证模型生成和切换严格串行。"""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._pending = 0

    @property
    def length(self) -> int:
        return self._pending

    async def run(self, operation: Callable[[], Awaitable[T]]) -> T:
        self._pending += 1
        try:
            async with self._lock:
                return await operation()
        finally:
            self._pending -= 1

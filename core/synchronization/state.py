from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


class AtomicState(Generic[T]):
    def __init__(self, initial: T | None = None) -> None:
        self._value = initial
        self._lock = asyncio.Lock()

    @property
    def value(self) -> T | None:
        return self._value

    async def set(self, value: T) -> None:
        async with self._lock:
            self._value = value

    async def get(self) -> T | None:
        async with self._lock:
            return self._value


@dataclass(slots=True)
class SequenceGuard:
    last_seq: int = -1

    def accept(self, seq: int) -> bool:
        if seq <= self.last_seq:
            return False
        self.last_seq = seq
        return True


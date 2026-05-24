from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


class RingBuffer(Generic[T]):
    __slots__ = ("_data", "_capacity", "_idx", "_size")

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._data: list[T | None] = [None] * capacity
        self._idx = 0
        self._size = 0

    def append(self, item: T) -> None:
        self._data[self._idx] = item
        self._idx = (self._idx + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def latest(self) -> T | None:
        if self._size == 0:
            return None
        return self._data[(self._idx - 1) % self._capacity]

    def clear(self) -> None:
        self._data = [None] * self._capacity
        self._idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterable[T]:
        start = (self._idx - self._size) % self._capacity
        for i in range(self._size):
            item = self._data[(start + i) % self._capacity]
            if item is not None:
                yield item


@dataclass(slots=True)
class EWMoment:
    halflife_s: float
    mean: float = 0.0
    var: float = 1e-12
    last_ts_ns: int = 0
    initialized: bool = False

    def update(self, x: float, ts_ns: int) -> tuple[float, float]:
        if not self.initialized:
            self.mean = x
            self.var = 1e-12
            self.last_ts_ns = ts_ns
            self.initialized = True
            return self.mean, self.var
        dt = max(0.0, (ts_ns - self.last_ts_ns) * 1e-9)
        alpha = 1.0 - 2.0 ** (-dt / max(1e-9, self.halflife_s))
        delta = x - self.mean
        self.mean += alpha * delta
        self.var = max(1e-12, (1.0 - alpha) * (self.var + alpha * delta * delta))
        self.last_ts_ns = ts_ns
        return self.mean, self.var

    @property
    def std(self) -> float:
        return self.var**0.5


@dataclass(slots=True)
class EWCorrelation:
    dim: int
    halflife_s: float
    mean: list[float] = field(default_factory=list)
    cov: list[list[float]] = field(default_factory=list)
    last_ts_ns: int = 0

    def __post_init__(self) -> None:
        self.mean = [0.0] * self.dim
        self.cov = [[1e-6 if i == j else 0.0 for j in range(self.dim)] for i in range(self.dim)]

    def update(self, x: list[float], ts_ns: int) -> list[list[float]]:
        if len(x) != self.dim:
            raise ValueError("dimension mismatch")
        if self.last_ts_ns == 0:
            self.mean = list(x)
            self.last_ts_ns = ts_ns
            return self.cov
        dt = max(0.0, (ts_ns - self.last_ts_ns) * 1e-9)
        alpha = 1.0 - 2.0 ** (-dt / max(1e-9, self.halflife_s))
        one_minus_alpha = 1.0 - alpha
        old_mean = self.mean[:]
        for i in range(self.dim):
            self.mean[i] += alpha * (x[i] - old_mean[i])
        for i in range(self.dim):
            di = x[i] - old_mean[i]
            row = self.cov[i]
            for j in range(self.dim):
                dj = x[j] - old_mean[j]
                row[j] = one_minus_alpha * (row[j] + alpha * di * dj)
            if row[i] < 1e-6:
                row[i] = 1e-6
        self.last_ts_ns = ts_ns
        return self.cov


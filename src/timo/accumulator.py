from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax import Array

from contextlib import contextmanager
from timo.sum_count import SumCount
from flax import nnx


class Accumulator(nnx.Pytree):
    def __init__(self, keys: set[str]):
        super().__init__()
        for key in keys:
            setattr(self, key, nnx.data(SumCount()))
        self.keys = nnx.static(keys)

    def __getitem__(self, key):
        return getattr(self, key)

    def add_value(self, value: int | float | Array | None, key: str, weight: float | Array = 1):
        if value is None:
            return
        key_sum_count: SumCount = getattr(self, key)
        key_sum_count.add_value(value, weight)
        return self

    def add_sum_count(self, sum_count: SumCount, key, weight: float | Array = 1):
        key_sum_count: SumCount = getattr(self, key)
        key_sum_count.add_sum_count(sum_count, weight=weight)
        return self

    def add_accumulator(self, accumulator: Accumulator, weight: float | Array = 1):
        for key in accumulator.keys:
            self.add_sum_count(getattr(accumulator, key), key, weight=weight)
        return self

    def mean(self):
        sum_count = SumCount()
        for key in self.keys:
            sum_count.add_sum_count(getattr(self, key))
        return sum_count.mean()

    def sum(self):
        sum_count = SumCount()
        for key in self.keys:
            sum_count.add_sum_count(getattr(self, key))
        return sum_count.sum

    def means(self):
        for key in self.keys:
            yield key, getattr(self, key).mean()

    def sums(self):
        for key in self.keys:
            yield key, float(getattr(self, key).sum)  # type: ignore

    def detached_mean(self):
        sum_count = SumCount()
        for key in self.keys:
            sum_count.detached_add_sum_count(getattr(self, key))
        return sum_count.mean()

    @contextmanager
    def timer(self, key: str):
        from time import monotonic

        start = monotonic()
        try:
            yield
        finally:
            end = monotonic()
            self.add_value(end - start, key)

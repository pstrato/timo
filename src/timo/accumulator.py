from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax import Array

from contextlib import contextmanager
from timo.sum_count import SumCount
from flax import nnx


class Accumulator(nnx.Module):
    def __init__(self, sum_counts: nnx.Dict | None = None):
        super().__init__()
        self.sum_counts: nnx.Dict[SumCount] = nnx.Dict(sum_counts or {})

    def __getitem__(self, key):
        return self.sum_counts[key]

    def add_value(self, value: int | float | Array | None, key: str, weight: float | Array = 1):
        if value is None:
            return
        try:
            key_sum_count = self.sum_counts.get(key)
        except AttributeError:
            key_sum_count = None

        if key_sum_count is None:
            self.sum_counts[key] = SumCount().add_value(value, weight)
        else:
            key_sum_count.add_value(value)
        return self

    def add_sum_count(self, sum_count: SumCount, key, weight: float | Array = 1):
        try:
            key_sum_count = self.sum_counts.get(key)
        except AttributeError:
            key_sum_count = None

        if key_sum_count is None:
            self.sum_counts[key] = SumCount().add_sum_count(sum_count, weight=weight)
        else:
            key_sum_count.add_sum_count(sum_count, weight=weight)
        return self

    def add_accumulator(self, accumulator: Accumulator, weight: float | Array = 1):
        for key, key_sum_count in accumulator.sum_counts.items():
            self.add_sum_count(key_sum_count, key, weight=weight)
        return self

    def mean(self):
        sum_count = SumCount()
        for key, key_sum_count in self.sum_counts.items():
            sum_count.add_sum_count(key_sum_count)
        return sum_count.mean()

    def sum(self):
        sum_count = SumCount()
        for key, key_sum_count in self.sum_counts.items():
            sum_count.add_sum_count(key_sum_count)
        return sum_count.sum

    def means(self):
        for key, key_sum_count in self.sum_counts.items():
            yield key, key_sum_count.mean()

    def sums(self):
        for key, key_sum_count in self.sum_counts.items():
            yield key, float(key_sum_count.sum)  # type: ignore

    def detached_mean(self):
        sum_count = SumCount()
        for key, key_sum_count in self.sum_counts.items():
            sum_count.detached_add_sum_count(key_sum_count)
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

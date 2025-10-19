from __future__ import annotations
from itertools import count
from typing import TYPE_CHECKING

from jax.lax import stop_gradient

if TYPE_CHECKING:
    from jax import Array

from flax import nnx


class SumCount(nnx.Module):
    def __init__(self, sum: Array | None = None, count: int | None = None, weight: float = 1):
        super().__init__()
        self.sum = nnx.Variable((sum or 0) * weight)
        self.count = nnx.Variable(count or 0)

    def add_value(self, value: int | float | Array, weight: float | Array = 1):
        if isinstance(value, int) or isinstance(value, float):
            self.sum.value += value
            self.count.value += 1
        else:
            self.sum.value += value.sum() * weight
            self.count.value += value.size
        return self

    def detached_add_value(self, value: Array, weight: float | Array = 1):
        self.sum.value += stop_gradient(value).sum() * weight
        self.count.value += value.size
        return self

    def add_sum_count(self, sum_count: SumCount, weight: float | Array = 1):
        self.sum.value += sum_count.sum * weight
        self.count.value += sum_count.count
        return self

    def detached_add_sum_count(self, sum_count: SumCount, weight: float | Array = 1):
        self.sum.value += stop_gradient(sum_count.sum) * weight
        self.count.value += sum_count.count
        return self

    def mean(self):
        return (self.sum / self.count) if self.count.value > 0 else float("nan")

    def detached_mean(self):
        return (stop_gradient(self.sum) / self.count) if self.count.value > 0 else float("nan")

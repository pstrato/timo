from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax import Array


class Out:
    def __init__(self):
        self._losses: dict[tuple, Array] = {}

    def add_loss(self, loss: Array, **keys):
        keys = tuple(keys.items())
        self._losses[keys] = self._losses.get(keys, 0) + loss

    def clear(self):
        self._losses.clear()

    def loss_sum(self):
        return sum(self._losses.values())

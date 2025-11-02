from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.context import Context

from jax import Array
from jax.lax import stop_gradient
from timo.factory import Factory
from timo.transform import Transform


class StopGradient(Factory[Array, Array]):
    def create_transform(self, ctx: Context):
        return Transform[Array, Array](transform, ctx)


def transform(input: Array):
    return stop_gradient(input)

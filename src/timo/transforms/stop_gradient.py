from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.context import Context
    from timo.info import Info
    from timo.out import Out

from jax import Array
from timo.factory import Factory
from timo.transform import Transform
from jax.lax import stop_gradient


class StopGradient(Factory):
    def create_transform(self, ctx: Context):
        return ctx.input_shapes, Transform[Array, Array](transform)


def transform(input: Array, info: Info, out: Out):
    return stop_gradient(input)

from __future__ import annotations
from timo.factory import Factory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.context import Context

from jax import Array
from timo.transform import Transform


class Id(Factory):

    def create_module(self, ctx: Context):
        return ctx.input_shapes, Transform[Array, Array](id)


def id(inputs, info, out):
    return inputs

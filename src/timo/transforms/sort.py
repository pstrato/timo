from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.named_shape import After, Before
    from timo.context import Context

from timo.named_axis import NamedAxisField
from timo.factory import Factory
from timo.transform import Transform
from jax import Array
from jax import numpy as jnp


class Sort(Factory[Array, Array]):
    on: NamedAxisField

    def create_transform(self, ctx: Context):
        input_shape = ctx.input_shapes.single_shape()

        transform = sort
        transform = self.vmap(transform, tuple(), self.on)
        return Transform[Array, Array](sort, ctx, self, input_shape)


def sort(inputs: Array):
    sorted = jnp.argsort(inputs)
    return inputs[sorted]

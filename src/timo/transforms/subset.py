from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.context import Context

from timo.named_axis import NamedAxisField
from jax import Array
from timo.factory import Factory
from timo.transform import Transform
from jax import numpy as jnp


class Subset(Factory[Array, Array]):
    on: NamedAxisField
    take: tuple[int, ...]

    def create_transform(self, ctx: Context):
        from timo.sized_named_axis import size

        input_shape = ctx.input_shapes.single_shape()
        axis = input_shape.indexof(self.on)
        index = self.take
        output_shape = input_shape.resize(size(self.on, len(self.take)))

        return Transform[Array, Array](subset, ctx, self, output_shape, static={"axis": axis, "indices": self.take})


def subset(inputs: Array, axis: int, indices: tuple[int]):
    return jnp.take(inputs, jnp.array(indices), axis)

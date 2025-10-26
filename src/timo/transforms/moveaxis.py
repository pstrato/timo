from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.named_shape import After, Before
    from timo.context import Context
    from timo.out import Out

from timo.named_axis import NamedAxisField
from timo.factory import Factory
from timo.transform import Transform
from jax import Array
from jax import numpy as jnp
from flax import nnx


class MoveAxis(Factory[Array, Array]):
    axis: NamedAxisField
    to: int | After | Before

    def create_transform(self, ctx: Context):
        input_shape = ctx.input_shapes.single_shape()
        output_shape = input_shape.moveaxis(self.axis, self.to)

        source = self.input_shapes.single_shape().indexof(self.axis)
        destination = self.output_shapes.single_shape().indexof(self.axis)
        return Transform[Array, Array](
            moveaxis, ctx, output_shape, static={"source": source, "destination": destination}
        )


def moveaxis(inputs: Array, out: Out, source: int, destination: int):
    return jnp.moveaxis(inputs, source, destination)

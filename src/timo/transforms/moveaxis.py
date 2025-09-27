from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.named_axis import NamedAxis
    from timo.named_shape import After, Before
    from timo.transform_context import TransformContext
    from timo.info import Info
    from timo.out import Out

from timo.transform_factory import TransformFactory
from timo.transform_module import TransformModule
from jax import Array
from jax import numpy as jnp


class MoveAxis(TransformFactory):
    def __init__(self, axis: str | NamedAxis, to: int | After | Before):
        super().__init__()
        self.axis = axis
        self.to = to

    def create_module(self, ctx: TransformContext):
        input_shape = ctx.input_shapes.single_shape()
        output_shape = input_shape.moveaxis(self.axis, self.to)

        source = self.input_shapes.single_shape().indexof(self.axis)
        destination = self.output_shapes.single_shape().indexof(self.axis)
        return output_shape, TransformModule[Array, Array](source=source, destination=destination)


def moveaxis(inputs: Array, info: Info, out: Out, source: int, destination: int):
    return jnp.moveaxis(inputs, source, destination)

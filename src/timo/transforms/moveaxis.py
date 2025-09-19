from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.named_axis import NamedAxis
    from timo.named_shape import After, Before
    from timo.transform_context import TransformContext
    from timo.info import Info
    from timo.out import Out

from timo.transform import Transform
import jax
from jax import numpy as jnp


def output_shape(ctx: TransformContext, axis: str | NamedAxis, to: int | After | Before):
    input_shape = ctx.input_shapes.single_shape()
    return input_shape.moveaxis(axis, to)


class MoveAxis(Transform):
    def __init__(self, ctx: TransformContext, axis: str | NamedAxis, to: int | After | Before):
        super().__init__(ctx, output_shape(ctx, axis, to))
        self.function = moveaxis
        self.source = self.input_shapes.single_shape().indexof(axis)
        self.destination = self.output_shapes.single_shape().indexof(axis)

    def transform(self, inputs, info: Info, out: Out):
        return self.function(inputs, self.source, self.destination)


def moveaxis(inputs: jax.Array, source: int, destination: int):
    return jnp.moveaxis(inputs, source, destination)

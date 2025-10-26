from __future__ import annotations
from typing import TYPE_CHECKING

from timo.transform import Transform

if TYPE_CHECKING:
    from timo.context import Context
    from timo.out import Out

from timo.named_axis import NamedAxisField
from jax import Array
from timo.factory import Factory
from jax import numpy as jnp


class Sum(Factory[Array, Array]):
    on: NamedAxisField
    keepdim: bool

    def create_transform(self, ctx: Context) -> Transform:
        from timo.sized_named_axis import size

        input_shape = ctx.input_shapes.single_shape()
        dimension = input_shape.indexof(self.on)
        if self.keepdim:
            output_shape = input_shape.resize(size(self.on, 1))
        else:
            output_shape = input_shape.remove(self.on)
        return Transform(sum, ctx, output_shape, static={"dimension": dimension, "keepdim": self.keepdim})


def sum(inputs: Array, out: Out, dimension: int, keepdim: bool):
    return jnp.sum(inputs, axis=dimension, keepdims=keepdim)

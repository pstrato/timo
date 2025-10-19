from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.context import Context
    from timo.named_axis import NamedAxis

from jax import Array
from timo.factory import Factory
from timo.transform import Transform
from flax import nnx


default_kernel_init = nnx.nn.initializers.lecun_normal()
default_bias_init = nnx.nn.initializers.zeros_init()


class Softmax(Factory[Array, Array]):
    def __init__(self, on: str | NamedAxis):
        super().__init__()
        self.on = on

    def create_transform(self, ctx: Context):
        input_shape = ctx.input_shapes.single_shape()
        axis = input_shape.indexof(self.on)
        return Transform[Array, Array](softmax, ctx, static={"axis": axis})


def softmax(inputs: Array, data: nnx.Dict, axis):
    return nnx.softmax(inputs, axis=axis)

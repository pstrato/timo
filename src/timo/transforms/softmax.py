from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.context import Context
    from timo.named_axis import NamedAxis
    from timo.info import Info
    from timo.out import Out

from jax import Array
from timo.factory import Factory
from timo.transform import Transform
from flax import nnx


default_kernel_init = nnx.nn.initializers.lecun_normal()
default_bias_init = nnx.nn.initializers.zeros_init()


class Softmax(Factory):
    def __init__(self, on: str | NamedAxis):
        super().__init__()
        self.on = on

    def create_transform(self, ctx: Context):
        axis = ctx.input_shapes.single_shape().indexof(self.on)
        return ctx.input_shapes, Transform[Array, Array](softmax, axis=axis)


def softmax(inputs: Array, info: Info, out: Out, axis):
    return nnx.softmax(inputs, axis=axis)

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


class Linear(Factory[Array, Array]):
    def __init__(self, on: str | NamedAxis, to: int | None = None, bias: bool = True):
        super().__init__()
        self.on = on
        self.to = to
        self.bias = bias

    def create_transform(self, ctx: Context):
        from timo.sized_named_axis import size

        in_size = ctx.in_size(self.on)
        to_size = self.to or in_size
        input_shape = ctx.input_shapes.single_shape()
        output_shape = input_shape.resize(size(self.on, to_size))

        kernel = ctx.params(self, "kernel", (in_size, to_size), default_kernel_init)
        if self.bias:
            bias = ctx.params(self, "bias", to_size, default_bias_init)
        else:
            bias = None
        transform = self.vmap(linear, (None,) * 3, self.on)
        return Transform[Array, Array](transform, ctx, output_shape, data={"kernel": kernel, "bias": bias})


def linear(inputs: Array, data: nnx.Dict, kernel: nnx.Param, bias: nnx.Param | None):
    outputs = inputs @ kernel
    if bias is None:
        return outputs
    return outputs + bias

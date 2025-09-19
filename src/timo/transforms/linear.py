from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax import Array
    from timo.transform_context import TransformContext
    from timo.named_axis import NamedAxis
    from timo.info import Info
    from timo.out import Out

from timo.transform import Transform
from flax import nnx


default_kernel_init = nnx.nn.initializers.lecun_normal()
default_bias_init = nnx.nn.initializers.zeros_init()


def output_shape(ctx: TransformContext, on: str | NamedAxis, to: int | None):
    from timo.named_axis import name
    from timo.sized_named_axis import size

    input_shape = ctx.input_shapes.single_shape()
    return input_shape.resize(size(on, to or input_shape[on].set_size))


class Linear(Transform):
    def __init__(self, ctx: TransformContext, on: str | NamedAxis, to: int | None = None, bias: bool = True):

        super().__init__(ctx, output_shape(ctx, on, to))
        self.kernel = self.params(ctx, "kernel", (self.in_size(on), self.to_size(on)), default_kernel_init)
        if bias:
            self.bias = self.params(ctx, "bias", self.to_size(on), default_bias_init)

        self.function = self.vmap(linear, (None, None), on)

    def transform(self, inputs, info: Info, out: Out):
        return self.function(inputs, self.kernel, self.bias)


def linear(inputs: Array, kernel: nnx.Param, bias: nnx.Param | None):
    outputs = inputs @ kernel
    if bias is None:
        return outputs
    return outputs + bias

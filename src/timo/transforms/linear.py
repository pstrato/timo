from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.transform_context import TransformContext
    from timo.named_axis import NamedAxis
    from timo.info import Info
    from timo.out import Out

from jax import Array
from timo.transform_factory import TransformFactory
from timo.transform_module import TransformModule
from flax import nnx


default_kernel_init = nnx.nn.initializers.lecun_normal()
default_bias_init = nnx.nn.initializers.zeros_init()


class Linear(TransformFactory):
    def __init__(self, ctx: TransformContext, on: str | NamedAxis, to: int | None = None, bias: bool = True):
        from timo.sized_named_axis import size

        input_shape = ctx.input_shapes.single_shape()

        super().__init__(ctx, input_shape.resize(size(on, to or input_shape[on].set_size)))
        self.on = on
        self.bias = bias

    def module(self):
        kernel = self.params("kernel", (self.in_size(self.on), self.to_size(self.on)), default_kernel_init)
        if self.bias:
            bias = self.params("bias", self.to_size(self.on), default_bias_init)
        else:
            bias = None
        transform = self.vmap(linear, (None,) * 4, self.on)
        return TransformModule[Array, Array](transform, kernel=kernel, bias=bias)


def linear(inputs: Array, info: Info, out: Out, kernel: nnx.Param, bias: nnx.Param | None):
    outputs = inputs @ kernel
    if bias is None:
        return outputs
    return outputs + bias

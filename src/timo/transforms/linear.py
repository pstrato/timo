from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax import Array
    from timo.transform_context import TransformContext
    from timo.dimension import Dimension

from timo.transform import Transform
from flax import nnx


default_kernel_init = nnx.nn.initializers.lecun_normal()
default_bias_init = nnx.nn.initializers.zeros_init()


class Linear(Transform):
    def __init__(self, ctx: TransformContext, on: str | Dimension, to: int | None = None, bias: bool = True):
        from timo import dim, size

        super().__init__()
        input_shape = ctx.input_shapes.single_shape()
        in_features = input_shape[on].set_count
        self.to = to or in_features
        self.on = dim(on)
        output_shape = input_shape.resize(size(on, self.to))
        self._set_shapes(ctx.input_shapes, output_shape)
        self.kernel = nnx.Param(
            ctx.initializer(("Linear", "kernel"), default_kernel_init)(ctx.rngs.params(), (in_features, to))
        )
        if bias:
            self.bias = nnx.Param(ctx.initializer(("Linear", "bias"), default_bias_init)(ctx.rngs.params(), (to,)))
        function = linear
        for _ in input_shape.before(on):
            function = nnx.vmap(function, in_axes=(0, None, None), out_axes=0)
        for _ in input_shape.after(on):
            function = nnx.vmap(function, in_axes=(-1, None, None), out_axes=-1)

        self.function = function

    def __call__(self, inputs):
        return self.function(inputs, self.kernel, self.bias)


def linear(inputs: Array, kernel: nnx.Param, bias: nnx.Param | None):
    outputs = inputs @ kernel
    if bias is None:
        return outputs
    return outputs + bias

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
from jax import numpy as jnp


default_scale_init = nnx.nn.initializers.ones_init()
default_bias_init = nnx.nn.initializers.zeros_init()


def output_shapes(ctx: TransformContext):
    return ctx.input_shapes


class DynTanH(Transform):
    def __init__(self, ctx: TransformContext, on: str | NamedAxis, bias: bool = True):
        super().__init__(ctx, output_shapes(ctx))
        in_size = ctx.input_shapes.single_shape()[on].set_size
        self.scale = self.params(ctx, "scale", in_size, default_scale_init)
        if bias:
            self.bias = self.params(ctx, "bias", in_size, default_bias_init)

        self.function = self.vmap(dyntanh, (None, None), on)

    def transform(self, inputs: Array, info: Info, out: Out):
        return self.function(inputs, self.scale, self.bias)


def dyntanh(inputs: Array, scale: nnx.Param, bias: nnx.Param | None):
    outputs = inputs * scale
    if bias is None:
        return outputs
    return jnp.tanh(outputs + bias)

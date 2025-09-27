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
from jax import numpy as jnp


default_scale_init = nnx.nn.initializers.ones_init()
default_bias_init = nnx.nn.initializers.zeros_init()


class DynTanh(TransformFactory):
    def __init__(self, on: str | NamedAxis, bias: bool = True):
        super().__init__()
        self.on = on
        self.bias = bias

    def create_module(self, ctx: TransformContext):
        in_size = ctx.in_size(self.on)
        scale = ctx.params("scale", in_size, default_scale_init)
        if self.bias:
            bias = ctx.params("bias", in_size, default_bias_init)
        else:
            bias = None
        transform = self.vmap(dyntanh, (None,) * 4, self.on)
        return ctx.input_shapes, TransformModule[Array, Array](transform, scale=scale, bias=bias)


def dyntanh(inputs: Array, info: Info, out: Out, scale: nnx.Param, bias: nnx.Param | None):
    outputs = inputs * scale
    if bias is None:
        return outputs
    return jnp.tanh(outputs + bias)

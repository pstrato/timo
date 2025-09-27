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
from jax import numpy as jnp


default_scale_init = nnx.nn.initializers.ones_init()
default_bias_init = nnx.nn.initializers.zeros_init()


class DynTanh(Factory):
    def __init__(self, on: str | NamedAxis, bias: bool = True, wide: bool = False):
        super().__init__()
        self.on = on
        self.bias = bias
        self.wide = wide

    def create_module(self, ctx: Context):
        in_size = ctx.in_size(self.on)
        scale_size = in_size if not self.wide else 1
        scale = ctx.params("scale", scale_size, default_scale_init)
        if self.bias:
            bias = ctx.params("bias", scale_size, default_bias_init)
        else:
            bias = None
        transform = self.vmap(dyntanh, (None,) * 4, self.on)
        return ctx.input_shapes, Transform[Array, Array](transform, scale=scale, bias=bias)


def dyntanh(inputs: Array, info: Info, out: Out, scale: nnx.Param, bias: nnx.Param | None):
    outputs = inputs * scale
    if bias is None:
        return outputs
    return jnp.tanh(outputs + bias)

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.context import Context
    from timo.named_axis import NamedAxis

from jax import Array
from timo.factory import Factory
from timo.transform import Transform
from flax import nnx
from jax import numpy as jnp


default_scale_init = nnx.nn.initializers.ones_init()
default_bias_init = nnx.nn.initializers.zeros_init()


class DynTanh(Factory[Array, Array]):
    def __init__(self, on: str | NamedAxis, bias: bool = True, wide: bool = False):
        super().__init__()
        self.on = on
        self.bias = bias
        self.wide = wide

    def create_transform(self, ctx: Context):
        in_size = ctx.in_size(self.on)
        scale_size = in_size if not self.wide else 1
        scale = ctx.params(self, "scale", scale_size, default_scale_init)
        if self.bias:
            bias = ctx.params(self, "bias", scale_size, default_bias_init)
        else:
            bias = None
        transform = self.vmap(dyntanh, (None,) * 3, self.on)
        return Transform[Array, Array](transform, ctx, data={"scale": scale, "bias": bias})


def dyntanh(inputs: Array, data: nnx.Dict, scale: nnx.Param, bias: nnx.Param | None):
    outputs = inputs * scale
    if bias is None:
        return outputs
    return jnp.tanh(outputs + bias)

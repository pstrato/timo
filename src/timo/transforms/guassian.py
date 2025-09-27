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
import jax.numpy as jnp


default_center_init = nnx.nn.initializers.lecun_normal()
default_covar_init = nnx.nn.initializers.ones_init()


class Gaussian(Factory):
    def __init__(self, on: str | NamedAxis, to: int | None = None):
        super().__init__()
        self.on = on
        self.to = to

    def create_transform(self, ctx: Context):
        from timo.sized_named_axis import size

        input_shape = ctx.input_shapes.single_shape()
        in_size = ctx.in_size(self.on)
        to_size = self.to or in_size
        output_shape = input_shape.resize(size(self.on, to_size))

        center = ctx.params("center", (in_size, to_size), default_center_init)
        inv_covar = nnx.Param(jnp.stack([jnp.eye(in_size) for _ in range(to_size)], axis=-1))
        transform = gaussian
        transform = nnx.vmap(transform, in_axes=(None, None, None, -1, -1), out_axes=-1)
        transform = self.vmap(transform, (None,) * 4, self.on)
        return output_shape, Transform[Array, Array](transform, center=center, inv_covar=inv_covar)


def gaussian(inputs: Array, info: Info, out: Out, center: nnx.Param, inv_covar: nnx.Param | None):
    delta = inputs - center
    inv_covar = jnp.abs(jnp.tril(inv_covar) + jnp.tril(inv_covar, -1).transpose())

    outputs = jnp.exp(-0.5 * delta.transpose() @ inv_covar @ delta)
    return outputs

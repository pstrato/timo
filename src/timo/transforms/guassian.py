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
import jax.numpy as jnp


default_center_init = nnx.nn.initializers.lecun_normal()
default_covar_init = nnx.nn.initializers.ones_init()


class Gaussian(TransformFactory):
    def __init__(self, on: str | NamedAxis, to: int | None = None):
        super().__init__()
        self.on = on
        self.to = to

    def create_module(self, ctx: TransformContext):
        from timo.sized_named_axis import size

        input_shape = ctx.input_shapes.single_shape()
        in_size = ctx.in_size(self.on)
        to_size = self.to or in_size
        output_shape = input_shape.resize(size(self.on, to_size))

        center = ctx.params("center", (in_size, to_size), default_center_init)
        covar = nnx.Param(jnp.stack([jnp.eye(in_size) for _ in range(to_size)], axis=-1))
        transform = gaussian
        transform = nnx.vmap(transform, in_axes=(None, None, None, -1, -1), out_axes=-1)
        transform = self.vmap(transform, (None,) * 4, self.on)
        return output_shape, TransformModule[Array, Array](transform, center=center, covar=covar)


def gaussian(inputs: Array, info: Info, out: Out, center: nnx.Param, covar: nnx.Param | None):
    delta = inputs - center
    covar = jnp.abs(jnp.tril(covar) + jnp.tril(covar, -1).transpose())
    covar_inv = jnp.linalg.inv(covar)

    outputs = jnp.exp(-0.5 * delta.transpose() @ covar_inv @ delta)
    return outputs

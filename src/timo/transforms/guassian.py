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
    def __init__(self, ctx: TransformContext, on: str | NamedAxis, to: int | None = None):
        from timo.sized_named_axis import size

        input_shape = ctx.input_shapes.single_shape()

        super().__init__(ctx, input_shape.resize(size(on, to or input_shape[on].set_size)))
        self.on = on

    def module(self):
        center = self.params("center", (self.in_size(self.on), self.to_size(self.on)), default_center_init)
        covar = nnx.Param(jnp.stack([jnp.eye(self.in_size(self.on)) for _ in range(self.to_size(self.on))], axis=-1))
        transform = gaussian
        transform = nnx.vmap(transform, in_axes=(None, None, None, -1, -1), out_axes=-1)
        transform = self.vmap(transform, (None,) * 4, self.on)
        return TransformModule[Array, Array](transform, center=center, covar=covar)


def gaussian(inputs: Array, info: Info, out: Out, center: nnx.Param, covar: nnx.Param | None):
    delta = inputs - center
    covar = jnp.abs(jnp.tril(covar) + jnp.tril(covar, -1).transpose())
    covar_inv = jnp.linalg.inv(covar)

    outputs = jnp.exp(-0.5 * delta.transpose() @ covar_inv @ delta)
    return outputs

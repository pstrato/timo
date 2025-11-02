from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.context import Context

from timo.named_axis import NamedAxisField
from jax import Array
from timo.factory import Factory
from timo.transform import Transform
from flax import nnx
import jax.numpy as jnp


default_center_init = nnx.nn.initializers.lecun_normal()


class Gaussian(Factory[Array, Array]):
    on: NamedAxisField
    to: int | None = None
    exclusive: bool = False

    def create_transform(self, ctx: Context):
        from timo.sized_named_axis import size

        input_shape = ctx.input_shapes.single_shape()
        in_size = ctx.in_size(self.on)
        to_size = self.to or in_size
        output_shape = input_shape.resize(size(self.on, to_size))

        center = ctx.params(self, "center", (in_size, to_size), default_center_init)
        scale = nnx.Param(jnp.stack([jnp.eye(in_size) for _ in range(to_size)], axis=-1))
        transform = gaussian
        transform = nnx.vmap(transform, in_axes=(None, -1, -1), out_axes=-1)
        if self.exclusive:
            transform = exclusive(transform, to_size)
        transform = self.vmap(transform, (None,) * 2, self.on)
        return Transform[Array, Array](transform, ctx, output_shape, data={"center": center, "scale": scale})


def gaussian(inputs: Array, center: nnx.Param[Array], scale: nnx.Param[Array]):
    delta = inputs - center
    outputs = jnp.exp(-((delta.transpose() @ scale @ delta) ** 2))
    return outputs


def exclusive(transform, to: int):
    others = []
    for i in range(to):
        others.append(i_others := [])
        for j in range(to):
            if j != i:
                i_others.append(j)
    others = jnp.array(others)

    def exclusive_transform(inputs: Array, center: nnx.Param[Array], scale: nnx.Param[Array]):
        outputs = transform(inputs, center, scale)
        other_outputs = outputs[others]
        exclusion = jnp.prod(1 - other_outputs, -1)
        return outputs * exclusion

    return exclusive_transform

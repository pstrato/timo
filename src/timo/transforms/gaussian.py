from __future__ import annotations
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from timo.context import Context

from timo.named_axis import NamedAxisField
from jax import Array
from timo.factory import Factory
from timo.transform import Transform
from flax import nnx
import jax.numpy as jnp
from jax.lax import stop_gradient

default_center_init = nnx.nn.initializers.normal(1)
default_scale_init = nnx.nn.initializers.normal(1)


class Gaussian(Factory[Array, Array]):
    on: NamedAxisField
    to: int | None = None
    exclusive: bool = False
    scale: bool = True
    symetric: bool = True
    center_init: Callable = default_center_init
    scale_init: Callable = default_scale_init

    def create_transform(self, ctx: Context):
        from timo.sized_named_axis import size

        input_shape = ctx.input_shapes.single_shape()
        in_size = ctx.in_size(self.on)
        to_size = self.to or in_size
        output_shape = input_shape.resize(size(self.on, to_size))

        center = ctx.params(self, "center", (in_size, to_size), self.center_init)
        if self.scale:
            scale = nnx.Param(
                jnp.stack(
                    [jnp.eye(in_size) + self.scale_init(ctx.rngs.params(), (in_size, in_size)) for _ in range(to_size)],
                    axis=-1,
                )
            )
        else:
            scale = jnp.stack(
                [jnp.eye(in_size) for _ in range(to_size)],
                axis=-1,
            )
        transform = gaussian
        transform = nnx.vmap(transform, in_axes=(None, -1, -1, None), out_axes=-1)
        if self.exclusive and to_size > 1:
            transform = exclusive(transform, to_size)
        transform = self.vmap(transform, (None,) * 3, self.on)
        return Transform[Array, Array](
            transform,
            ctx,
            self,
            output_shape,
            data={"center": center, "scale": scale},
            static={"symetric": self.symetric},
        )


def gaussian(inputs: Array, center: Array, scale: Array, symetric: bool):
    delta = inputs - center
    if symetric:
        scale_tril = jnp.tril(scale)
        scale_tril1 = jnp.tril(scale, -1)
        symetric_scale = scale_tril + scale_tril1.T
        scale = symetric_scale
    outputs = jnp.exp(-abs((delta.transpose() @ scale @ delta)))
    return outputs


def exclusive(transform, to: int):
    others = []
    for i in range(to):
        others.append(i_others := [])
        for j in range(to):
            if j != i:
                i_others.append(j)
    others = jnp.array(others)

    def exclusive_transform(inputs: Array, center: nnx.Param[Array], scale: nnx.Param[Array], symetric: bool):
        outputs = transform(inputs, center, scale, symetric)
        max_other_output = stop_gradient(outputs)[others].max(-1)
        return outputs * (1 - max_other_output)

    return exclusive_transform


def gaussian_transform_loss(create_inputs: Callable[[Array, Array, nnx.Rngs], Array]):

    def loss(transform: Transform, rngs: nnx.Rngs):
        center = transform.data["center"]
        inputs = create_inputs(center, transform.data["scale"], rngs)
        result = transform(inputs)
        diag = jnp.diag_indices(center.shape[1])
        result = result[*diag]
        return nnx.relu(1 - result - 0.3)

    return loss

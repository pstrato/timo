from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.context import Context
    from timo.named_axis import NamedAxis

from jax import Array
from timo.factory import Factory
from timo.transform import Transform
from flax import nnx
import jax.numpy as jnp


default_center_init = nnx.nn.initializers.normal()
default_spread_init = nnx.nn.initializers.normal()


class GaussianActivation(Factory[Array, Array]):
    def __init__(self, on: str | NamedAxis, eps=1e-4):
        super().__init__()
        self.on = on
        self.eps = eps

    def create_transform(self, ctx: Context):
        from timo.sized_named_axis import size

        in_size = ctx.in_size(self.on)

        center = ctx.params(self, "center", (in_size,), default_center_init)
        spread = ctx.params(self, "spread", (in_size,), default_spread_init)
        transform = gaussian
        transform = self.vmap(transform, (None,) * 4, self.on)
        return Transform[Array, Array](
            transform, ctx, data={"center": center, "spread": spread}, static={"eps": self.eps}
        )


def gaussian(inputs: Array, data: nnx.Dict, center: nnx.Param[Array], spread: nnx.Param[Array], eps: float):
    delta = inputs - center

    outputs = jnp.exp(-(delta**2) / (spread**2 + eps))
    return outputs

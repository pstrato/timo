from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.context import Context

from jax import Array
from timo.factory import Factory
from timo.transform import Transform
from flax import nnx


default_kernel_init = nnx.nn.initializers.lecun_normal()
default_bias_init = nnx.nn.initializers.zeros_init()


class Softplus(Factory[Array, Array]):

    def create_transform(self, ctx: Context):
        return Transform[Array, Array](softplus, ctx, self)


def softplus(inputs: Array):
    return nnx.softplus(inputs)

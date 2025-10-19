from __future__ import annotations
from timo.factory import Factory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.context import Context

from jax import Array
from flax.nnx import relu, leaky_relu
from timo.transform import Transform


class Function(Factory[Array, Array]):
    def __init__(self, function, data: dict = {}, static: dict = {}):
        super().__init__()
        self.function = function
        self.data = data
        self.static = static

    def create_transform(self, ctx: Context):
        return Transform[Array, Array](call, ctx, data=self.data, static={"function": self.function, **self.static})


def call(inputs, data, function, **kwargs):
    return function(inputs, **kwargs)


class Id(Function):
    def __init__(self):
        super().__init__(id)


def id(inputs):
    return inputs


class ReLU(Function):
    def __init__(self):
        super().__init__(relu)


class LeakyReLU(Function):
    def __init__(self, negative_slope=0.01):
        super().__init__(leaky_relu, static={"negative_slope": negative_slope})

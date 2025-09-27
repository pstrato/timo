from __future__ import annotations
from timo.factory import Factory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.context import Context

from jax import Array
from flax.nnx import relu, leaky_relu, softmax
from timo.transform import Transform


class Function(Factory):
    def __init__(self, function, **kwargs):
        super().__init__()
        self.function = function
        self.kwargs = kwargs

    def create_transform(self, ctx: Context):
        return ctx.input_shapes, Transform[Array, Array](call, function=self.function, **self.kwargs)


def call(inputs, info, out, function, **kwargs):
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
        super().__init__(leaky_relu, negative_slope=negative_slope)

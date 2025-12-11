from __future__ import annotations
from timo.factory import Factory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.context import Context

from typing import Callable
from jax import Array
from flax.nnx import relu, leaky_relu, sigmoid, tanh
from timo.transform import Transform


class Function(Factory[Array, Array]):
    function: Callable
    data: dict = {}
    static: dict = {}

    def create_transform(self, ctx: Context):
        return Transform[Array, Array](
            call, ctx, self, data=self.data, static={"function": self.function, **self.static}
        )


def call(inputs, function, **kwargs):
    return function(inputs, **kwargs)


class Id(Function):
    def __init__(self):
        super().__init__(function=id)


def id(inputs):
    return inputs


class ReLU(Function):
    def __init__(self):
        super().__init__(function=relu)


class LeakyReLU(Function):
    def __init__(self, negative_slope=0.01):
        super().__init__(function=leaky_relu, static={"negative_slope": negative_slope})


class Sigmoid(Function):
    def __init__(self):
        super().__init__(function=sigmoid)


class TanH(Function):
    def __init__(self):
        super().__init__(function=tanh)

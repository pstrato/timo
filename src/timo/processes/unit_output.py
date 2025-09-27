from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.named_axis import NamedAxis
    from timo.context import Context
    from timo.factory import Factory
    from timo.info import Info
    from timo.out import Out

from timo.process import Process
from timo.transform import Transform
from jax.lax import stop_gradient
from jax import numpy as jnp


class UnitOutput(Process):
    def __init__(self, on: str | NamedAxis, weight: float = 1, target: float = 1):
        super().__init__()
        self.on = on
        self.weight = weight
        self.target = target

    def module(self, ctx: Context, factory: Factory):
        transform = factory.transform(ctx)
        axis = factory.input_shapes.single_shape().indexof(self.on)
        return Transform(unit_output, normed=transform, axis=axis, target=self.target, weight=self.weight)


def unit_output(inputs, info: Info, out: Out, normed: Transform, axis: int, target: float, weight: float):
    training = normed.training
    if not training:
        return normed(inputs, info=info, out=out)

    unit_output = normed(stop_gradient(inputs), info=info, out=out)
    loss = (jnp.sum(unit_output**2, axis=axis) ** 0.5 - target).mean() * weight
    out.add_loss(loss)
    return normed(inputs, info=info, out=out)

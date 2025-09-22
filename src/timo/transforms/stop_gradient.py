from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.transform_context import TransformContext
    from timo.info import Info
    from timo.out import Out

from jax import Array
from timo.transform_factory import TransformFactory
from timo.transform_module import TransformModule
from jax.lax import stop_gradient


class StopGradient(TransformFactory):
    def __init__(self, ctx: TransformContext):
        super().__init__(ctx, ctx.input_shapes)

    def module(self):
        return TransformModule[Array, Array](transform)


def transform(input: Array, info: Info, out: Out):
    return stop_gradient(input)

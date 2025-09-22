from __future__ import annotations
from timo.transform_factory import TransformFactory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.transform_context import TransformContext

from jax import Array
from timo.transform_module import TransformModule


class Id(TransformFactory):
    def __init__(self, ctx: TransformContext):
        super().__init__(ctx, ctx.input_shapes)

    def module(self):
        return TransformModule[Array, Array](id)


def id(inputs, info, out):
    return inputs

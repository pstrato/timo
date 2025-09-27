from __future__ import annotations
from timo.transform_factory import TransformFactory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.transform_context import TransformContext

from jax import Array
from timo.transform_module import TransformModule


class Id(TransformFactory):

    def create_module(self, ctx: TransformContext):
        return ctx.input_shapes, TransformModule[Array, Array](id)


def id(inputs, info, out):
    return inputs

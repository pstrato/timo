from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.context import Context

from jax import Array
from timo.factory import Factory
from timo.transform import Transform

from flax.nnx import Sequential as SequentialNNX


class Sequential(Factory[Array, Array]):
    transforms: tuple[Factory, ...]

    def create_transform(self, ctx: Context):

        output_ctx = ctx
        modules = []
        for transform in self.transforms:
            module = transform.transform(output_ctx)
            modules.append(module)
            output_ctx = module.output_ctx
        sequential_nnx = SequentialNNX(*modules)
        return Transform[Array, Array](sequential, ctx, output_ctx.input_shapes, data={"transforms": sequential_nnx})


def sequential(inputs: Array, transforms: SequentialNNX):
    return transforms(inputs)

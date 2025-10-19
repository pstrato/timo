from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.context import Context

from jax import Array
from flax import nnx
from timo.factory import Factory
from timo.transform import Transform


class Sequential(Factory[Array, Array]):
    def __init__(self, *transforms: Factory):
        super().__init__()
        self.transforms = transforms

    def create_transform(self, ctx: Context):

        output_ctx = ctx
        modules = []
        for transform in self.transforms:
            module = transform.transform(output_ctx)
            modules.append(module)
            output_ctx = module.output_ctx
        return Transform[Array, Array](sequential, ctx, output_ctx.input_shapes, data={"transforms": modules})


def sequential(inputs: Array, data: nnx.Dict, transforms: tuple[Transform, ...]):
    outputs = inputs
    for transform in transforms:
        outputs = transform(outputs, data)
    return outputs

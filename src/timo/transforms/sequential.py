from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax import Array
    from timo.transform_context import TransformContext
    from timo.info import Info
    from timo.out import Out

from timo.transform import Transform


class Sequential(Transform):
    def __init__(self, ctx: TransformContext, *transforms: Transform):

        super().__init__(ctx, transforms[-1].output_shapes)
        self.function = sequential
        self._transforms = transforms

    def transform(self, inputs, info: Info, out: Out):
        return self.function(inputs, *self._transforms, info=info, out=out)


def sequential(inputs: Array, *transforms: Transform, info: Info, out: Out):
    outputs = inputs
    for transform in transforms:
        outputs = transform(outputs, info=info, out=out)
    return outputs

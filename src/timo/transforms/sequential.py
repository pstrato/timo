from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.info import Info
    from timo.out import Out
    from timo.context import Context

from jax import Array
from timo.factory import Factory
from timo.transform import Transform


class Sequential(Factory):
    def __init__(self, *transforms: Factory):
        super().__init__()
        self.transforms = transforms

    def create_transform(self, ctx: Context):
        modules = []
        for transform in self.transforms:
            module = transform.transform(ctx)
            modules.append(module)
            ctx = ctx.push(transform)
        return ctx.input_shapes, Transform[Array, Array](sequential, transforms=modules)


def sequential(inputs: Array, transforms: tuple[Factory, ...], info: Info, out: Out):
    outputs = inputs
    for transform in transforms:
        outputs = transform(outputs, info=info, out=out)
    return outputs

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.info import Info
    from timo.out import Out
    from timo.transform_context import TransformContext

from jax import Array
from timo.transform_factory import TransformFactory
from timo.transform_module import TransformModule


class Sequential(TransformFactory):
    def __init__(self, *transforms: TransformFactory):
        super().__init__()
        self.transforms = transforms

    def create_module(self, ctx: TransformContext):
        modules = []
        for transform in self.transforms:
            module = transform.module(ctx)
            modules.append(module)
            ctx = ctx.push(transform)
        return ctx.input_shapes, TransformModule[Array, Array](sequential, transforms=modules)


def sequential(inputs: Array, transforms: tuple[TransformFactory, ...], info: Info, out: Out):
    outputs = inputs
    for transform in transforms:
        outputs = transform(outputs, info=info, out=out)
    return outputs

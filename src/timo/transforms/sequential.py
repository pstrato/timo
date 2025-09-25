from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.info import Info
    from timo.out import Out

from jax import Array
from timo.transform_factory import TransformFactory
from timo.transform_module import TransformModule


class Sequential(TransformFactory):
    def __init__(self, *transforms: TransformFactory):

        super().__init__(transforms[0].ctx, transforms[-1].transform_output_shapes)
        self._transforms = transforms

    def module(self):
        transforms = []
        for transform in self._transforms:
            transforms.append(transform.module())
        return TransformModule[Array, Array](sequential, transforms=transforms)


def sequential(inputs: Array, transforms: tuple[TransformFactory, ...], info: Info, out: Out):
    outputs = inputs
    for transform in transforms:
        outputs = transform(outputs, info=info, out=out)
    return outputs

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.named_axis import NamedAxis
    from timo.info import Info
    from timo.out import Out

from jax import Array
from timo.transform_factory import TransformFactory
from timo.transform_module import TransformModule
from jax.lax import concatenate


class Thread(TransformFactory):
    def __init__(self, *transforms: TransformFactory, on: str | NamedAxis):
        from timo.sized_named_axis import size

        if len(transforms) == 0:
            raise ValueError("Expected at least one transform")
        first_transform = transforms[0]
        first_output_shape = first_transform.transform_output_shapes.single_shape()
        concat_shape = first_output_shape.remove(on)
        axis = first_output_shape.indexof(on)
        concat_size = first_output_shape[on].set_size
        for other_transform in transforms[1:]:
            if first_transform.input_shapes != other_transform.input_shapes:
                raise ValueError("Incompatible input shapes")
            other_output_shape = other_transform.transform_output_shapes.single_shape()
            if axis != other_output_shape.indexof(on):
                raise ValueError("Incompatible output shape")
            if other_output_shape.remove(on) != concat_shape:
                raise ValueError("Incompatible output shape")
            concat_size += other_output_shape[on].set_size
        output_shape = first_output_shape.resize(size(on, concat_size))
        super().__init__(first_transform.ctx, output_shape)
        self._transforms = transforms
        self._axis = axis

    def module(self):
        transforms = []
        for transform in self._transforms:
            transforms.append(transform.module())
        return TransformModule[Array, Array](thread, transforms=transforms, axis=self._axis)


def thread(inputs: Array, info: Info, out: Out, transforms: tuple[TransformFactory, ...], axis: int):
    outputs = []
    for transform in transforms:
        outputs.append(transform(inputs, info=info, out=out))
    return concatenate(outputs, dimension=axis)

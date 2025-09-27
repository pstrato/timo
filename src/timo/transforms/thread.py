from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.named_axis import NamedAxis
    from timo.transform_context import TransformContext
    from timo.info import Info
    from timo.out import Out

from jax import Array
from timo.transform_factory import TransformFactory
from timo.transform_module import TransformModule
from jax.lax import concatenate


class Thread(TransformFactory):
    def __init__(self, *transforms: TransformFactory, on: str | NamedAxis):
        super().__init__()
        self.transforms = transforms
        self.on = on

    def create_module(self, ctx: TransformContext):
        from timo.sized_named_axis import size

        if len(self.transforms) == 0:
            raise ValueError("Expected at least one transform")

        modules = []
        for transform in self.transforms:
            modules.append(transform.module(ctx))

        first_transform = self.transforms[0]
        first_output_shape = first_transform.output_shapes.single_shape()
        concat_shape = first_output_shape.remove(self.on)
        axis = first_output_shape.indexof(self.on)
        concat_size = first_output_shape[self.on].set_size
        for other_transform in self.transforms[1:]:
            if first_transform.input_shapes != other_transform.input_shapes:
                raise ValueError("Incompatible input shapes")
            other_output_shape = other_transform.output_shapes.single_shape()
            if axis != other_output_shape.indexof(self.on):
                raise ValueError("Incompatible output shape")
            if other_output_shape.remove(self.on) != concat_shape:
                raise ValueError("Incompatible output shape")
            concat_size += other_output_shape[self.on].set_size
        output_shape = first_output_shape.resize(size(self.on, concat_size))

        return output_shape, TransformModule[Array, Array](thread, transforms=modules, axis=axis)


def thread(inputs: Array, info: Info, out: Out, transforms: tuple[TransformFactory, ...], axis: int):
    outputs = []
    for transform in transforms:
        outputs.append(transform(inputs, info=info, out=out))
    return concatenate(outputs, dimension=axis)

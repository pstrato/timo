from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.named_axis import NamedAxis
    from timo.transform_context import TransformContext
    from timo.info import Info
    from timo.out import Out

from timo.transform import Transform
from jax.lax import concatenate


class Thread(Transform):
    def __init__(self, *transforms: Transform, on: str | NamedAxis):
        from timo.sized_named_axis import size

        if len(transforms) == 0:
            raise ValueError("Expected at least one transform")
        first_transform = transforms[0]
        first_output_shape = first_transform.output_shapes.single_shape()
        concat_shape = first_output_shape.remove(on)
        axis = first_output_shape.indexof(on)
        concat_size = first_output_shape[on].set_size
        for other_transform in transforms[1:]:
            if first_transform.input_shapes != other_transform.input_shapes:
                raise ValueError("Incompatible input shapes")
            other_output_shape = other_transform.output_shapes.single_shape()
            if axis != other_output_shape.indexof(on):
                raise ValueError("Incompatible output shape")
            if other_output_shape.remove(on) != concat_shape:
                raise ValueError("Incompatible output shape")
            concat_size += other_output_shape[on].set_size
        output_shape = first_output_shape.resize(size(on, concat_size))
        super().__init__(first_transform.ctx, output_shape)
        self.function = thread
        self.transforms = transforms
        self.axis = axis

    def transform(self, *args, info, out):
        return self.function(*args, info=info, out=out, transforms=self.transforms, axis=self.axis)


def thread(*args, info: Info, out: Out, transforms: tuple[Transform, ...], axis: int):
    outputs = []
    for transform in transforms:
        outputs.append(transform(*args, info=info, out=out))
    return concatenate(outputs, dimension=axis)

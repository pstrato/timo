from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.context import Context

from timo.named_axis import NamedAxisField
from jax import Array
from timo.factory import Factory
from timo.transform import Transform
from jax.lax import concatenate


class Thread(Factory[Array, Array]):
    transforms: tuple[Factory, ...]
    on: NamedAxisField

    def create_transform(self, ctx: Context):
        from timo.transforms.concat import concat_shape

        if len(self.transforms) == 0:
            raise ValueError("Expected at least one transform")

        modules = []
        for transform in self.transforms:
            modules.append(transform.transform(ctx))

        first_transform = self.transforms[0]
        shapes = [first_transform.output_shapes.single_shape()]
        for other_transform in self.transforms[1:]:
            if first_transform.input_shapes != other_transform.input_shapes:
                raise ValueError("Incompatible input shapes")
            shapes.append(other_transform.output_shapes.single_shape())
        dimension, output_shape = concat_shape(self.on, *shapes)

        return Transform[Array, Array](
            thread, ctx, output_shape, data={"transforms": modules}, static={"dimension": dimension}
        )


def thread(inputs: Array, transforms: tuple[Transform, ...], dimension: int):
    return concatenate(tuple(transform(inputs) for transform in transforms), dimension=dimension)

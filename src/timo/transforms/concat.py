from __future__ import annotations
from typing import TYPE_CHECKING

from timo.transform import Transform

if TYPE_CHECKING:
    from timo.named_axis import NamedAxis
    from timo.named_shape import NamedShape
    from timo.context import Context

from timo.context import Context
from timo.named_axis import NamedAxisField
from jax import Array
from timo.factory import Factory
from jax.lax import concatenate


def concat_shape(on: str | NamedAxis, *shapes: NamedShape):
    from timo.sized_named_axis import size

    if len(shapes) == 0:
        raise ValueError("At least one shape required")
    first_shape = shapes[0]
    dimension = first_shape.indexof(on)
    if len(shapes) == 1:
        return dimension, shapes[0]
    concat_shape = first_shape.remove(on)
    dimension = first_shape.indexof(on)
    concat_size = first_shape[on].set_size
    for other_shape in shapes[1:]:
        if dimension != other_shape.indexof(on):
            raise ValueError("Incompatible output shape")
        if other_shape.remove(on) != concat_shape:
            raise ValueError("Incompatible output shape")
        concat_size += other_shape[on].set_size
    output_shape = first_shape.resize(size(on, concat_size))
    return dimension, output_shape


class Concat(Factory[tuple[Array, Array], Array]):
    on: NamedAxisField

    def create_transform(self, ctx: Context) -> Transform:
        dimension, output_shape = concat_shape(self.on, *ctx.input_shapes)
        return Transform(concat, ctx, output_shape, static={"dimension": dimension})


def concat(inputs: tuple[Array, Array], dimension: int):
    return concatenate(inputs, dimension)

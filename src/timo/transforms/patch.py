from __future__ import annotations
from typing import TYPE_CHECKING

from numpy import concat

from timo.sized_named_axis import size

if TYPE_CHECKING:
    from timo.context import Context

from typing import Callable
from timo.named_axis import NamedAxis

PatchCoordinates = Callable[[tuple[str | NamedAxis, ...]], tuple[tuple[tuple[int, ...], ...]]]

from timo.named_axis import NamedAxisField
from functools import cache, partial
from timo.factory import Factory
from timo.transform import Transform
from jax import jit, numpy as jnp, Array
from flax import nnx


def s3(distance) -> PatchCoordinates:
    if distance <= 0:
        raise ValueError()

    def patch(on: tuple[str | NamedAxis, ...]) -> tuple[tuple[tuple[int, ...], ...]]:
        ndim = len(on)
        if ndim != 2:
            raise ValueError("s3 only for dimension 2")

        coordinates = []

        coordinates.append(((-distance, -distance), (-distance, -distance + 1), (-distance + 1, -distance)))
        coordinates.append(((-distance, +distance), (-distance, +distance - 1), (-distance + 1, +distance)))
        coordinates.append(((+distance, -distance), (+distance, -distance + 1), (+distance - 1, -distance)))
        coordinates.append(((+distance, +distance), (+distance, +distance - 1), (+distance - 1, +distance)))

        for i in range(-distance + 2, +distance - 1, 2):

            coordinates.append(((i - 1, -distance), (i, -distance), (i + 1, -distance)))
            coordinates.append(((i - 1, +distance), (i, +distance), (i + 1, +distance)))
            coordinates.append(((-distance, i - 1), (-distance, i), (-distance, i + 1)))
            coordinates.append(((+distance, i - 1), (+distance, i), (+distance, i + 1)))

        return tuple(sorted(coordinates))

    return patch


def corner(distance: int, dy: int, dx: int) -> PatchCoordinates:
    if distance < 0:
        raise ValueError()

    def patch(on: tuple[str | NamedAxis, ...]) -> tuple[tuple[tuple[int, ...], ...]]:
        ndim = len(on)
        if ndim == 2:
            if distance == 0:
                return (((0, 0),),)
            coordinates = []
            for i in range(distance):
                coordinates.append((distance * dy, i * dx))
            coordinates.append((distance * dy, distance * dx))
            for i in range(distance):
                coordinates.append((i * dy, distance * dx))
            return (tuple(sorted(coordinates)),)
        raise ValueError("Corner for dimension 2 only")

    return patch


def tl_corner(distance: int) -> PatchCoordinates:
    return corner(distance, dy=-1, dx=-1)


def tr_corner(distance: int) -> PatchCoordinates:
    return corner(distance, dy=-1, dx=+1)


def bl_corner(distance: int) -> PatchCoordinates:
    return corner(distance, dy=+1, dx=-1)


def br_corner(distance: int) -> PatchCoordinates:
    return corner(distance, dy=+1, dx=+1)


def square(distance: int) -> PatchCoordinates:
    if distance < 0:
        raise ValueError()

    def patch(on: tuple[str | NamedAxis, ...]):
        ndim = len(on)
        if ndim == 2:
            if distance == 0:
                return (((0, 0),),)
            coordinates = []
            for x in range(-distance, distance + 1):
                coordinates.append((x, -distance))
                coordinates.append((x, +distance))
            for y in range(-distance + 1, distance):
                coordinates.append((-distance, y))
                coordinates.append((+distance, y))
            return (tuple(sorted(coordinates)),)

        raise ValueError(f"Square for dimension 2 only")

    return patch


def compbine_patches(*patch_coordinates: PatchCoordinates):
    def patch(on: tuple[str | NamedAxis, ...]):
        coordinates = []
        for pc in patch_coordinates:
            coordinates.extend(pc(on))
        return tuple(coordinates)

    return patch


@cache
def padding(on: tuple[NamedAxis, ...], coordinates: tuple[tuple[tuple[int, ...]]]):
    padding = [0 for _ in on]
    for patch in coordinates:
        for coordinate in patch:
            for i, c in enumerate(coordinate):
                padding[i] = max(padding[i], abs(c))
    return padding


@cache
def offset(on: tuple[NamedAxis, ...], coordinates: tuple[tuple[tuple[int, ...]]]):
    offsets = []
    for patch in coordinates:
        offset = [[] for _ in on]
        for coordinate in patch:
            for i, c in enumerate(coordinate):
                offset[i].append(c)
        offset = tuple(tuple(o) for o in offset)
        offsets.append(offset)
    return tuple(offsets)


class Patch(Factory[Array, Array]):
    on: tuple[NamedAxisField, ...]
    coordinates: PatchCoordinates
    value_axis: NamedAxisField = "V"  # type: ignore
    patch_axis: NamedAxisField = "P"  # type: ignore
    stat: str | None = None
    pad_value: float | None = None
    concat: NamedAxisField | None = None  # type: ignore

    def create_transform(self, ctx: Context):
        from timo.named_shape import shape

        coordinates = self.coordinates(self.on)

        p = padding(self.on, coordinates)
        if self.stat is None:
            transform = patch
        elif self.stat == "max":
            transform = patch_max
        elif self.stat == "min":
            transform = patch_min
        elif self.stat == "mean":
            transform = patch_mean
        else:
            raise ValueError(f"Unsupported stat: `{self.stat}`")
        transform = self.vmap(transform, (None,) * 3, *self.on)

        input_shape = ctx.input_shapes.single_shape()
        if self.concat is not None:
            channel_axis = input_shape.indexof(self.concat)
            transform = concat_patch(transform, channel_axis, self.stat is not None)

        o = offset(self.on, coordinates)

        if self.stat is None:
            patch_dim_size = len(coordinates)
            if self.concat is None:
                output_shape = shape(input_shape, (self.patch_axis, len(o)), (self.value_axis, patch_dim_size))
            else:
                channels = input_shape[self.concat].set_size
                output_shape = input_shape.resize(size(self.concat, channels * len(o) * patch_dim_size))
        else:
            if self.concat is None:
                output_shape = shape(input_shape, (self.patch_axis, len(o)))
            else:
                channels = input_shape[self.concat].set_size
                output_shape = input_shape.resize(size(self.concat, channels * len(o)))

        return Transform(
            transform,
            ctx,
            self,
            output_shape,
            static={"pad_value": self.pad_value, "padding": p, "offset": o},
        )


def _patch(inputs: Array, padding: tuple, offset: tuple, pad_value: float) -> Array:
    patch_index = index(inputs.shape, padding, offset)
    patch_size = len(offset[0][0])
    patch_count = len(offset)
    inputs_shape = inputs.shape
    inputs = jnp.pad(inputs, padding, mode="constant", constant_values=pad_value)
    output = inputs[*patch_index]
    output = jnp.reshape(output, (patch_count, *inputs_shape, patch_size))
    output = jnp.moveaxis(output, 0, -2)
    return output


def patch(inputs: Array, padding: tuple, offset: tuple, pad_value: float | None) -> Array:
    return _patch(inputs, padding, offset, pad_value if pad_value is not None else 0)


def patch_min(inputs: Array, padding: tuple, offset: tuple, pad_value: float | None) -> Array:
    outputs = _patch(inputs, padding, offset, pad_value if pad_value is not None else jnp.inf)
    return jnp.min(outputs, -1)


def patch_max(inputs: Array, padding: tuple, offset: tuple, pad_value: float | None) -> Array:
    outputs = _patch(inputs, padding, offset, pad_value if pad_value is not None else -jnp.inf)
    return jnp.max(outputs, -1)


def patch_mean(inputs: Array, padding: tuple, offset: tuple, pad_value: float | None) -> Array:
    counts = _patch(jnp.ones_like(inputs), padding, offset, pad_value if pad_value is not None else 0).sum(-1)
    counts = jnp.where(counts > 0, counts, 1)
    outputs = _patch(inputs, padding, offset, 0).sum(-1)
    return outputs / counts


def concat_patch(transform, axis: int, stat: bool):

    def with_concat(inputs: Array, padding: tuple, offset: tuple, pad_value: float):
        outputs = transform(inputs, padding, offset, pad_value)
        outputs = jnp.moveaxis(outputs, axis, -1)
        if stat:
            outputs = jnp.reshape(outputs, (*outputs.shape[:-2], -1))
        else:
            outputs = jnp.reshape(outputs, (*outputs.shape[:-3], -1))
        outputs = jnp.moveaxis(outputs, -1, axis)
        return outputs

    return with_concat


def index(shape: tuple[int, ...], padding: tuple, offset: tuple):
    if len(shape) == 2:
        i_size = shape[0]
        j_size = shape[1]
        input_size = i_size * j_size
        i_offsets = []
        j_offsets = []
        for p in range(len(offset)):
            offset_size = len(offset[p][0])

            i = jnp.repeat(jnp.repeat(jnp.arange(0, i_size), j_size), offset_size)
            j = jnp.repeat(jnp.tile(jnp.arange(0, j_size), i_size), offset_size)

            i_offset = jnp.tile(jnp.array(offset[p][0]), input_size) + padding[0] + i
            j_offset = jnp.tile(jnp.array(offset[p][1]), input_size) + padding[1] + j

            i_offsets.append(i_offset)
            j_offsets.append(j_offset)
        return jnp.stack(i_offsets), jnp.stack(j_offsets)

    raise NotImplementedError()

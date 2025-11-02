from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.context import Context

from typing import Callable
from timo.named_axis import NamedAxis

PatchCoordinates = Callable[[tuple[str | NamedAxis, ...]], tuple[tuple[int, ...]]]

from timo.named_axis import NamedAxisField
from functools import cache
from timo.factory import Factory
from timo.transform import Transform
from jax import numpy as jnp, Array, lax


def square(distance: int):
    if distance <= 0:
        raise ValueError()

    def patch(on: tuple[str | NamedAxis, ...]):
        ndim = len(on)
        if ndim == 2:
            coordinates = []
            for x in range(-distance, distance + 1):
                coordinates.append((x, -distance))
                coordinates.append((x, +distance))
            for y in range(-distance + 1, distance):
                coordinates.append((-distance, y))
                coordinates.append((+distance, y))
            return tuple(sorted(coordinates))

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
def padding(on: tuple[NamedAxis, ...], coordinates: tuple[tuple[int, ...]]):
    padding = [0 for _ in on]
    for coordinate in coordinates:
        for i, c in enumerate(coordinate):
            padding[i] = max(padding[i], abs(c))
    return tuple(padding)


@cache
def offset(on: tuple[NamedAxis, ...], coordinates: tuple[tuple[int, ...]]):
    offset = [[] for _ in on]
    for coordinate in coordinates:
        for i, c in enumerate(coordinate):
            offset[i].append(c)
    offset = tuple(tuple(o) for o in offset)
    return offset


class Patch(Factory[Array, Array]):
    on: tuple[NamedAxisField, ...]
    coordinates: PatchCoordinates
    axis: NamedAxisField = "P"  # type: ignore
    stat: str | None = None

    def create_transform(self, ctx: Context):
        from timo.named_shape import shape

        coordinates = self.coordinates(self.on)
        if self.stat is None:
            patch_dim_size = len(coordinates)
            input_shape = ctx.input_shapes.single_shape()
            output_shape = shape(input_shape, (self.axis, patch_dim_size))
        else:
            output_shape = ctx.input_shapes

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

        o = offset(self.on, coordinates)

        transform = self.vmap(transform, (None,) * 2, *self.on)
        return Transform(transform, ctx, output_shape, static={"padding": p, "offset": o})


def _patch(inputs: Array, padding: tuple[int, ...], offset: tuple[tuple[int, ...], ...], pad_value: float) -> Array:
    patch_index = index(inputs.shape, padding, offset)
    patch_dim_size = len(offset[0])
    inputs_shape = inputs.shape
    inputs = jnp.pad(inputs, padding, mode="constant", constant_values=pad_value)
    outputs = inputs[*patch_index]
    outputs = jnp.reshape(outputs, (*inputs_shape, patch_dim_size))
    return outputs


def patch(inputs: Array, padding: tuple[int, ...], offset: tuple[tuple[int, ...], ...]) -> Array:
    return _patch(inputs, padding, offset, 0)


def patch_min(inputs: Array, padding: tuple[int, ...], offset: tuple[tuple[int, ...], ...]) -> Array:
    outputs = _patch(inputs, padding, offset, jnp.inf)
    return jnp.min(outputs, -1)


def patch_max(inputs: Array, padding: tuple[int, ...], offset: tuple[tuple[int, ...], ...]) -> Array:
    outputs = _patch(inputs, padding, offset, -jnp.inf)
    return jnp.max(outputs, -1)


def patch_mean(inputs: Array, padding: tuple[int, ...], offset: tuple[tuple[int, ...], ...]) -> Array:
    counts = _patch(jnp.ones_like(inputs), padding, offset, 0).sum(-1)
    outputs = _patch(inputs, padding, offset, 0).sum(-1)
    return outputs / counts


def index(shape: tuple[int, ...], padding: tuple[int, ...], offset: tuple[tuple[int, ...], ...]):
    if len(shape) == 2:
        i_size = shape[0]
        j_size = shape[1]
        input_size = i_size * j_size
        offset_size = len(offset[0])

        i = jnp.repeat(jnp.repeat(jnp.arange(0, i_size), j_size), offset_size)
        j = jnp.repeat(jnp.tile(jnp.arange(0, j_size), i_size), offset_size)

        i_offset = jnp.tile(jnp.array(offset[0]), input_size) + padding[0] + i
        j_offset = jnp.tile(jnp.array(offset[1]), input_size) + padding[1] + j

        return jnp.array((i_offset, j_offset))
    raise NotImplementedError()

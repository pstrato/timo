from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable
    from jax import Array
    from timo.context import Context
    from timo.named_axis import NamedAxis
    from timo.info import Info
    from timo.out import Out

    PatchCoordinates = Callable[[tuple[NamedAxis, ...]], tuple[tuple[int, ...]]]

from functools import cache
from timo.factory import Factory
from timo.transform import Transform
from jax import numpy as jnp


def square(distance: int):
    if distance <= 0:
        raise ValueError()

    def patch(on: tuple[NamedAxis, NamedAxis]):
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
    def patch(on: tuple[NamedAxis, ...]):
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


class Patch(Factory):
    def __init__(
        self,
        on: tuple[str | NamedAxis, ...],
        coordinates: PatchCoordinates,
        axis: str | NamedAxis = "P",
        stat: str | None = None,
    ):
        super().__init__()
        self.on = on
        self.coordinates = coordinates
        self.axis = axis
        self.stat = stat

    def create_module(self, ctx: Context):
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

        transform = self.vmap(transform, (None,) * 4, *self.on)
        return output_shape, Transform(transform, padding=p, coordinates=coordinates)


def _patch(inputs: Array, padding: tuple[int, ...], coordinates: tuple[tuple[int, ...]], pad_value: float) -> Array:
    patch_index = index(inputs.shape, padding, coordinates)
    patch_dim_size = len(coordinates)
    inputs_shape = inputs.shape
    inputs = jnp.pad(inputs, padding, mode="constant", constant_values=pad_value)
    outputs = inputs[*patch_index]
    outputs = jnp.reshape(outputs, (*inputs_shape, patch_dim_size))
    return outputs


def patch(inputs: Array, info: Info, out: Out, padding: tuple[int, ...], coordinates: tuple[tuple[int, ...]]) -> Array:
    return _patch(inputs, padding, coordinates, 0)


def patch_min(
    inputs: Array, info: Info, out: Out, padding: tuple[int, ...], coordinates: tuple[tuple[int, ...]]
) -> Array:
    outputs = _patch(inputs, padding, coordinates, jnp.inf)
    return jnp.min(outputs, -1)


def patch_max(
    inputs: Array, info: Info, out: Out, padding: tuple[int, ...], coordinates: tuple[tuple[int, ...]]
) -> Array:
    outputs = _patch(inputs, padding, coordinates, -jnp.inf)
    return jnp.max(outputs, -1)


def patch_mean(inputs: Array, info: Info, out: Out, padding: tuple[int, ...], coordinates: tuple[tuple[int]]) -> Array:
    patch_count = count(inputs.shape, coordinates)
    outputs = _patch(inputs, padding, coordinates, 0)
    outputs = jnp.sum(outputs, -1)
    return outputs / patch_count


def index(shape: tuple[int, ...], padding: tuple[int, ...], coordinates: tuple[tuple[int, ...]]):
    if len(shape) == 2:
        index = [[], []]
        for i in range(shape[0]):
            for j in range(shape[1]):
                for coordinate in coordinates:
                    index[0].append(padding[0] + i + coordinate[0])
                    index[1].append(padding[1] + j + coordinate[1])
        return jnp.array(index)
    raise NotImplementedError()


def count(shape: tuple[int, ...], coordinates: tuple[tuple[int, ...]]):
    if len(shape) == 2:
        count = []
        for i in range(shape[0]):
            j_count = []
            for j in range(shape[1]):
                patch_count = 0
                for coordinate in coordinates:
                    pi = i + coordinate[0]
                    if pi < 0 or pi >= shape[0]:
                        continue
                    pj = j + coordinate[1]
                    if pj < 0 or pj >= shape[1]:
                        continue
                    patch_count += 1
                j_count.append(patch_count)
            count.append(j_count)
        return jnp.array(count)
    raise NotImplementedError()

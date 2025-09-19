from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable
    from jax import Array
    from timo.transform_context import TransformContext
    from timo.named_axis import NamedAxis
    from timo.info import Info
    from timo.out import Out

    PatchCoordinates = Callable[[tuple[NamedAxis, ...]], tuple[tuple[int, ...]]]

from functools import cache
from timo.transform import Transform
from jax import numpy as jnp
from jax.lax import stop_gradient


def square_patch(distance: int):
    if distance <= 0:
        raise ValueError()

    def patch(on: tuple[NamedAxis, ...]):
        ndim = len(on)
        if ndim == 1:
            return ((-distance,), (+distance,))
        if ndim == 2:
            coordinates = []
            for x in range(-distance, distance + 1):
                coordinates.append((x, -distance))
                coordinates.append((x, +distance))
            for y in range(-distance + 1, distance):
                coordinates.append((-distance, y))
                coordinates.append((+distance, y))
            return tuple(coordinates)

        raise NotImplementedError()

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


def count(shape: tuple[int, ...], padding: tuple[int, ...], coordinates: tuple[tuple[int, ...]]):
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
                j_count.append(count)
            count.append(j_count)
        return jnp.array(count)
    raise NotImplementedError()


class Patch(Transform):
    def __init__(
        self,
        ctx: TransformContext,
        on: tuple[str | NamedAxis, ...],
        coordinates: PatchCoordinates,
        axis: str | NamedAxis = "P",
        stat: str | None = None,
    ):
        from timo.named_shape import shape

        coords = coordinates(on)
        if stat is None:
            patch_dim_size = len(coords)
            input_shape = ctx.input_shapes.single_shape()
            output_shape = shape(input_shape, (axis, patch_dim_size))
        else:
            output_shape = ctx.input_shapes
        super().__init__(ctx, output_shape)

        self.coordinates = coords
        self.padding = padding(on, self.coordinates)
        self.function = self.vmap(patch, (None, None, None), *on)
        self.stat = stat

    def transform(self, inputs, info: Info, out: Out):
        return self.function(inputs, self.padding, self.stat, self.coordinates)


def patch(inputs: Array, padding: tuple[int, ...], stat: str | None, coordinates: tuple[tuple[int]]):

    if stat is not None:
        if stat == "min":
            pad_value = jnp.inf
        elif stat == "max":
            pad_value = -jnp.inf
        elif stat == "mean":
            pad_value = 0
        else:
            raise ValueError(f"Unknown stat: `{stat}`, expected `min`, `max`, or `mean`")

    patch_index = stop_gradient(index(inputs.shape, padding, coordinates))
    patch_dim_size = len(coordinates)
    inputs_shape = inputs.shape
    inputs = jnp.pad(inputs, padding, mode="constant", constant_values=pad_value)
    outputs = inputs[*patch_index]
    outputs = jnp.reshape(outputs, (*inputs_shape, patch_dim_size))
    if stat is not None:
        if stat == "min":
            return jnp.min(outputs, -1)
        if stat == "max":
            return jnp.max(outputs, -1)
        if stat == "mean":
            patch_count = stop_gradient(count(inputs.shape, padding, coordinates))
            return jnp.sum(outputs, -1) / patch_count
        else:
            raise ValueError()
    return outputs / patch_count

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.named_axis import NamedAxis
    from timo.named_shape import NamedShape
    from timo.named_shape_sequence import NamedShapeSequence

from timo.transform_module import TransformModule
from timo.transform_context import TransformContext

from flax.nnx import Param, vmap, Initializer


class TransformFactory(TransformContext):
    def __init__(self, ctx: TransformContext, output_shapes: NamedShapeSequence | NamedShape):
        from timo.named_shape_sequence import shapes

        self._ctx = ctx
        self._input_shapes = ctx.input_shapes
        self._output_shapes = shapes(output_shapes)

    def get(self, name, default=...):
        if name == "input_shapes":
            return self._input_shapes
        if name == "output_shapes":
            return self._output_shapes
        return self._ctx.get(name, default)

    @property
    def ctx(self):
        return self._ctx

    @property
    def input_shapes(self):
        if self._input_shapes is None:
            raise ValueError("Transform shape not set")
        return self._input_shapes

    @property
    def output_shapes(self):
        if self._output_shapes is None:
            raise ValueError("Transform shape not set")
        return self._output_shapes

    def module(self) -> TransformModule:
        raise NotImplementedError()

    def params(self, kind: str, shape: int | tuple[int, ...], default_init: Initializer):
        return Param(self.initializer(self, kind, default_init)(self.rngs.params(), shape))

    def in_size(self, on: str | NamedAxis):
        return self.input_shapes.single_shape()[on].set_size

    def to_size(self, on: str | NamedAxis):
        return self.output_shapes.single_shape()[on].set_size

    def vmap(self, function: callable, non_mapped_args: tuple, *on: str | NamedAxis):
        for _ in self.input_shapes.single_shape().before(*on):
            function = vmap(function, in_axes=(0, *non_mapped_args), out_axes=0)
        for _ in self.input_shapes.single_shape().after(*on):
            function = vmap(function, in_axes=(-1, *non_mapped_args), out_axes=-1)
        return function

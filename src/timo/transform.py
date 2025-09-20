from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.named_axis import NamedAxis
    from timo.named_shape import NamedShape
    from timo.named_shape_sequence import NamedShapeSequence
    from timo.info import Info
    from timo.out import Out
    from timo.transform_context import TransformContext

from flax.nnx import Module, Param, vmap, Initializer


class Transform(Module):
    def __init__(self, ctx: TransformContext, output_shapes: NamedShapeSequence | NamedShape):
        from timo.named_shape_sequence import shapes

        self._input_shapes = ctx.input_shapes
        self._output_shapes = shapes(output_shapes)

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

    def __call__(self, *args, info: Info | None = None, out: Out | None = None):
        return self.transform(*args, info=info, out=out)

    def transform(self, *args, info: Info, out: Out):
        raise NotImplementedError()

    def params(self, ctx: TransformContext, kind: str, shape: int | tuple[int, ...], default_init: Initializer):
        return Param(ctx.initializer(self, kind, default_init)(ctx.rngs.params(), shape))

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

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.named_axis import NamedAxis
    from timo.named_shape import NamedShape
    from timo.named_shape_sequence import NamedShapeSequence

from timo.transform import Transform
from timo.context import Context

from flax.nnx import vmap


class Factory:
    def __init__(self):
        super().__init__()
        self._input_shapes: NamedShapeSequence | None = None
        self._output_shapes: NamedShapeSequence | None = None

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

    def module(self, ctx: Context) -> Transform:
        from timo.named_shape_sequence import shapes

        self._ctx = ctx
        self._input_shapes = ctx.input_shapes
        output_shapes, module = self.create_module(ctx)
        self._output_shapes = shapes(output_shapes)
        return module

    def create_module(self, ctx: Context) -> tuple[NamedShapeSequence | NamedShape, Transform]:
        raise NotImplementedError()

    def vmap(self, function: callable, non_mapped_args: tuple, *on: str | NamedAxis):
        for _ in self.input_shapes.single_shape().before(*on):
            function = vmap(function, in_axes=(0, *non_mapped_args), out_axes=0)
        for _ in self.input_shapes.single_shape().after(*on):
            function = vmap(function, in_axes=(-1, *non_mapped_args), out_axes=-1)
        return function

    def __rshift__(self, value: Factory):
        from timo.transforms.sequential import Sequential

        transforms = []
        if isinstance(self, Sequential):
            transforms.extend(self.transforms)
        else:
            transforms.append(self)
        if isinstance(value, Sequential):
            transforms.extend(value.transforms)
        else:
            transforms.append(value)

        return Sequential(*transforms)

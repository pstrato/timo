from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.shape import Shape
    from timo.shapes import Shapes

from flax.nnx import Module


class Transform(Module):
    def __init__(self):
        self._input_shapes: Shapes | None = None
        self._output_shapes: Shapes | None = None

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

    def _set_shapes(self, input_shapes: Shapes | Shape, output_shapes: Shapes | Shape):
        from timo.shapes import shapes

        if self._input_shapes is not None or self._output_shapes is not None:
            raise ValueError("Transform shape already set")
        self._input_shapes = shapes(input_shapes)
        self._output_shapes = shapes(output_shapes)

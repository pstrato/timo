from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.dimension import Dimension

from timo.transform import Transform


class Linear(Transform):
    def __init__(self, on: str | Dimension, to: int | None = None):
        from timo import dim

        super().__init__()

        self.on = dim(on)
        self.to = to

    def validate(self, inputs):
        from timo.node import UnaryNode

        assert len(inputs) == 1
        assert isinstance(inputs[0], UnaryNode)

    def name(self, inputs, output_shapes):
        input_shape = inputs[0].shapes[0]
        output_shape = output_shapes[0]
        return f"Linear({output_shape - input_shape})"

    def output_shapes(self, inputs):
        from timo.size import size

        input_shape = inputs[0].shapes[0]
        if self.to is None:
            return input_shape
        return input_shape.resize(size(self.on, self.to))

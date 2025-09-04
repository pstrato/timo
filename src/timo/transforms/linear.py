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

    def name(self, output_shape, input_shape):
        return f"Linear({output_shape - input_shape})"

    def output_shape(self, input_shape):
        from timo.size import size

        if self.to is None:
            return input_shape
        return input_shape.resize(size(self.on, self.to))

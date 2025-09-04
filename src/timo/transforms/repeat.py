from __future__ import annotations
from typing import TYPE_CHECKING

from timo.split import SplitOp, SplitNode


class RepeatOp(SplitOp):
    def __init__(self, to: int):
        super().__init__()
        self.to = to

    def name(self, input_shape, *output_shapes):
        return f"Repeat({input_shape})"

    def output_shapes(self, input_shape):
        return [input_shape for _ in range(self.to)]


class Repeat(SplitNode):
    def __init__(self, parent, to: int):
        super().__init__(RepeatOp(to), parent)

    def __iter__(self):
        for _ in range(self.split.to):
            yield self.parent

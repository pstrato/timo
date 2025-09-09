from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


from timo.transform import Transform


class Repeat(Transform):
    def __init__(self, to: int):
        super().__init__()
        self.to = to

    def validate(self, inputs):
        from timo.node import UnaryNode

        assert len(inputs) == 1
        assert isinstance(inputs[0], UnaryNode)

    def name(self, inputs, output_shapes):
        input_shape = inputs[0].shapes[0]
        return f"Repeat({input_shape}, {self.to})"

    def output_shapes(self, inputs):
        input_shape = inputs[0].shapes[0]
        return (input_shape,) * self.to

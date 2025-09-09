from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.dimension import Dimension
    from timo.node import UnaryNode

from timo.transform import Transform, TransformNode


class ConcatOp(Transform):
    def __init__(self, on: str | Dimension):
        from timo.dimension import dim

        self.on = dim(on)

    def validate(self, inputs):
        from timo.node import UnaryNode

        assert len(inputs) >= 1
        for input in inputs:
            assert isinstance(input, UnaryNode)

        if len(inputs) == 1:
            return
        input = inputs[0]
        for other_input in inputs[1:]:
            if input.shapes != other_input.shapes:
                raise ValueError()

    def name(self, inputs, output_shapes):
        from timo.size import size

        input_shape = inputs[0].shapes[0].resize(size(self.on, None))
        return f"Concat({output_shapes[0] - input_shape})"

    def output_shapes(self, inputs):
        from timo.size import size

        check_shape = None
        concat_size = 0
        for input in inputs:
            input_shape = input.shapes[0]
            input_check_shape = input_shape.resize(size(self.on, None))
            concat_size += input_shape[self.on].size
            if check_shape is None:
                check_shape = input_check_shape
            else:
                if input_check_shape != check_shape:
                    raise ValueError(f"Different size: {input_check_shape} != {check_shape}")
        output_shape = check_shape.resize(size(self.on, concat_size))
        return output_shape


class Concat(TransformNode):
    def __init__(self, on: str | Dimension, *inputs: UnaryNode):
        from timo.node import Node

        super().__init__(ConcatOp(on), *map(Node.unary, inputs))

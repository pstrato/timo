from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.dimension import Dimension

from timo.group import Group, GroupNode


class ConcatOp(Group):
    def __init__(self, on: str | Dimension):
        from timo.dimension import dim

        super().__init__()

        self.on = dim(on)

    def name(self, output_shape, *input_shapes):
        from timo.size import size

        assert len(input_shapes) > 1
        input_shape = input_shapes[0].resize(size(self.on, None))
        return f"Concat({output_shape - input_shape})"

    def output_shape(self, *input_shapes):
        from timo.size import size
        from timo.shape import Shape

        assert len(input_shapes) > 1

        check_shape = None
        concat_size = 0
        for input_shape in input_shapes:
            input_check_shape = input_shape.resize(size(self.on, None))
            concat_size += input_shape[self.on].size
            if check_shape is None:
                check_shape = input_check_shape
            else:
                if input_check_shape != check_shape:
                    raise ValueError(f"Different size: {input_check_shape} != {check_shape}")
        output_shape = check_shape.resize(size(self.on, concat_size))
        return output_shape


class Concat(GroupNode):
    def __init__(self, *parents, on=None):
        super().__init__(ConcatOp(on), *parents)

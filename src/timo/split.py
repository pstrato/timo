from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.shape import Shape

from timo.node import Node


class SplitOp:
    def name(self, input_shape: Shape, *output_shapes: Shape):
        raise NotImplementedError()

    def output_shapes(self, input_shape: Shape):
        raise NotImplementedError()


class SplitNode(Node):
    __slots__ = [*Node.__slots__, "_split"]

    def __init__(self, split: SplitOp, parent: Node):
        input_shape = parent.shape
        output_shape = split.output_shapes(input_shape)
        super().__init__(output_shape, split.name(input_shape, output_shape), parent)
        self._split = split

    @property
    def split(self):
        return self._split

    def __iter__(self):
        raise NotImplementedError()

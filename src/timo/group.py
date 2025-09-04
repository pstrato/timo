from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.shape import Shape

from timo.node import Node


class Group:
    def name(self, output_shape: Shape, *input_shapes: Shape):
        raise NotImplementedError()

    def output_shape(self, *input_shapes: Shape):
        raise NotImplementedError()


class GroupNode(Node):
    __slots__ = [*Node.__slots__, "_group"]

    def __init__(self, group: Group, *parents: Node):
        assert len(parents) > 1

        input_shapes = [parent.shape for parent in parents]
        output_shape = group.output_shape(*input_shapes)
        super().__init__(output_shape, group.name(output_shape, *input_shapes), None)
        self._group = group

    @property
    def group(self):
        return self._group

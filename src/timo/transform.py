from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.shape import Shape

from timo.node import Node


class Transform:
    def name(self, input_shape: Shape, output_shape: Shape):
        raise NotImplementedError()

    def output_shape(self, input_shape: Shape):
        raise NotImplementedError()


class TransformNode(Node):
    __slots__ = [*Node.__slots__, "_transform"]

    def __init__(self, transform: Transform, parent: Node):
        input_shape = parent.shape
        output_shape = transform.output_shape(input_shape)
        super().__init__(output_shape, transform.name(input_shape, output_shape), parent)
        self._transform = transform

    @property
    def transform(self):
        return self._transform

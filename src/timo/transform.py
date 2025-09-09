from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.shape import Shape

from timo.node import Node


class Transform:
    def validate(self, inputs: tuple[Node]):
        raise NotImplementedError()

    def name(self, inputs: tuple[Node], output_shapes: tuple[Shape]):
        raise NotImplementedError()

    def output_shapes(self, inputs: tuple[Node]) -> Shape | tuple[Shape]:
        raise NotImplementedError()


class TransformNode(Node):
    __slots__ = ["_name", "_inputs", "_shapes", "_outputs", "_transform"]

    def __init__(self, transform: Transform, *inputs: Node):
        from timo.shape import Shape
        from timo.node import OutputNode

        super().__init__()
        transform.validate(inputs)
        self._transform = transform
        self._inputs = inputs
        self._shapes = transform.output_shapes(inputs)
        if isinstance(self._shapes, Shape):
            self._shapes = (self._shapes,)
        self._name = transform.name(inputs, self._shapes)
        self._outputs = tuple(
            OutputNode(f"[{i}]" if len(self._shapes) > 1 else None, self, shape) for i, shape in enumerate(self._shapes)
        )

    @property
    def transform(self):
        return self._transform

    @property
    def name(self):
        return self._name

    @property
    def inputs(self):
        return self._inputs

    @property
    def shapes(self):
        return self._shapes

    @property
    def outputs(self):
        return self._outputs

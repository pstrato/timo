from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.shape import Shape
    from timo.transform import Transform


class Node:
    __slots__ = ["_shape", "_name", "_parent", "_id"]

    def __init__(self, shape: Shape, name: str, parent: Node | None):
        self._shape = shape
        self._name = name
        self._parent = parent

    @property
    def shape(self):
        return self._shape

    @property
    def name(self):
        return self._name

    @property
    def parent(self):
        return self._parent

    def parents(self):
        parent = self._parent
        while parent is not None:
            yield parent
            parent = parent.parent

    def path(self):
        yield self
        yield from self.parents()

    def __str__(self):
        return "/".join(map(lambda n: n.name, reversed(list(self.path()))))

    def __repr__(self):
        return str(self)

    def __rshift__(self, transform: Transform):
        from timo.transform import TransformNode

        return TransformNode(transform, self)

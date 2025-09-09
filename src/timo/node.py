from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterable
    from timo.shape import Shape
    from timo.transform import Transform


class Node:

    @property
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def inputs(self) -> tuple[UnaryNode]:
        raise NotImplementedError()

    @property
    def shapes(self) -> tuple[Shape]:
        raise NotImplementedError()

    @property
    def outputs(self) -> tuple[UnaryNode]:
        raise NotImplementedError()

    def parents(self):
        for input in self.inputs:
            yield input
            yield from input.parents()

    def path(self):
        yield self
        yield from self.parents()

    def __str__(self):
        path = self.path()
        named = filter(lambda n: n.name is not None, path)
        hierarchy = reversed(list(named))

        return "/".join(map(lambda n: n.name, hierarchy))

    def __repr__(self):
        return str(self)

    def __iter__(self):
        yield from self._outputs

    def __rshift__(self, transform: Transform):
        from timo.transform import TransformNode

        if len(self.outputs) != 1:
            raise ValueError()

        return TransformNode(transform, self.unary())

    def unary(self):
        if isinstance(self, UnaryNode):
            return self
        if len(self.outputs) == 1:
            return self.outputs[0]
        raise ValueError()


class UnaryNode(Node):
    def __rshift__(self, transform: Transform):
        from timo.transform import TransformNode

        return TransformNode(transform, self)


class InputNode(UnaryNode):
    __slots__ = ["_name", "_shape"]

    def __init__(self, name, shape):
        super().__init__()
        self._name = name
        self._shape = shape

    @property
    def name(self):
        return self._name

    @property
    def inputs(self):
        return tuple()

    @property
    def shapes(self):
        return (self._shape,)

    @property
    def outputs(self):
        return (self,)


class OutputNode(UnaryNode):
    __slots__ = ["_name", "_parent", "_shape"]

    def __init__(self, name, parent: Node, shape: Shape):
        super().__init__()
        self._name = name
        self._parent = parent
        self._shape = shape

    @property
    def name(self):
        return self._name

    @property
    def inputs(self):
        return (self._parent,)

    @property
    def shapes(self):
        return (self._shape,)

    @property
    def outputs(self):
        return (self,)

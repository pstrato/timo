from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.dimension import Dimension, SingleDimension
    from timo.size import Size
    from timo.transform import Transform


class Shape:
    __slots__ = ["_sizes"]

    def __init__(self, *sizes: Size):
        self._sizes = sizes

    @property
    def sizes(self):
        return self._sizes

    def __or__(self, value: SingleDimension | Size | Shape):
        from timo.dimension import SingleDimension
        from timo.size import size

        if isinstance(value, SingleDimension):
            return Shape(*self.sizes, size(value, None))
        if isinstance(value, Size):
            return Shape(*self.sizes, value)
        if isinstance(value, Shape):
            return Shape(*self.sizes, *value.sizes)
        raise ValueError()

    def __sub__(self, value: Shape):
        unique_sizes = []
        for size in self.sizes:
            if size not in value.sizes:
                unique_sizes.append(size)
        return Shape(*unique_sizes)

    def __rshift__(self, transform: Transform):
        from timo.node import Node
        from timo.transform import TransformNode

        root = Node(self, str(self), None)
        return TransformNode(transform, root)

    def resize(self, new_size: Size):
        from timo.size import Size

        new_sizes = []
        found = False
        for size in self.sizes:
            if size.dimension == new_size.dimension:
                found = True
                new_sizes.append(new_size)
            else:
                new_sizes.append(size)
        assert found
        return Shape(*new_sizes)

    def __eq__(self, value):
        if not isinstance(value, Shape):
            return False

        return self.sizes == value.sizes

    def __ne__(self, value):
        if not isinstance(value, Shape):
            return True

        return self.sizes != value.sizes

    def __getitem__(self, index: int | str | Dimension):
        from timo.dimension import dim

        if isinstance(index, int):
            return self.sizes[index]
        index = dim(index)
        for size in self.sizes:
            if size.dimension == index:
                return size
        raise IndexError(f"Dimension: `{index}` not in shape {self}")

    def __str__(self):
        return "|".join(map(str, self.sizes))

    def __repr__(self):
        return str(self)


def shape(*sizes: Size):
    return Shape(*sizes)

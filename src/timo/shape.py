from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.dimension import Dimension, SingleDimension
    from timo.size import Size


class Shape:
    __slots__ = ["_sizes"]

    def __init__(self, *sizes: Size):
        self._sizes = sizes

    @property
    def sizes(self):
        return self._sizes

    def __or__(self, value: SingleDimension | Size | Shape):
        return shape(self, value)

    def __sub__(self, value: Shape):
        kept = []
        for size in self.sizes:
            if size not in value.sizes:
                kept.append(size)
        return Shape(*kept)

    def resize(self, new_size: Size):

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

    def indexof(self, dimension: str | Dimension):
        from timo.dimension import dim

        dimension = dim(dimension)
        for i, size in enumerate(self._sizes):
            if size.dimension == dimension:
                return i
        raise ValueError()

    def before(self, dimension: str | Dimension):
        return self._sizes[: self.indexof(dimension)]

    def after(self, dimension: str | Dimension):
        return self._sizes[self.indexof(dimension) + 1 :]

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


def shape(*values: SingleDimension | Size | Shape):
    from timo.dimension import SingleDimension
    from timo.size import Size, SingleSize

    sizes = []
    for value in values:
        if isinstance(value, SingleDimension):
            sizes.append(SingleSize(value, None))
        elif isinstance(value, Size):
            sizes.append(value)
        elif isinstance(value, Shape):
            sizes.extend(value.sizes)
        else:
            raise ValueError()
    return Shape(*sizes)

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.dimension import Dimension


class Size:
    @property
    def dimension(self) -> Dimension:
        raise NotImplementedError()

    @property
    def size(self) -> int | None:
        raise NotImplementedError()

    @property
    def is_set(self):
        return self.size is not None

    def __str__(self):
        return f"{self.dimension} {self.size or '*'}"

    def __repr__(self):
        return f"{self.dimension} {self.size or '*'}"

    def __eq__(self, value):
        if not isinstance(value, Size):
            return False
        return self.dimension == value.dimension and self.size == value.size

    def __ne__(self, value):
        if not isinstance(value, Size):
            return True

        return self.dimension != value.dimension or self.size != value.size

    def __add__(self, value: Size):
        if not isinstance(value, Size):
            raise ValueError()
        if isinstance(self, MultiSize):
            return MultiSize(*self.sizes, value)
        if isinstance(self, SingleSize):
            return MultiSize(self, value)
        raise ValueError()

    def shape(self):
        from timo.shape import Shape

        return Shape(self)


class SingleSize(Size):
    __slots__ = ["_dimension", "_size"]

    def __init__(self, dimension: Dimension, size: int):
        from timo.dimension import Dimension

        assert isinstance(dimension, Dimension)
        assert size is None or isinstance(size, int)

        self._dimension = dimension
        self._size = size

    @property
    def dimension(self):
        return self._dimension

    @property
    def size(self):
        return self._size


def size(dimension: Dimension | str, size: int | None):
    from timo.dimension import dim

    return SingleSize(dim(dimension), size)


class MultiSize(Size):
    __slots__ = ["_dimension", "_size", "_sizes"]

    def __init__(self, *sizes: Size):
        from timo.dimension import MultiDimension

        assert len(sizes) > 0
        for s in sizes:
            assert isinstance(s, Size)

        self._sizes = sizes
        self._size = 1
        for s in sizes:
            if s.size == None:
                self._size = None
                break
            self._size *= s.size
        dimensions = []
        for s in sizes:
            dimensions.append(s.dimension)
        self._dimension = MultiDimension(*dimensions)

    @property
    def dimension(self):
        return self._dimension

    @property
    def size(self):
        return self._size

    @property
    def sizes(self):
        return self._sizes


def sizes(**dims_sizes):
    sizes = []
    for dim, dim_size in dims_sizes.items():
        sizes.append(size(dim, dim_size))
    return MultiSize(*sizes)

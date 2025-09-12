from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.dimension import Dimension, SingleDimension
    from timo.shape import Shape
    from typing import Iterable


class Size:
    @property
    def dimension(self) -> Dimension:
        raise NotImplementedError()

    @property
    def count(self) -> int | None:
        raise NotImplementedError()

    @property
    def set_count(self):
        count = self.count
        if count is None:
            raise ValueError()
        return count

    @property
    def is_set(self):
        return self.count is not None

    def __str__(self):
        return f"{self.dimension}{self.count or '*'}"

    def __repr__(self):
        return str(self)

    def __eq__(self, value):
        if not isinstance(value, Size):
            return False
        return self.dimension == value.dimension and self.count == value.count

    def __ne__(self, value):
        if not isinstance(value, Size):
            return True

        return self.dimension != value.dimension or self.count != value.count

    def __add__(self, value: SingleDimension | Size):
        from timo.dimension import SingleDimension

        if isinstance(value, SingleDimension):
            return GroupedSize(*self._single_sizes(), size(value, None))
        if isinstance(value, Size):
            return GroupedSize(*self._single_sizes(), *value._single_sizes())
        raise ValueError()

    def __or__(self, value: SingleDimension | Size | Shape):
        from timo.shape import shape

        return shape(self, value)

    def _single_sizes(self) -> Iterable[SingleSize]:
        raise NotImplementedError()


class SingleSize(Size):
    __slots__ = ["_dimension", "_count"]

    def __init__(self, dimension: SingleDimension, count: int):
        from timo.dimension import SingleDimension

        assert isinstance(dimension, SingleDimension)
        assert count is None or isinstance(count, int)

        self._dimension = dimension
        self._count = count

    @property
    def dimension(self):
        return self._dimension

    @property
    def count(self):
        return self._count

    def _single_sizes(self):
        yield self


def size(dimension: Dimension | str, count: int | None):
    from timo.dimension import dim

    return SingleSize(dim(dimension), count)


class GroupedSize(Size):
    __slots__ = ["_dimension", "_count", "_sizes"]

    def __init__(self, *sizes: Size):
        from timo.dimension import group_dim

        assert len(sizes) > 0
        self._sizes = sizes
        self._count = 1
        for s in sizes:
            if s.count == None:
                self._count = None
                break
            self._count *= s.count
        self._dimension = group_dim(*map(lambda s: s.dimension, sizes))

    @property
    def dimension(self):
        return self._dimension

    @property
    def count(self):
        return self._count

    @property
    def sizes(self):
        return self._sizes

    def _single_sizes(self):
        return self._sizes


def group_size(**values):
    return GroupedSize(*map(size, values.items()))

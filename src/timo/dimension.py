from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.size import Size
    from timo.shape import Shape


class Dimension:
    @property
    def name(self) -> str:
        raise NotImplementedError()

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def __eq__(self, value):
        if not isinstance(value, Dimension):
            return False
        return self.name == value.name

    def __ne__(self, value):
        if not isinstance(value, Dimension):
            return True
        return self.name != value.name

    def __add__(self, value: Dimension | str):
        value = dim(value)
        return GroupedDimension(*self._single_dimensions(), *value._single_dimensions())

    def _single_dimensions(self):
        raise NotImplementedError()


class SingleDimension(Dimension):
    __slots__ = ["_name"]

    def __init__(self, name: str):
        assert name is not None and len(name) > 0

        self._name = name

    @property
    def name(self):
        return self._name

    def __mul__(self, value: int | None):
        return self.size(value)

    def size(self, size: int | None):
        from timo.size import SingleSize

        if size is not None and not isinstance(size, int):
            raise ValueError()

        return SingleSize(self, size)

    def _single_dimensions(self):
        yield self

    def __or__(self, value: SingleDimension | Size | Shape):
        from timo.shape import shape

        return shape(self, value)


def dim(name_or_dim: str | Dimension):
    from timo.dimension import Dimension

    if isinstance(name_or_dim, Dimension):
        return name_or_dim
    if isinstance(name_or_dim, str):
        return SingleDimension(name_or_dim)
    raise ValueError()


class GroupedDimension(Dimension):
    __slots__ = ["_name", "_dimensions"]

    def __init__(self, *dimensions: SingleDimension):
        assert len(dimensions) > 0

        self._dimensions = dimensions
        self._name = ".".join(map(lambda d: d.name, self._dimensions))

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimensions(self):
        return self._dimensions

    def __getitem__(self, index: int):
        return self._dimensions[index]

    def _single_dimensions(self):
        return self._dimensions


def group(*names_or_dims: str | Dimension):
    dimensions = []
    for name_or_dim in names_or_dims:
        dimensions.append(dim(name_or_dim))
    return GroupedDimension(*dimensions)

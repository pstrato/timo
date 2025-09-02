from __future__ import annotations


class Dimension:
    @property
    def name(self) -> str:
        raise NotImplementedError()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, value):
        if not isinstance(value, Dimension):
            return False
        return self.name == value.name

    def __ne__(self, value):
        if not isinstance(value, Dimension):
            return True
        return self.name != value.name

    def __add__(self, value: Dimension | str):
        if isinstance(value, Dimension):
            dim = value
        elif isinstance(value, str):
            dim = SingleDimension(value)
        else:
            raise ValueError()
        if isinstance(self, MultiDimension):
            return MultiDimension(*self.dimensions, dim)
        if isinstance(self, SingleDimension):
            return MultiDimension(self, dim)
        raise ValueError()


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


def dim(name: str):
    return SingleDimension(name)


class MultiDimension(Dimension):
    __slots__ = ["_name", "_dimensions"]

    def __init__(self, *dimensions: Dimension):
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


def dims(*names_or_dims: str | Dimension):
    dimensions = []
    for name_or_dim in names_or_dims:
        if isinstance(name_or_dim, Dimension):
            dimensions.append(name_or_dim)
        elif isinstance(name_or_dim, str):
            dimensions.append(Dimension(name_or_dim))
        else:
            raise ValueError()
    return MultiDimension(*dimensions)

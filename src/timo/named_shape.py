from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.named_axis import NamedAxis
    from timo.sized_named_axis import SizedNamedAxis
    from numpy import ndarray


class NamedShape:
    __slots__ = ["_sizes"]

    def __init__(self, *sizes: SizedNamedAxis):
        from numpy import asarray

        self._sizes: ndarray[SizedNamedAxis] = asarray(sizes)

    @property
    def sizes(self):
        return self._sizes

    def moveaxis(
        self,
        source: int | tuple[int, ...] | str | NamedAxis | tuple[str, ...] | tuple[NamedAxis, ...],
        destination: int | tuple[int, ...] | Position | tuple[Position, ...],
    ):
        from timo.named_axis import NamedAxis

        if isinstance(source, tuple):
            if len(source) == 0:
                raise ValueError()
            if len(set(map(type, source))) > 1:
                raise ValueError()
            if isinstance(source[0], str) or isinstance(source[0], NamedAxis):
                source = tuple(map(self.indexof, source))
        elif isinstance(source, str) or isinstance(source, NamedAxis):
            source = self.indexof(source)

        moved_sizes = list(self._sizes)
        if isinstance(source, tuple):
            for s, d in zip(source, destination):
                if isinstance(d, Position):
                    d = d.position(self)
                if d < 0:
                    d = len(moved_sizes) + d + 1

                moved_sizes.insert(d, moved_sizes.pop(s))
        else:
            if isinstance(destination, Position):
                destination = destination.position(self)
            if destination < 0:
                destination = len(moved_sizes) + destination
            moved_sizes.insert(destination, moved_sizes.pop(source))
        return NamedShape(*moved_sizes)

    def remove(self, axis: str | NamedAxis, raise_if_not_found: bool = True):
        from timo.named_axis import name

        axis = name(axis)
        sizes = []
        found = False
        for size in self._sizes:
            if size.axis == axis:
                found = True
            sizes.append(size)
        if raise_if_not_found and not found:
            raise ValueError("Axis not found")
        return NamedShape(*sizes)

    def resize(self, new_size: SizedNamedAxis):
        index = self.indexof(new_size.axis)
        sizes = self._sizes.copy()
        sizes[index] = new_size
        return NamedShape(*sizes)

    def indexof(self, axis: str | NamedAxis):
        from timo.named_axis import name

        axis = name(axis)
        for i, size in enumerate(self._sizes):
            if size.axis == axis:
                return i
        raise ValueError()

    def before(self, *axes: str | NamedAxis):
        return self._sizes[: min(map(self.indexof, axes))]

    def after(self, *axes: str | NamedAxis):
        return self._sizes[max(map(self.indexof, axes)) + 1 :]

    def __eq__(self, value):
        if isinstance(value, str):
            return str(self) == value

        if not isinstance(value, NamedShape):
            return False

        if len(self._sizes) != len(value._sizes):
            return False

        return (self._sizes == value._sizes).all()

    def __ne__(self, value):
        return not self.__eq__(value)

    def __getitem__(self, index: int | str | NamedAxis) -> SizedNamedAxis:
        from timo.named_axis import name

        if isinstance(index, int):
            return self._sizes[index]
        index = name(index)
        for size in self._sizes:
            if size.axis == index:
                return size
        raise IndexError(f"Dimension: `{index}` not in shape {self}")

    def __str__(self):
        return "".join(map(str, self.sizes))

    def __repr__(self):
        return str(self)


class Position:
    def position(self, shape: NamedShape):
        raise NotImplementedError()


class After(Position):
    def __init__(self, axis: str | NamedAxis):
        from timo.named_axis import name

        self._axis = name(axis)

    def position(self, shape: NamedShape):
        return shape.indexof(self._axis)


class Before(Position):
    def __init__(self, axis: str | NamedAxis):
        from timo.named_axis import name

        self._axis = name(axis)

    def position(self, shape: NamedShape):
        return shape.indexof(self._axis) - 1


def shape(*values: tuple[str | NamedAxis, int | None] | NamedAxis | str | SizedNamedAxis | NamedShape):
    from timo.named_axis import NamedAxis
    from timo.sized_named_axis import SizedNamedAxis

    sizes = []
    for value in values:
        if isinstance(value, str):
            sizes.append(SizedNamedAxis(value, None))
        elif isinstance(value, tuple):
            if len(value) == 2:
                sizes.append(SizedNamedAxis(value[0], value[1]))
            elif len(value) == 1:
                sizes.append(SizedNamedAxis(value[0], None))
        elif isinstance(value, NamedAxis) or isinstance(value, str):
            sizes.append(SizedNamedAxis(value, None))
        elif isinstance(value, SizedNamedAxis):
            sizes.append(value)
        elif isinstance(value, NamedShape):
            sizes.extend(value._sizes)
        else:
            raise ValueError()
    return NamedShape(*sizes)

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.named_axis import NamedAxis


class SizedNamedAxis:
    __slots__ = ["_axis", "_size"]

    def __init__(self, axis: str | NamedAxis, size: int | None):
        from timo.named_axis import name

        self._axis = name(axis)
        self._size = size

    @property
    def axis(self) -> NamedAxis:
        return self._axis

    @property
    def size(self) -> int | None:
        return self._size

    @property
    def set_size(self):
        count = self._size
        if count is None:
            raise ValueError()
        return count

    @property
    def is_set(self):
        return self._size is not None

    def __str__(self):
        return f"{self._axis}{self._size or '*'}"

    def __repr__(self):
        return str(self)

    def __eq__(self, value):
        if not isinstance(value, SizedNamedAxis):
            return False
        return self._axis == value._axis and self._size == value._size

    def __ne__(self, value):
        if not isinstance(value, SizedNamedAxis):
            return True

        return self._axis != value._axis or self._size != value._size


def size(axis: NamedAxis | str, size: int | None = None):
    from timo.named_axis import name

    return SizedNamedAxis(name(axis), size)

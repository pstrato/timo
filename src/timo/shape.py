from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.size import Size


class Shape:
    __slots__ = ["_sizes"]

    def __init__(self, *sizes: Size):
        self._sizes = sizes

    @property
    def sizes(self):
        return self._sizes

    def __add__(self, value: Size | Shape):
        from timo.size import Size

        if isinstance(value, Size):
            return Shape(*self.sizes, value)
        if isinstance(value, Shape):
            return Shape(*self.sizes, *value.sizes)
        raise ValueError()

    def __eq__(self, value):
        if not isinstance(value, Shape):
            return False

        return self.sizes == value.sizes

    def __ne__(self, value):
        if not isinstance(value, Shape):
            return True

        return self.sizes != value.sizes

    def __getitem__(self, index: int):
        return self.sizes[index]


def shape(*sizes: Size):
    return Shape(*sizes)

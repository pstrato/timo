from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.shape import Shape


class Shapes:
    __slots__ = ["_shapes"]

    def __init__(self, *shapes: Shape):
        assert len(shapes) > 0
        self._shapes = shapes

    def __str__(self):
        return ",".join(map(str, self._shapes))

    def __repr__(self):
        return str(self)

    @property
    def shapes(self):
        return self._shapes

    @property
    def __getitem__(self, index: int):
        return self._shapes[index]

    def single_shape(self):
        if len(self._shapes) != 1:
            raise ValueError()
        return self._shapes[0]


def shapes(*shapes: Shape):
    from timo.shape import Shape, shape

    all_shapes = []
    for s in shapes:
        if isinstance(s, Shapes):
            all_shapes.extend(s.shapes)
        elif isinstance(s, Shape):
            all_shapes.append(s)
        else:
            all_shapes.append(shape(s))

    return Shapes(*all_shapes)

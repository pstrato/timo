from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.named_shape import NamedShape


class NamedShapeSequence:
    __slots__ = ["_shapes"]

    def __init__(self, *shapes: NamedShape):
        assert len(shapes) > 0
        self._shapes = shapes

    def __str__(self):
        return ",".join(map(str, self._shapes))

    def __repr__(self):
        return str(self)

    def __eq__(self, value):
        from timo.named_shape import NamedShape

        if isinstance(value, str):
            return str(self) == value
        if isinstance(value, tuple):
            value = shapes(*value)
        if isinstance(value, NamedShape):
            value = shapes(value)

        if not isinstance(value, NamedShapeSequence):
            return False

        if len(self._shapes) != len(value._shapes):
            return False

        for shape, other_shape in zip(self._shapes, value._shapes):
            if shape != other_shape:
                return False

        return True

    @property
    def shapes(self):
        return self._shapes

    @property
    def __getitem__(self, index: int) -> NamedShape:
        return self._shapes[index]

    def single_shape(self):
        if len(self._shapes) != 1:
            raise ValueError()
        return self._shapes[0]


def shapes(*shapes: NamedShape):
    from timo.named_shape import NamedShape, shape

    all_shapes = []
    for s in shapes:
        if isinstance(s, NamedShapeSequence):
            all_shapes.extend(s.shapes)
        elif isinstance(s, NamedShape):
            all_shapes.append(s)
        else:
            all_shapes.append(shape(s))

    return NamedShapeSequence(*all_shapes)

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.shape_sequence import ShapeSequence
    from timo.transform import Transform
    from flax.nnx.rnglib import Rngs
    from flax.typing import Initializer

_unset = object()


class TransformContext:
    def __init__(self, parent: TransformContext | None = None, **args):
        self.parent = parent
        self.args = args

    def get(self, name, default=_unset):
        arg = self.args.get(name, _unset)
        if arg is _unset:
            if self.parent is not None:
                arg = self.parent.get(name, default)
            else:
                if default is _unset:
                    raise ValueError(f"Not set: {name}")
                return default
        return arg

    @property
    def input_shapes(self) -> ShapeSequence:
        return self.get("input_shapes")

    @property
    def rngs(self) -> Rngs:
        return self.get("rngs")

    def initializer(self, kind, default=_unset) -> Initializer:
        return self.get(("initializer", kind), default)

    def __call__(self, transform: Transform):
        return TransformContext(self, input_shapes=transform.output_shapes)

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.named_shape_sequence import NamedShapeSequence
    from timo.transform_factory import TransformFactory
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
    def input_shapes(self) -> NamedShapeSequence:
        return self.get("input_shapes")

    @property
    def rngs(self) -> Rngs:
        return self.get("rngs")

    def initializer(self, transform: TransformFactory, kind, default=_unset) -> Initializer:
        return self.get(("initializer", type(transform), kind), default)

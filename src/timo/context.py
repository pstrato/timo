from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.named_axis import NamedAxis
    from timo.named_shape_sequence import NamedShapeSequence
    from timo.factory import Factory
    from flax.nnx.rnglib import Rngs
    from flax.typing import Initializer

from flax.nnx import Param

_unset = object()


class Context:
    def __init__(self, parent: Context | None = None, **args):
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

    def in_size(self, on: str | NamedAxis):
        return self.input_shapes.single_shape()[on].set_size

    @property
    def input_shapes(self) -> NamedShapeSequence:
        return self.get("input_shapes")

    @property
    def rngs(self) -> Rngs:
        return self.get("rngs")

    def initializer(self, factory: Factory, kind, default=_unset) -> Initializer:
        return self.get(("initializer", type(factory), kind), default)

    def params(self, kind: str, shape: int | tuple[int, ...], default_init: Initializer):
        return Param(self.initializer(self, kind, default_init)(self.rngs.params(), shape))

    def push(self, factory: Factory):
        return Context(self, input_shapes=factory.output_shapes)

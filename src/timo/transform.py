from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flax.nnx import Param
    from typing import Callable
    from timo.info import Info
    from timo.out import Out

from typing import TypeVar, Generic

from jax import Array

I = TypeVar("I", bound=Array | tuple[Array, ...])
O = TypeVar("O", bound=Array | tuple[Array, ...])

from flax.nnx import Module


class Transform(Module, Generic[I, O]):
    def __init__(self, transform: Callable[[I, Info, Out], O], **params: Param):
        Module.__init__(self)
        self.transform = transform
        for name, param in params.items():
            setattr(self, name, param)
        self.args = params

    def __call__(self, inputs: I, info: Info | None = None, out: Out | None = None) -> O:
        return self.transform(inputs, info=info, out=out, **self.args)

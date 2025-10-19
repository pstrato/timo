from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable, Any
    from timo.named_shape import NamedShape
    from timo.named_shape_sequence import NamedShapeSequence
    from timo.context import Context

from typing import Generic, TypeVar

I = TypeVar("I")
O = TypeVar("O")

from flax import nnx


class Transform(nnx.Module, Generic[I, O]):
    def __init__(
        self,
        transform: Callable,
        ctx: Context,
        output_shapes: NamedShape | NamedShapeSequence | None = None,
        data: dict[str, Any] = {},
        static: dict[str, Any] = {},
    ):
        from timo.named_shape_sequence import shapes
        from timo.context import Context

        nnx.Module.__init__(self)
        self.transform = nnx.static(transform)
        self.training = True
        self.data = nnx.data(data)
        self.input_ctx = nnx.data(ctx)
        self.output_ctx = nnx.data(Context(ctx, input_shapes=shapes(output_shapes or ctx.input_shapes)))
        self.static = nnx.static(static)

    @property
    def input_shapes(self):
        return self.input_ctx.input_shapes

    @property
    def output_shapes(self):
        return self.output_ctx.input_shapes

    def train(self, **attributes):
        return super().train(**attributes, training=True)

    def eval(self, **attributes):
        return super().eval(**attributes, training=False)

    def __call__(self, inputs: I, data: nnx.Dict | None = None) -> O:
        return self.transform(inputs, data, **self.static, **self.data)

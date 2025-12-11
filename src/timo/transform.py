from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable, Any
    from timo.named_shape import NamedShape
    from timo.named_shape_sequence import NamedShapeSequence
    from timo.context import Context
    from timo.factory import Factory

from timo.out import Out

from typing import Generic, TypeVar

I = TypeVar("I")
O = TypeVar("O")

from flax import nnx


class Transform(nnx.Module, Generic[I, O]):
    def __init__(
        self,
        transform: Callable,
        ctx: Context,
        factory: Factory,
        output_shapes: NamedShape | NamedShapeSequence | None = None,
        data: dict[str, Any] = {},
        static: dict[str, Any] = {},
        with_out: bool = False,
    ):
        from timo.named_shape_sequence import shapes
        from timo.context import Context

        nnx.Module.__init__(self)
        self.transform = nnx.static(transform)
        self.factory = nnx.static(factory)
        self.with_out = nnx.static(with_out)
        self.input_ctx = nnx.static(ctx)
        self.output_ctx = nnx.static(Context(ctx, input_shapes=shapes(output_shapes or ctx.input_shapes)))
        self.args = []
        for arg, value in data.items():
            setattr(self, arg, nnx.data(value))
            self.args.append(arg)
        for arg, value in static.items():
            setattr(self, arg, nnx.static(value))
            self.args.append(arg)
        if with_out:
            self.args.append("out")
        self.training = True
        self.out = nnx.data(None)

    def set_out(self, out: Out | None):
        self.set_attributes(out=out, raise_if_not_found=False)

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

    def __call__(self, inputs: I) -> O:
        args = {arg: getattr(self, arg) for arg in self.args}
        return self.transform(inputs, **args)

    def create_out(self) -> Out:
        outs = self.input_ctx.out_keys
        if outs is not None:
            return Out(outs)
        else:
            raise ValueError()

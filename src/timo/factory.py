from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable
    from timo.named_axis import NamedAxis
    from timo.named_shape import NamedShape
    from timo.named_shape_sequence import NamedShapeSequence
    from timo.transform import Transform
    from timo.context import Context
    from timo.observer import Observer

from typing import Generic, TypeVar

I = TypeVar("I")
O = TypeVar("O")

from flax.nnx import vmap


class Factory(Generic[I, O]):
    def __init__(self):
        super().__init__()
        self._input_ctx: Context | None = None
        self._output_ctx: Context | None = None

    @property
    def input_ctx(self):
        if self._input_ctx is None:
            raise ValueError("Transform input context not set")
        return self._input_ctx

    @property
    def input_shapes(self):
        return self.input_ctx.input_shapes

    @property
    def output_ctx(self):
        if self._output_ctx is None:
            raise ValueError("Transform output context not set")
        return self._output_ctx

    @property
    def output_shapes(self):
        return self.output_ctx.input_shapes

    def transform(self, ctx: Context) -> Transform[I, O]:
        if self._input_ctx is not None:
            raise RuntimeError("Transform already created")

        self._input_ctx = ctx
        module = self.create_transform(ctx)
        self._output_ctx = module.output_ctx
        return module

    def create_transform(self, ctx: Context) -> Transform:
        raise NotImplementedError()

    def vmap(self, function: Callable, non_mapped_args: tuple, *on: str | NamedAxis):
        for _ in self.input_shapes.single_shape().before(*on):
            function = vmap(function, in_axes=(0, *non_mapped_args), out_axes=0)
        for _ in self.input_shapes.single_shape().after(*on):
            function = vmap(function, in_axes=(-1, *non_mapped_args), out_axes=-1)
        return function

    def __rshift__(self, value: Factory):
        from timo.transforms.sequential import Sequential

        transforms = []
        if isinstance(self, Sequential):
            transforms.extend(self.transforms)
        else:
            transforms.append(self)
        if isinstance(value, Sequential):
            transforms.extend(value.transforms)
        else:
            transforms.append(value)

        return Sequential(*transforms)

    def __add__(self, observer: Observer) -> Factory:
        from timo.observer import ObserverFactory

        return ObserverFactory[I, O](self, observer)

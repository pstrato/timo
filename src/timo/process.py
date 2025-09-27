from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.context import Context
    from timo.transform import Transform

from timo.factory import Factory


class Process:
    def module(self, ctx: Context, factory: Factory) -> Transform:
        raise NotImplementedError()


class ProcessFactory(Factory):
    def __init__(self, process: Process, factory: Factory):
        super().__init__()
        self.process = process
        self.factory = factory

    @property
    def input_shapes(self):
        return self.factory.input_shapes

    @property
    def output_shapes(self):
        return self.factory.output_shapes

    def transform(self, ctx: Context) -> Transform:
        return self.process.module(ctx, self.factory)

    def __add__(self, process: Process):
        return ProcessFactory(process, self)

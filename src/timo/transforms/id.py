from __future__ import annotations
from timo.transform import Transform
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.transform_context import TransformContext
    from timo.info import Info
    from timo.out import Out


class Id(Transform):
    def __init__(self, ctx: TransformContext):
        super().__init__(ctx, ctx.input_shapes)

    def transform(self, inputs, info: Info, out: Out):
        return inputs

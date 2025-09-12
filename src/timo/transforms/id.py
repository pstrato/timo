from timo.transform import Transform
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.transform_context import TransformContext


class Id(Transform):
    def __init__(self, ctx: TransformContext):
        super().__init__()
        self._set_shapes(ctx.input_shapes, ctx.input_shapes)

    def __call__(self, inputs):
        return inputs

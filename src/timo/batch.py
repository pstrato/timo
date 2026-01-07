from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Self
    from timo.out import Out
    from timo.transform import Transform

from flax import nnx


class Batch(nnx.Pytree):
    def __init__(
        self,
        inputs=None,
        targets=None,
        step: str | None = None,
        epoch: int | None = None,
        index: int | None = None,
        load_time: float | None = None,
        out: Out | None = None,
        model: Transform | None = None,
    ) -> None:
        super().__init__()
        self.inputs = nnx.data(inputs)
        self.targets = nnx.data(targets)
        self.out = nnx.data(out)
        self.step = nnx.static(step)
        self.epoch = nnx.static(epoch)
        self.index = nnx.static(index)
        self.load_time = nnx.static(load_time)
        self.model = nnx.data(model)

    def clone(self, data: dict | None = None, metadata: dict | None = None) -> Self:
        data = data or {}
        metadata = metadata or {}

        inputs = data.get("inputs", self.inputs)
        targets = data.get("targets", self.targets)
        out = data.get("out", self.out)
        model = data.get("model", self.model)

        step = metadata.get("step", self.step)
        epoch = metadata.get("epoch", self.epoch)
        index = metadata.get("index", self.index)
        load_time = metadata.get("load_time", self.load_time)
        return Batch(inputs, targets, step, epoch, index, load_time, out, model)  # type: ignore

    @staticmethod
    def as_batch(
        items: list[Batch],
        step: str | None = None,
        epoch: int | None = None,
        index: int | None = None,
        load_time: float | None = None,
    ):
        return Batch(
            stack(map(lambda i: i.inputs, items)),
            stack(map(lambda i: i.targets, items)),
            step,
            epoch,
            index,
            load_time,
            None,
            None,
        )


def as_list(values):
    return list(values)


def stack(values):
    from jax.numpy import stack

    values = list(values)

    if all(map(lambda v: v is None, values)):
        return None
    if len(values) == 0:
        return None

    return stack(values)

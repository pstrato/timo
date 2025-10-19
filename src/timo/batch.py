from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from flax import nnx


class Batch(nnx.Module):
    def __init__(
        self,
        inputs=None,
        targets=None,
        step: str | None = None,
        epoch: int | None = None,
        index: int | None = None,
        load_time: float | None = None,
    ) -> None:
        super().__init__()
        self.inputs = inputs
        self.targets = targets
        self.step = step
        self.epoch = epoch
        self.index = index
        self.load_time = load_time

    def clone(self, data: dict | None = None, metadata: dict | None = None):
        if data is not None:
            inputs = data.get("inputs") or self.inputs
            targets = data.get("targets") or self.targets
        else:
            inputs = self.inputs
            targets = self.targets
        if metadata is not None:
            step = metadata.get("step") or self.step
            epoch = metadata.get("epoch") or self.epoch
            index = metadata.get("index") or self.index
            load_time = metadata.get("load_time") or self.load_time
        else:
            step = self.step
            epoch = self.epoch
            index = self.index
            load_time = self.load_time
        return Batch(inputs, targets, step, epoch, index, load_time)

    @staticmethod
    def as_batch(
        items: list[Batch],
        step: str | None = None,
        epoch: int | None = None,
        index: int | None = None,
        load_time: float | None = None,
    ):
        return Batch(
            stack(map(lambda i: i.inputs, items)), stack(map(lambda i: i.targets, items)), step, epoch, index, load_time
        )


def as_list(values):
    return list(values)


def stack(values):
    from jax.numpy import stack

    return stack(list(values))

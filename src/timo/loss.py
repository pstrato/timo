from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable
    from jax import Array
    from timo.out import Out

from jax import numpy as jnp
from flax import nnx
from timo.accumulator import Accumulator


class Loss(nnx.Module):

    def __call__(self, out: Out, accumulator: Accumulator | None = None, weight: float = 1) -> Accumulator:
        raise NotImplementedError()

    def __mul__(self, weight: float):
        return WeightedLoss(self, weight)

    def __add__(self, loss: Loss):
        losses = []
        if isinstance(self, CombinedLoss):
            losses.extend(self.losses)
        else:
            losses.append(self)
        if isinstance(loss, CombinedLoss):
            losses.extend(loss.losses)
        else:
            losses.append(loss)
        return CombinedLoss(*losses)

    def __floordiv__(self, *weighted_losses: tuple[Loss, float]):
        return ProportionalLoss(self, *weighted_losses)


class WeightedLoss(Loss):
    def __init__(self, loss: Loss, weight: float):
        super().__init__()
        self.loss = nnx.static(loss)
        self.weight = nnx.static(weight)

    def __call__(self, out, accumulator=None, weight=1):
        return self.loss(out, accumulator=accumulator, weight=weight * self.weight)


class CombinedLoss(Loss):
    def __init__(self, *losses: Loss):
        super().__init__()
        self.losses = nnx.data(losses)

    def __call__(self, out, accumulator=None, weight=1):
        if accumulator is None:
            accumulator = Accumulator()
        for loss in self.losses:
            loss(out, accumulator, weight)
        return accumulator


class ProportionalLoss(Loss):
    def __init__(self, loss: Loss, *weighted_losses: tuple[Loss, float]):
        self.main = nnx.static(loss)
        self.weighted = nnx.static(weighted_losses)

    def __call__(self, out, accumulator=None, weight=1):

        if accumulator is None:
            accumulator = Accumulator()

        main_accumulator = Accumulator()
        self.main(out, main_accumulator, weight)
        accumulator.add_accumulator(main_accumulator)

        main_loss = main_accumulator.detached_mean()
        for weighted, weighted_weight in self.weighted:
            weighted_accumulator = Accumulator()
            weighted(out, weighted_accumulator, 1)
            weighted_loss = weighted_accumulator.detached_mean()
            weighted_scale = main_loss * weighted_weight / weighted_loss
            accumulator.add_accumulator(weighted_accumulator, weighted_scale)
        return accumulator


class ValueLoss(Loss):
    def __init__(
        self,
        target: Callable[[Out], Array],
        output: Callable[[Out], Array],
        function: Callable[[Array, Array], Array],
        key: str,
    ) -> None:
        super().__init__()
        self.target = nnx.static(target)
        self.output = nnx.static(output)
        self.function = nnx.static(function)
        self.key = nnx.static(key)

    def __call__(self, out: Out, accumulator: Accumulator | None = None, weight: float = 1) -> Accumulator:
        if accumulator is None:
            accumulator = Accumulator()
        target = self.target(out)
        output = self.output(out)
        loss = self.function(target, output)
        accumulator.add_value(loss, self.key)
        return accumulator


def inputs(out: Out):
    return out.inputs


def targets(out: Out):
    return out.targets


def outputs(out: Out):
    return out.outputs


def data(key: str):
    def value(out: Out):
        return getattr(out, key)

    return value


def constant(constant: int | float | Array):
    def value(out: Out):
        return constant

    return value


def rmse(targets: Array, outputs: Array):
    residuals = targets - outputs
    return jnp.sqrt((residuals**2).sum(axis=-1))

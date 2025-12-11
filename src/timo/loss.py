from __future__ import annotations
from typing import TYPE_CHECKING

import jax

if TYPE_CHECKING:
    from typing import Callable
    from jax import Array
    from timo.out import Out
    from timo.factory import Factory

from jax import numpy as jnp
from flax import nnx
from timo.accumulator import Accumulator
from timo.transform import Transform


class Loss(nnx.Module):

    def create_accumulator(self) -> Accumulator:
        raise NotImplementedError()

    def __call__(
        self, model: Transform, out: Out, rngs: nnx.Rngs, accumulator: Accumulator | None = None, weight: float = 1
    ) -> Accumulator:
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
        self.loss = nnx.data(loss)
        self.weight = nnx.static(weight)

    def create_accumulator(self) -> Accumulator:
        return self.loss.create_accumulator()

    def __call__(self, model: Transform, out, rngs: nnx.Rngs, accumulator=None, weight=1):
        return self.loss(model, out, rngs, accumulator=accumulator, weight=weight * self.weight)


class CombinedLoss(Loss):
    def __init__(self, *losses: Loss):
        super().__init__()
        self.losses = nnx.data(losses)
        keys = set()
        for loss in self.losses:
            accumulator = loss.create_accumulator()
            keys.update(accumulator.keys)
        self.keys = keys

    def create_accumulator(self) -> Accumulator:
        return Accumulator(self.keys)

    def __call__(self, model: Transform, out, rngs: nnx.Rngs, accumulator=None, weight=1):
        if accumulator is None:
            accumulator = self.create_accumulator()
        for loss in self.losses:
            loss(model, out, rngs, accumulator, weight)
        return accumulator


class ProportionalLoss(Loss):
    def __init__(self, loss: Loss, *weighted_losses: tuple[Loss, float]):
        self.main = nnx.static(loss)
        self.weighted = nnx.static(weighted_losses)
        keys = set()
        for loss, _ in self.weighted:
            accumulator = loss.create_accumulator()
            keys.update(accumulator.keys)
        accumulator = self.main.create_accumulator()
        keys.update(accumulator.keys)
        self.keys = keys

    def create_accumulator(self) -> Accumulator:
        return Accumulator(self.keys)

    def __call__(self, model, out, rngs: nnx.Rngs, accumulator=None, weight=1):

        if accumulator is None:
            accumulator = self.create_accumulator()

        main_accumulator = self.create_accumulator()
        self.main(model, out, rngs, main_accumulator, weight)
        accumulator.add_accumulator(main_accumulator)

        main_loss = main_accumulator.detached_mean()
        for weighted, weighted_weight in self.weighted:
            weighted_accumulator = self.create_accumulator()
            weighted(model, out, rngs, weighted_accumulator, 1)
            weighted_loss = weighted_accumulator.detached_mean()
            weighted_scale = main_loss * weighted_weight / weighted_loss
            accumulator.add_accumulator(weighted_accumulator, weighted_scale)
        return accumulator


class ValueLoss(Loss):
    def __init__(
        self,
        target: Callable[[Out], Array] | tuple[Callable[[Out], Array], ...],
        output: Callable[[Out], Array],
        function: Callable[[Array | tuple[Array, ...], Array], Array],
        key: str,
        static: dict = {},
    ) -> None:
        super().__init__()
        self.target = nnx.static(target)
        self.output = nnx.static(output)
        self.function = nnx.static(function)
        self.key = nnx.static(key)
        self.static = nnx.static(static)

    def create_accumulator(self) -> Accumulator:
        return Accumulator({self.key})

    def __call__(
        self, model, out: Out, rngs: nnx.Rngs, accumulator: Accumulator | None = None, weight: float = 1
    ) -> Accumulator:
        if accumulator is None:
            accumulator = self.create_accumulator()
        if isinstance(self.target, tuple):
            target = [t(out) for t in self.target]
        else:
            target = self.target(out)
        output = self.output(out)
        loss = self.function(target, output, **self.static)
        accumulator.add_value(loss, self.key, weight)
        return accumulator


class SelfLoss(Loss):
    def __init__(
        self,
        output: Callable[[Out], Array],
        function: Callable[[Array], Array],
        key: str,
    ) -> None:
        super().__init__()
        self.output = nnx.static(output)
        self.function = nnx.static(function)
        self.key = nnx.static(key)

    def create_accumulator(self) -> Accumulator:
        return Accumulator({self.key})

    def __call__(
        self, model, out: Out, rngs: nnx.Rngs, accumulator: Accumulator | None = None, weight: float = 1
    ) -> Accumulator:
        if accumulator is None:
            accumulator = self.create_accumulator()
        output = self.output(out)
        loss = self.function(output)
        accumulator.add_value(loss, self.key, weight)
        return accumulator


class TransformLoss(Loss):
    def __init__(
        self,
        transform: Callable[[Transform], Transform | None],
        function: Callable[[Transform, nnx.Rngs], Array],
        key: str,
    ):
        self.transform = nnx.static(transform)
        self.function = nnx.static(function)
        self.key = nnx.static(key)

    def create_accumulator(self) -> Accumulator:
        return Accumulator({self.key})

    def __call__(self, model, out: Out, rngs: nnx.Rngs, accumulator: Accumulator | None = None, weight: float = 1):
        if accumulator is None:
            accumulator = self.create_accumulator()
        transform = self.transform(model)
        if transform is None:
            return accumulator
        loss = self.function(transform, rngs)
        accumulator.add_value(loss, self.key, weight)
        return accumulator


def model_factories(model: Transform):
    for _, t in model.iter_modules():
        if isinstance(t, Transform):
            yield t.factory


def transform_of_factory(factory: Factory):
    def transform(model: Transform):
        for _, t in model.iter_modules():
            if isinstance(t, Transform) and t.factory is factory:
                return t

    return transform


def inputs(out: Out):
    return out.inputs


def targets(out: Out):
    return out.targets


def outputs(out: Out):
    return out.outputs


def out(key: str):
    def value(out: Out):
        return getattr(out, key)

    return value


def constant(constant: int | float | Array):
    constant_array = jax.numpy.asarray(constant)
    assert constant_array is not None

    def value(out: Out):
        return constant_array

    return value


def rmse(targets: Array, outputs: Array):
    residuals = targets - outputs
    return jnp.sqrt((residuals**2).sum(axis=-1))

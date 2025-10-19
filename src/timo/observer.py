from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.context import Context
    from timo.transform import Transform

from timo.factory import Factory
from timo.transform import Transform, I, O
from jax.lax import stop_gradient
from flax.nnx import Dict


def ref(key: str, on_train: bool = True, on_eval: bool = True):
    return Observer(reference_outputs, on_train, on_eval, key)


def detach(key, on_train: bool = True, on_eval: bool = True):
    return Observer(detach_outputs, on_train, on_eval, key)


def copy(key, on_train: bool = True, on_eval: bool = True):
    return Observer(copy_outputs, on_train, on_eval, key)


def reference_outputs(inputs, data: Dict, observed: Transform, key: str, on_train: bool, on_eval: bool):
    outputs = observed(inputs, data)
    training = observed.training
    if on_train and training or (not training and on_eval):
        data[key] = outputs
    return outputs


def detach_outputs(inputs, data: Dict, observed: Transform, key: str, on_train: bool, on_eval: bool):
    outputs = observed(inputs, data)
    training = observed.training
    if (on_train and training) or (not training and on_eval):
        data[key] = observed(stop_gradient(inputs), data)
    return outputs


def copy_outputs(inputs, data: Dict, observed: Transform, key: str, on_train: bool, on_eval: bool):
    outputs = observed(inputs, data)
    training = observed.training
    if (on_train and training) or (not training and on_eval):
        data[key] = observed(inputs, data)
    return outputs


class Observer:
    def __init__(self, action, on_train: bool, on_eval: bool, key):
        self.on_train = on_train
        self.on_eval = on_eval
        self.action = action
        self.key = key


class ObserverFactory(Factory[I, O]):
    def __init__(self, factory: Factory[I, O], observer: Observer):
        self.factory = factory
        self.observer = observer

    def create_transform(self, ctx: Context):
        observed = self.factory.transform(ctx)
        action = self.observer.action
        on_train = self.observer.on_train
        on_eval = self.observer.on_eval
        keys = self.observer.key
        return Transform(
            action,
            ctx,
            ctx.input_shapes,
            data={"observed": observed},
            static={"keys": keys, "on_train": on_train, "on_eval": on_eval},
        )

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.context import Context
    from timo.transform import Transform
    from timo.out import Out

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


def reference_outputs(inputs, out: Out, observed: Transform, key: str, on_train: bool, on_eval: bool):
    outputs = observed(inputs, out)
    training = observed.training
    if on_train and training or (not training and on_eval):
        setattr(out, key, outputs)
    return outputs


def detach_outputs(inputs, out: Out, observed: Transform, key: str, on_train: bool, on_eval: bool):
    outputs = observed(inputs, out)
    training = observed.training
    if (on_train and training) or (not training and on_eval):
        setattr(out, key, observed(stop_gradient(inputs), out))
    return outputs


def copy_outputs(inputs, out: Out, observed: Transform, key: str, on_train: bool, on_eval: bool):
    outputs = observed(inputs, out)
    training = observed.training
    if (on_train and training) or (not training and on_eval):
        setattr(out, key, observed(inputs, out))
    return outputs


class Observer:
    def __init__(self, action, on_train: bool, on_eval: bool, key):
        self.on_train = on_train
        self.on_eval = on_eval
        self.action = action
        self.key = key


class ObserverFactory(Factory[I, O]):
    factory: Factory[I, O]
    observer: Observer

    def create_transform(self, ctx: Context):
        observed = self.factory.transform(ctx)
        action = self.observer.action
        on_train = self.observer.on_train
        on_eval = self.observer.on_eval
        key = self.observer.key
        ctx.add_out(key)
        return Transform(
            action,
            ctx,
            observed.output_shapes,
            data={"observed": observed},
            static={"key": key, "on_train": on_train, "on_eval": on_eval},
        )

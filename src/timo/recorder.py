from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.context import Context
    from timo.transform import Transform

from timo.factory import Factory
from timo.transform import Transform, I, O
from jax.lax import stop_gradient


def ref(key: str, on_train: bool = True, on_eval: bool = True):
    return Recorder(reference_outputs, on_train, on_eval, key)


def detach(key, on_train: bool = True, on_eval: bool = True):
    return Recorder(detach_outputs, on_train, on_eval, key)


def copy(key, on_train: bool = True, on_eval: bool = True):
    return Recorder(copy_outputs, on_train, on_eval, key)


def reference_outputs(inputs, observed: Transform, key: str, on_train: bool, on_eval: bool):
    outputs = observed(inputs)
    training = observed.training
    if on_train and training or (not training and on_eval):
        setattr(observed.out, key, outputs)
    return outputs


def detach_outputs(inputs, observed: Transform, key: str, on_train: bool, on_eval: bool):
    outputs = observed(inputs)
    training = observed.training
    if (on_train and training) or (not training and on_eval):
        setattr(observed.out, key, observed(stop_gradient(inputs)))
    return outputs


def copy_outputs(inputs, observed: Transform, key: str, on_train: bool, on_eval: bool):
    outputs = observed(inputs)
    training = observed.training
    if (on_train and training) or (not training and on_eval):
        setattr(observed.out, key, observed(inputs))
    return outputs


class Recorder:
    def __init__(self, action, on_train: bool, on_eval: bool, key):
        self.on_train = on_train
        self.on_eval = on_eval
        self.action = action
        self.key = key


class RecorderFactory(Factory[I, O]):
    factory: Factory[I, O]
    observer: Recorder

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

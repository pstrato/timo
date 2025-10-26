from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.transform import Transform
    from timo.batch import Batch
    from timo.loader import Loader
    from timo.loss import Loss
    from timo.out import Out

from timo.accumulator import Accumulator
from flax import nnx

from timo.loader import B


class Epoch:
    def __init__(self, step: str, epoch: int, times: Accumulator, losses: Accumulator, state):
        self.step = step
        self.epoch = epoch
        self.times = times
        self.losses = losses
        self.state = state

    def mean_summary(self):
        return " ".join(map(lambda m: f"{m[0]}:{float(m[1])}", self.losses.means()))

    def time_summary(self):
        return " ".join(map(lambda t: f"{t[0]}:{float(t[1])}s", self.times.sums()))

    def __str__(self) -> str:
        return f"#{self.epoch}/{self.step}\n{self.mean_summary()}\n{self.time_summary()}"

    def __repr__(self) -> str:
        return str(self)


class StopCondition(nnx.Module):
    def __call__(self, train: Epoch, eval: Epoch) -> bool:
        raise NotImplementedError()


class StopAfterEpoch(StopCondition):
    def __init__(self, epoch: int):
        super().__init__()
        self.epoch = epoch

    def __call__(self, train: Epoch, eval: Epoch):
        return train.epoch >= self.epoch


def eval_batch(step_graph, step_state, batch: Batch):
    transform, optimiser, loss = nnx.merge(step_graph, step_state)
    out: Out = transform.create_out()
    out.inputs = batch.inputs
    out.targets = batch.targets
    outputs = transform(batch.inputs, out)
    out.outputs = outputs
    accumulator = loss(out)
    return accumulator


def train_batch(step_graph, step_state, batch: Batch):

    def loss_fn(transform: Transform, loss: Loss):
        out: Out = transform.create_out()
        out.inputs = batch.inputs
        out.targets = batch.targets
        outputs = transform(batch.inputs, out)
        out.outputs = outputs
        accumulator = loss(out)
        return accumulator.mean(), accumulator

    transform, optimizer, loss = nnx.merge(step_graph, step_state)

    (loss_value, accumulator), grad = nnx.value_and_grad(loss_fn, has_aux=True)(transform, loss)

    optimizer.update(transform, grad)

    step_state = nnx.state((transform, optimizer, loss))
    return step_state, accumulator


def fit(
    transform: Transform,
    optimizer: nnx.Optimizer,
    stop_condition: StopCondition,
    loss: Loss,
    train: Loader[B],
    eval: Loader[B],
):
    step_graph, step_state = nnx.split((transform, optimizer, loss))

    epoch = 0
    stop = False
    while not stop:
        epoch += 1

        train_times = Accumulator()
        with train_times.timer("epoch time"):
            train_losses = Accumulator()
            transform.train()
            for batch in train.batches({"step": "train", "epoch": epoch}):
                with train_times.timer("batch compute time"):
                    step_state, batch_losses = train_batch(step_graph, step_state, batch)
                train_losses.add_accumulator(batch_losses)
                train_times.add_value(batch.load_time, "batch load time")
        yield (train_epoch := Epoch("train", epoch, train_times, train_losses, step_state[0]))

        eval_times = Accumulator()
        with eval_times.timer("epoch time"):
            eval_losses = Accumulator()
            for batch in eval.batches({"step": "eval", "epoch": epoch}):
                with eval_times.timer("batch compute time"):
                    batch_losses = eval_batch(step_graph, step_state, batch)
                eval_losses.add_accumulator(batch_losses)
                eval_times.add_value(batch.load_time, "batch load time")
        yield (eval_epoch := Epoch("eval", epoch, eval_times, eval_losses, step_state[0]))

        stop = stop_condition(train_epoch, eval_epoch)

    nnx.update((transform, optimizer, loss), step_state)

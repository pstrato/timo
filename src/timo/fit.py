from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.transform import Transform
    from timo.batch import Batch
    from timo.loader import Loader
    from timo.loss import Loss

from timo.accumulator import Accumulator
from timo.batch_profiler import BatchProfiler, NoBatchProfiler
from flax import nnx
from jax import Array

from timo.loader import B


class Epoch:
    def __init__(self, step: str, epoch: int, times: Accumulator, losses: Accumulator, state, model: Transform):
        self.step = step
        self.epoch = epoch
        self.times = times
        self.losses = losses
        self.state = state
        self.model = model

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


@nnx.jit
def eval_batch(step_graph, step_state, inputs: Array | None, targets: Array | None, rngs: nnx.Rngs):
    transform, optimiser, loss = nnx.merge(step_graph, step_state)
    out = transform.create_out()
    out.inputs = inputs
    out.targets = targets
    transform.set_out(out)
    outputs = transform(inputs)
    transform.set_out(None)
    out.outputs = outputs
    accumulator = loss(transform, out, rngs)
    return accumulator, out


@nnx.jit
def train_batch(step_graph, step_state, inputs: Array | None, targets: Array | None, rngs: nnx.Rngs):

    def loss_fn(transform: Transform, loss: Loss, rngs: nnx.Rngs):
        out = transform.create_out()
        out.inputs = inputs
        out.targets = targets
        transform.set_out(out)
        outputs = transform(inputs)
        transform.set_out(None)
        out.outputs = outputs
        accumulator = loss(transform, out, rngs)
        return accumulator.loss(), (accumulator, out)

    transform, optimizer, loss = nnx.merge(step_graph, step_state)

    (loss_value, (accumulator, out)), grad = nnx.value_and_grad(loss_fn, has_aux=True)(transform, loss, rngs)

    optimizer.update(transform, grad)

    step_state = nnx.state((transform, optimizer, loss))
    return step_state, accumulator, out


def fit(
    transform: Transform,
    optimizer: nnx.Optimizer,
    stop_condition: StopCondition,
    loss: Loss,
    train: Loader[B],
    eval: Loader[B],
    batch_profiler: BatchProfiler = NoBatchProfiler(),
):
    step_graph, step_state = nnx.split((transform, optimizer, loss))

    epoch = 0
    stop = False
    rngs = nnx.Rngs(sample=20251115)

    while not stop:
        epoch += 1

        train_times = Accumulator({"epoch time", "batch compute time", "batch load time"})
        with train_times.timer("epoch time"):
            train_losses = loss.create_accumulator()
            transform.train()
            for batch in train.batches({"step": "train", "epoch": epoch}):
                with train_times.timer("batch compute time"):
                    with batch_profiler.profile("train") as profiling:
                        step_state, batch_losses, batch_out = train_batch(
                            step_graph, step_state, batch.inputs, batch.targets, rngs
                        )
                yield batch.clone(data={"out": batch_out, "model": transform})
                train_losses.add_accumulator(batch_losses)
                train_times.add_value(batch.load_time, "batch load time")

        yield (train_epoch := Epoch("train", epoch, train_times, train_losses, step_state[0], transform))

        eval_times = Accumulator({"epoch time", "batch compute time", "batch load time"})
        with eval_times.timer("epoch time"):
            eval_losses = loss.create_accumulator()
            for batch in eval.batches({"step": "eval", "epoch": epoch}):
                with eval_times.timer("batch compute time"):
                    with batch_profiler.profile("eval") as profiling:
                        batch_losses, batch_out = eval_batch(step_graph, step_state, batch.inputs, batch.targets, rngs)
                yield batch.clone(data={"out": batch_out, "model": transform})
                eval_losses.add_accumulator(batch_losses)
                eval_times.add_value(batch.load_time, "batch load time")
        yield (eval_epoch := Epoch("eval", epoch, eval_times, eval_losses, step_state[0], transform))

        stop = stop_condition(train_epoch, eval_epoch)

    nnx.update((transform, optimizer, loss), step_state)

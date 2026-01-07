from __future__ import annotations
from typing import TYPE_CHECKING, Iterable

import numpy as np

if TYPE_CHECKING:
    from timo.session import Session
    from typing import Iterable

from timo.fit import Epoch
from timo.batch import Batch
import tensorboardX
from flax import nnx


class Observer:

    def observe(self, training: Iterable[Batch | Epoch]) -> Iterable[Batch | Epoch]:
        for item in training:
            if isinstance(item, Epoch):
                self.add_step_epoch(item)
            elif isinstance(item, Batch):
                self.add_step_batch(item)
            yield item

    def add_step_epoch(self, epoch: Epoch):
        pass

    def add_step_batch(self, batch: Batch):
        pass


class PeriodicObserver(Observer):
    def __init__(self, observer: Observer, epoch_period: int | None, batch_period: int | None) -> None:
        super().__init__()
        self.observer = observer
        self.epoch_period = epoch_period
        self.batch_period = batch_period

        self.last_epoch: int | None = None
        self.last_batch: int | None = None

    def add_step_epoch(self, epoch: Epoch):
        super().add_step_epoch(epoch)
        if self.last_epoch is None:
            self.last_epoch = epoch.epoch
        if epoch.epoch - self.last_epoch >= (self.epoch_period or 0):
            self.last_epoch = epoch.epoch

        if self.last_epoch == epoch.epoch:
            self.observer.add_step_epoch(epoch)

    def add_step_batch(self, batch: Batch):
        super().add_step_batch(batch)
        batch_epoch = batch.epoch
        if batch_epoch != self.last_epoch:
            return
        if self.last_batch is None:
            self.last_batch = batch.index or 0
        if (batch.index or 0) - self.last_batch >= (self.batch_period or 0):
            self.last_batch = batch.index

        if self.last_batch == batch.index:
            self.observer.add_step_batch(batch)


class MultiObsever(Observer):
    def __init__(self, *observers: Observer) -> None:
        super().__init__()
        self.observers = observers

    def add_step_batch(self, batch: Batch):
        super().add_step_batch(batch)
        for observer in self.observers:
            observer.add_step_batch(batch)

    def add_step_epoch(self, epoch: Epoch):
        super().add_step_epoch(epoch)
        for observer in self.observers:
            observer.add_step_epoch(epoch)


class NoObserver(Observer):
    def observe(self, training: Iterable[Batch | Epoch]):
        return training


class TensorboardObserver(Observer):
    def __init__(self, session: Session, output: str) -> None:
        super().__init__()
        self.writer = tensorboardX.SummaryWriter(session.output_path("fitting", output), flush_secs=5)


class TensorboardLossObserver(TensorboardObserver):
    def __init__(self, session: Session):
        super().__init__(session, "losses")

    def add_step_epoch(self, epoch: Epoch):
        for loss, value in epoch.losses.means():
            self.writer.add_scalar(f"{loss}/{epoch.step}", value, epoch.epoch)
        self.writer.add_scalar(f"loss/{epoch.step}", epoch.losses.loss(), epoch.epoch)


class EpochPrintObserver(Observer):
    def add_step_epoch(self, epoch: Epoch):
        print(epoch)


class TensorboardParamObserver(TensorboardObserver):
    def __init__(self, session: Session) -> None:
        super().__init__(session, "params")

    def add_step_epoch(self, epoch: Epoch):
        if epoch.step != "eval":
            return
        transform = epoch.model
        for path, module in transform.iter_modules():
            state = nnx.state(module)
            for name, param in state.items():
                if not isinstance(param, nnx.Param):
                    continue
                self.writer.add_histogram(f"params/{path}/{name}", np.array(param), epoch.epoch)

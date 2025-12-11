from __future__ import annotations
import os
from typing import TYPE_CHECKING

import orbax.checkpoint


if TYPE_CHECKING:
    from timo.session import Session
    from typing import Iterable
    from timo.batch import Batch

from timo.fit import Epoch

import orbax


class Keeper:
    def __init__(self):
        self.best_train: Epoch | None = None
        self.last_train: Epoch | None = None
        self.best_eval: Epoch | None = None
        self.last_eval: Epoch | None = None

    def keep(self, training: Iterable[Batch | Epoch]):
        for epoch in training:
            if isinstance(epoch, Epoch):
                self.update(epoch)
            yield epoch

    def update(self, epoch: Epoch):
        if epoch.step == "train":
            if self.best_train is None or self.best_train.losses.mean() >= epoch.losses.mean():
                self.best_train = epoch
        elif epoch.step == "eval":
            if self.best_eval is None or self.best_eval.losses.mean() >= epoch.losses.mean():
                self.best_eval = epoch

    def save(self, session: Session):
        path = session.output_path("models")
        checkpointer = orbax.checkpoint.StandardCheckpointer()
        if self.best_train is not None:
            output_path = os.path.abspath(f"{path}/best_train/")
            checkpointer.save(output_path, self.best_train.state)
        if self.last_train is not None:
            output_path = os.path.abspath(f"{path}/last_train/")
            checkpointer.save(output_path, self.last_train.state)
        if self.best_eval is not None:
            output_path = os.path.abspath(f"{path}/best_eval/")
            checkpointer.save(output_path, self.best_eval.state)
        if self.last_eval is not None:
            output_path = os.path.abspath(f"{path}/last_eval/")
            checkpointer.save(output_path, self.last_eval.state)
        checkpointer.wait_until_finished()

    def load_best_eval(self, path: str):
        checkpointer = orbax.checkpoint.StandardCheckpointer()
        return checkpointer.restore(f"{path}/best_eval")

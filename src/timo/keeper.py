from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterable
    from timo.fit import Epoch


class Keeper:
    def __init__(self):
        self.best_train: Epoch | None = None
        self.best_eval: Epoch | None = None

    def keep(self, training: Iterable[Epoch]):
        for epoch in training:
            self.update(epoch)
            yield epoch

    def update(self, epoch: Epoch):
        if epoch.step == "train":
            if self.best_train is None or self.best_train.losses.mean() >= epoch.losses.mean():
                self.best_train = epoch
        elif epoch.step == "eval":
            if self.best_eval is None or self.best_eval.losses.mean() >= epoch.losses.mean():
                self.best_eval = epoch

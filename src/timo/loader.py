from __future__ import annotations
from copy import deepcopy
from tkinter import NO
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterable


from time import monotonic
from typing import Generic, TypeVar
from timo.batch import Batch

B = TypeVar("B", bound=Batch)


class Loader(Generic[B]):
    def batches(self, metadata: dict | None = None) -> Iterable[B]:
        for i in range(len(self)):
            yield self.get(i, metadata=metadata)

    def __len__(self):
        raise NotImplementedError()

    def get(self, index: int, metadata: dict | None = None) -> B:
        raise NotImplementedError()


class DataLoader(Loader[B]):
    def __init__(self, infos: list[B]):
        super().__init__()
        self.infos = infos

    def __len__(self):
        return len(self.infos)

    def get(self, index, metadata=None):
        clone: B = self.infos[index].clone(metadata=metadata)  # type: ignore
        return clone


class ShuffleLoader(Loader[B]):
    def __init__(self, loader: Loader[B]):
        super().__init__()
        self._loader = loader
        self._next = []

    def __len__(self):
        return len(self._loader)

    def get(self, index: int, metadata=None):
        if len(self._next) == 0:
            self._next = self._shuffle_indices()
        next = self._next.pop()
        return self._loader.get(next, metadata=metadata)

    def _shuffle_indices(self):
        import random

        indices = list(range(len(self._loader)))
        random.shuffle(indices)
        return indices


class BatchLoader(Loader[B]):
    def __init__(self, loader: Loader[B], size: int):
        super().__init__()
        self._loader = loader
        self._size = size

    def __len__(self):
        import math

        return int(math.ceil(len(self._loader) / self._size))

    def get(self, index: int, metadata=None):

        start = monotonic()
        items = []
        batch_type: type[B] | None = None
        for i in range(self._size):
            if i >= len(self._loader):
                break
            item = self._loader.get(index + i)
            if item is None:
                continue
            batch_type: type[B] | None = type(item)
            items.append(item)

        if batch_type is None:
            raise RuntimeError("Empty batch")

        if metadata is None:
            metadata = {}
        metadata["index"] = index
        metadata["load_time"] = monotonic() - start
        batch: B = batch_type.as_batch(items, **metadata)  # type: ignore
        return batch

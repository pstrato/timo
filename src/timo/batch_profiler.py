from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timo.session import Session

import contextlib
import jax


class BatchProfiler:
    @contextlib.contextmanager
    def profile(self, step: str):
        try:
            yield self.start_profiling(step)
        finally:
            self.stop_profiling()

    def start_profiling(self, step) -> bool:
        raise NotImplementedError()

    def stop_profiling(self):
        raise NotImplementedError()


class SkipBatchProfile(BatchProfiler):
    def __init__(self, profiler: BatchProfiler, skip_before: int | None = None, skip_after: int | None = None) -> None:
        super().__init__()
        self.profiler = profiler
        self.skip_before = skip_before
        self.skip_after = skip_after
        self.count = 0

    def start_profiling(self, step: str):
        self.count += 1
        if self.skip_before is not None and self.count < self.skip_before:
            return False
        if self.skip_after is not None and self.count > self.skip_after:
            return False
        return self.profiler.start_profiling(step)

    def stop_profiling(self):
        self.profiler.stop_profiling()


class StepBatchProfiler(BatchProfiler):
    def __init__(self, profiler: BatchProfiler, *steps: str) -> None:
        super().__init__()
        self.profiler = profiler
        self.steps = set(steps)

    def start_profiling(self, step):
        if step in self.steps:
            return self.profiler.start_profiling(step)
        return False

    def stop_profiling(self):
        self.profiler.stop_profiling()


class SessionBatchProfiler(BatchProfiler):
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
        self.profiling = False
        self.batch_count = 0

    def start_profiling(self, step):
        jax.profiler.start_trace(self.session.output_path("profile", f"batch-{self.batch_count}"))
        self.batch_count += 1
        self.profiling = True
        return True

    def stop_profiling(self):
        if self.profiling:
            jax.profiler.stop_trace()


class NoBatchProfiler(BatchProfiler):
    def start_profiling(self, step):
        return False

    def stop_profiling(self):
        pass

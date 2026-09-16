"""Run tasks in this process, or across a pool of workers.

Used for tests, for single-node runs and for anything small enough that a
scheduler is overhead. Failures are captured per task rather than raised
immediately, so one bad chunk does not abandon the rest of a level and the
caller sees the same failure report the cluster runner produces.
"""

from __future__ import annotations

import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Sequence

from .base import RunReport, TaskStatus

__all__ = ["LocalRunner"]


class LocalRunner:
    """Execute task ids locally.

    ``workers=1`` runs in-process, which keeps tracebacks intact and is what
    the tests use. Above that, tasks are dispatched to a pool; threads suit the
    IO-bound blend and update passes, processes suit the compute-bound register
    pass where one worker holds one device.
    """

    name = "local"

    def __init__(self, workers: int = 1, use_processes: bool = False) -> None:
        if workers < 1:
            raise ValueError(f"workers must be at least 1, got {workers}")
        self.workers = int(workers)
        self.use_processes = bool(use_processes)

    def run(
        self,
        pass_name: str,
        fn: Callable[..., Any],
        ids: Sequence[int],
        **kwargs: Any,
    ) -> RunReport:
        report = RunReport(pass_name=pass_name)
        ids = list(ids)
        if not ids:
            return report

        if self.workers == 1:
            for i in ids:
                report.statuses.append(self._one(fn, i, kwargs))
            return report

        pool = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        with pool(max_workers=self.workers) as ex:
            futures = {ex.submit(self._one, fn, i, kwargs): i for i in ids}
            for fut in futures:
                report.statuses.append(fut.result())
        report.statuses.sort(key=lambda s: s.task_id)
        return report

    @staticmethod
    def _one(fn: Callable[..., Any], task_id: int, kwargs: dict) -> TaskStatus:
        try:
            return TaskStatus(task_id=task_id, ok=True, result=fn(task_id, **kwargs))
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            return TaskStatus(
                task_id=task_id,
                ok=False,
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )

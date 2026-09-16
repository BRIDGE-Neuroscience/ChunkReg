"""The runner contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

__all__ = ["TaskStatus", "Runner"]


@dataclass
class TaskStatus:
    task_id: int
    ok: bool
    result: Any = None
    error: str | None = None


@dataclass
class RunReport:
    pass_name: str
    statuses: list[TaskStatus] = field(default_factory=list)

    @property
    def failed(self) -> list[TaskStatus]:
        return [s for s in self.statuses if not s.ok]

    @property
    def ok(self) -> bool:
        return not self.failed

    def results(self) -> list[Any]:
        return [s.result for s in self.statuses if s.ok]

    def raise_for_failures(self) -> None:
        bad = self.failed
        if not bad:
            return
        head = bad[0]
        raise RuntimeError(
            f"{len(bad)} of {len(self.statuses)} {self.pass_name} tasks failed; "
            f"first failure was task {head.task_id}:\n{head.error}"
        )


@runtime_checkable
class Runner(Protocol):
    name: str

    def run(
        self, pass_name: str, fn, ids: Sequence[int], **kwargs
    ) -> RunReport: ...

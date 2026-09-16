"""Runners: where task ids execute.

A runner is handed a pass name and a list of task ids and is responsible only
for getting each id run somewhere. It never decides what a task does. That
separation is why the same pass functions serve a laptop with one GPU and a
cluster with a thousand.
"""

from __future__ import annotations

from .base import Runner, TaskStatus
from .local import LocalRunner

__all__ = ["Runner", "TaskStatus", "LocalRunner", "get_runner"]


def get_runner(name: str, **kwargs) -> Runner:
    if name == "local":
        return LocalRunner(**kwargs)
    if name == "slurm":
        from .slurm import SlurmRunner

        return SlurmRunner(**kwargs)
    raise ValueError(f"unknown runner {name!r}; use 'local' or 'slurm'")

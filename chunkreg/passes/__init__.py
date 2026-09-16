"""Passes: pure, idempotent units of work over a level.

Every pass is a function of ``(config, manifest, task_id)`` that reads stores,
writes only the objects it owns, and returns a small status dictionary. None of
them knows whether it was called by a local pool or a scheduler array element,
which is what makes the local and cluster paths the same code.

Ownership is the invariant that removes locking: ``register`` writes one task
array, and ``blend``, ``update`` and ``promote`` are parallel over output
shards where a shard is a chunk core. No two tasks ever write the same object.
"""

from __future__ import annotations

from .manifest import Manifest, TaskPlan, build_manifest
from .register import run_register_task
from .blend import run_blend_task
from .update import run_update_task
from .promote import promote_level

__all__ = [
    "Manifest",
    "TaskPlan",
    "build_manifest",
    "run_register_task",
    "run_blend_task",
    "run_update_task",
    "promote_level",
]

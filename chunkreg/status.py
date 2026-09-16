"""Progress read from the stores, not from the scheduler.

Task completion is recorded in the task array's own metadata, so progress is
visible whether the run is on a local pool, on a cluster, or stopped. That also
means a resumed run can tell exactly which task ids still have work, which is
what makes resubmitting a failed scheduler array element safe and cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import RunConfig
from .passes.manifest import Manifest
from .store import Field, TaskArray, Volume

__all__ = ["LevelStatus", "status", "pending_tasks"]


@dataclass
class LevelStatus:
    level: int
    iteration: int
    tasks_total: int
    tasks_done: int
    has_template: bool
    fields_written: int
    recorded: bool = False
    """True when the pass left a completion record, which outlives the task
    arrays that retention deletes."""

    @property
    def complete(self) -> bool:
        if self.recorded:
            return True
        return self.tasks_total > 0 and self.tasks_done == self.tasks_total


def _iteration_of(path: Path) -> int:
    digits = "".join(c for c in path.stem if c.isdigit())
    return int(digits) if digits else 0


def _manifests(cfg: RunConfig):
    """Every manifest on disk, ordered by level then pass.

    Sorted numerically rather than lexicographically: with ten or more levels
    or passes, ``L10`` sorts before ``L2`` as text, which put the wrong pass
    last and so named the wrong outstanding task ids.
    """
    root = cfg.root_path / "levels"
    if not root.exists():
        return
    found = []
    for level_dir in root.glob("L*"):
        try:
            level = int(level_dir.name[1:])
        except ValueError:
            continue
        for path in level_dir.glob("manifest_it*.json"):
            found.append((level, _iteration_of(path), path))
    for level, _, path in sorted(found):
        yield level, path


def pending_tasks(cfg: RunConfig, level: int, iteration: int) -> list[int]:
    """Task ids at this level and pass whose output is not marked complete."""
    path = cfg.level_dir(level) / f"manifest_it{iteration}.json"
    if not Path(path).exists():
        return []
    if cfg.pass_record_path(level, iteration).exists():
        return []  # the pass finished; its task arrays were cleaned up
    manifest = Manifest.load(path)
    return [
        t.task_id
        for t in manifest.tasks
        if not TaskArray.is_complete(
            cfg.task_path(level, iteration, t.task_id), cfg.backend
        )
    ]


def collect(cfg: RunConfig) -> list[LevelStatus]:
    out: list[LevelStatus] = []
    for level, path in _manifests(cfg):
        manifest = Manifest.load(path)
        recorded = cfg.pass_record_path(level, manifest.iteration).exists()
        # A recorded pass has already had its task arrays cleaned up, so there
        # is nothing left to count and no reason to go looking.
        done = (
            manifest.n_tasks
            if recorded
            else sum(
                1
                for t in manifest.tasks
                if TaskArray.is_complete(
                    cfg.task_path(level, manifest.iteration, t.task_id), cfg.backend
                )
            )
        )
        out.append(
            LevelStatus(
                level=level,
                iteration=manifest.iteration,
                tasks_total=manifest.n_tasks,
                tasks_done=done,
                recorded=recorded,
                has_template=Volume.exists(cfg.template_path(level), cfg.backend),
                fields_written=sum(
                    1
                    for s in cfg.registered_ids
                    if Field.exists(cfg.field_path(level, s), cfg.backend)
                ),
            )
        )
    return out


def status(cfg: RunConfig) -> str:
    rows = collect(cfg)
    if not rows:
        return (
            f"no levels started under {cfg.root_path}. Run 'chunkreg setup' to "
            f"check the configuration, then 'chunkreg run' to begin."
        )
    head = f"{'lvl':>3} {'pass':>5} {'tasks':>13} {'template':>9} {'fields':>7}"
    out = [head, "-" * len(head)]
    for r in rows:
        bar = f"{r.tasks_done}/{r.tasks_total}"
        out.append(
            f"{r.level:>3} {r.iteration:>5} {bar:>13} "
            f"{'yes' if r.has_template else 'no':>9} "
            f"{r.fields_written}/{len(cfg.registered_ids)}".rjust(7)
        )
    incomplete = [r for r in rows if not r.complete]
    if incomplete:
        r = incomplete[-1]
        ids = pending_tasks(cfg, r.level, r.iteration)
        out.append("")
        out.append(
            f"level {r.level} pass {r.iteration} has {len(ids)} task(s) "
            f"outstanding: {ids[:10]}{' ...' if len(ids) > 10 else ''}"
        )
    return "\n".join(out)

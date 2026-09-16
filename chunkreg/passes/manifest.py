"""The work list for one level and one template pass.

A manifest fixes, before any task runs, exactly which (subject, chunk) pairs
exist, how they are grouped into tasks, and which task array holds each result.
Tasks then need no coordination: a register task reads its own slice of the
list, and a blend task looks up the results that touch its shard by index.

Grouping is chunk-major: a task holds every subject of one chunk before moving
to the next chunk, in the chunk lattice's lexicographic order. Every subject in
a chunk is registered against the same template chunk, so a task extracts that
chunk's template features once and reuses them for all subjects instead of
once per subject. With anatomix features that is most of a chunk's feature
cost at four subjects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from ..config import RunConfig
from ..grid import Chunk, GridSpec, Profile, tile
from ..store import TaskEntry

__all__ = ["TaskPlan", "Manifest", "build_manifest"]


@dataclass(frozen=True)
class TaskPlan:
    """One register task: a contiguous run of (subject, chunk) pairs."""

    task_id: int
    entries: tuple[TaskEntry, ...]

    def __len__(self) -> int:
        return len(self.entries)


@dataclass
class Manifest:
    level: int
    iteration: int
    grid: GridSpec
    subjects: tuple[str, ...]
    chunks: tuple[Chunk, ...]
    tasks: tuple[TaskPlan, ...]
    d_max_mm: float | None
    seeded: bool
    _index: dict[tuple[str, int], tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self._index:
            self._index = {
                (e.subject, e.chunk_id): (t.task_id, i)
                for t in self.tasks
                for i, e in enumerate(t.entries)
            }

    # -- lookups ------------------------------------------------------------ #
    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    @property
    def n_entries(self) -> int:
        return sum(len(t) for t in self.tasks)

    def chunk(self, chunk_id: int) -> Chunk:
        return self.chunks[chunk_id]

    def locate(self, subject: str, chunk_id: int) -> tuple[int, int]:
        """Which task array holds a result, and at which entry index."""
        try:
            return self._index[(subject, chunk_id)]
        except KeyError as exc:
            raise KeyError(
                f"no entry for subject {subject!r} chunk {chunk_id} in the "
                f"level {self.level} manifest"
            ) from exc

    def chunks_touching_core(self, chunk_id: int) -> list[Chunk]:
        """Chunks whose padded box reaches into this chunk's core.

        This is the read set of the blend task that owns the core: the chunk
        itself plus, at a halo smaller than the core, at most its 26 neighbours.
        """
        own = self.chunks[chunk_id]
        return [c for c in self.chunks if c.intersects(own.core_origin, own.core_shape)]

    # -- serialisation ------------------------------------------------------ #
    def to_json(self) -> dict:
        return {
            "level": self.level,
            "iteration": self.iteration,
            "grid": {
                "shape": list(self.grid.shape),
                "spacing_mm": self.grid.spacing_mm,
                "origin_mm": list(self.grid.origin_mm),
            },
            "subjects": list(self.subjects),
            "d_max_mm": self.d_max_mm,
            "seeded": self.seeded,
            "chunks": [
                {
                    "id": c.id,
                    "index": list(c.index),
                    "core_origin": list(c.core_origin),
                    "core_shape": list(c.core_shape),
                    "pad_origin": list(c.pad_origin),
                    "pad_shape": list(c.pad_shape),
                }
                for c in self.chunks
            ],
            "tasks": [
                {"task_id": t.task_id, "entries": [e.to_json() for e in t.entries]}
                for t in self.tasks
            ],
        }

    @classmethod
    def from_json(cls, d: dict) -> "Manifest":
        g = d["grid"]
        return cls(
            level=int(d["level"]),
            iteration=int(d["iteration"]),
            grid=GridSpec(tuple(g["shape"]), g["spacing_mm"], tuple(g["origin_mm"])),
            subjects=tuple(d["subjects"]),
            chunks=tuple(
                Chunk(
                    id=int(c["id"]),
                    index=tuple(c["index"]),
                    core_origin=tuple(c["core_origin"]),
                    core_shape=tuple(c["core_shape"]),
                    pad_origin=tuple(c["pad_origin"]),
                    pad_shape=tuple(c["pad_shape"]),
                )
                for c in d["chunks"]
            ),
            tasks=tuple(
                TaskPlan(
                    task_id=int(t["task_id"]),
                    entries=tuple(TaskEntry.from_json(e) for e in t["entries"]),
                )
                for t in d["tasks"]
            ),
            d_max_mm=d.get("d_max_mm"),
            seeded=bool(d.get("seeded", True)),
        )

    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_json(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Manifest":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def build_manifest(
    cfg: RunConfig,
    level: int,
    iteration: int,
    grid: GridSpec,
    subjects: Sequence[str] | None = None,
) -> Manifest:
    """Enumerate the work for one template pass at one level."""
    p = cfg.profile
    chunks = tile(grid, p)
    subs = tuple(subjects) if subjects is not None else cfg.registered_ids
    per_task = max(1, cfg.slurm.register.chunks_per_task)

    entries = [
        TaskEntry(
            subject=s,
            chunk_id=c.id,
            chunk_index=c.index,
            pad_origin=c.pad_origin,
            pad_shape=c.pad_shape,
        )
        for c in chunks
        for s in subs
    ]
    tasks = tuple(
        TaskPlan(task_id=i, entries=tuple(entries[o : o + per_task]))
        for i, o in enumerate(range(0, len(entries), per_task))
    )

    # A single-chunk level has no halo to bound it, so it is bounded by the
    # aperture: a fraction of the volume's own extent.
    if len(chunks) == 1:
        d_max = cfg.levels.level0_max_disp_frac * min(grid.extent_mm)
    else:
        d_max = p.d_max_mm(grid.spacing_mm)
    return Manifest(
        level=level,
        iteration=iteration,
        grid=grid,
        subjects=subs,
        chunks=tuple(chunks),
        tasks=tasks,
        d_max_mm=d_max,
        seeded=level > 0 or iteration > 0,
    )

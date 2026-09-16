"""The register pass: the only GPU stage, and the only place features exist.

One task handles a contiguous run of (subject, chunk) pairs and writes one task
array. For each pair it reads the template and subject chunks, pre-warps the
subject by its seed, extracts features from both on the template grid, solves a
residual, clamps it, composes it back onto the seed, and stores the total on the
displacement lattice.

Three behaviours here are robustness controls rather than algorithm:

* a chunk with almost no foreground emits its seed rather than registering air;
* the residual is clamped to what the halo can justify, which is what turns the
  halo bound into a guarantee instead of an assumption;
* a chunk whose result folds is retried up a ladder and, failing that, falls
  back to its seed and is flagged, so one bad chunk cannot poison a level.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .. import fields as _fields
from .. import stats as _stats
from ..config import RunConfig
from ..engines import get_engine
from ..features import get_extractor
from ..store import Field, TaskArray, TaskEntry, Volume
from ._common import tissue_fraction, warped_subject_block
from .manifest import Manifest

__all__ = ["run_register_task", "register_chunk"]


def _seed_block(
    cfg: RunConfig, manifest: Manifest, entry: TaskEntry, shape
) -> np.ndarray:
    """The seed for a chunk: the current field, recentred onto the template.

    The previous pass moved the template by ``-eps * u_bar`` to remove its
    shape bias. The stored field still points at where the template used to be,
    so the seed is that field composed with the same motion. Skipping this is
    what makes the mean displacement grow instead of decaying by ``(1 - eps)``
    per pass: the fields end up following the template rather than the template
    converging on the fields.
    """
    level = manifest.level
    path = cfg.field_path(level, entry.subject)
    if not Field.exists(path, cfg.backend):
        return np.zeros((3,) + tuple(shape), dtype=np.float32)
    seed = Field.open(path, cfg.backend).read_dense(entry.pad_origin, shape)

    recentre = cfg.recentre_path(level)
    if manifest.iteration > 0 and Field.exists(recentre, cfg.backend):
        motion = Field.open(recentre, cfg.backend).read_dense(entry.pad_origin, shape)
        seed = _fields.compose(motion, seed, manifest.grid.spacing_mm)
    return seed


def register_chunk(
    cfg: RunConfig,
    manifest: Manifest,
    entry: TaskEntry,
    template: Volume,
    subject: Volume,
    engine,
    extractor,
    seed: np.ndarray | None = None,
) -> dict[str, Any]:
    """Solve one chunk. Returns the total field and a QC record."""
    level = manifest.level
    spacing = manifest.grid.spacing_mm
    shape = tuple(entry.pad_shape)
    stages = list(cfg.levels.stages(level))

    if seed is None:
        seed = _seed_block(cfg, manifest, entry, shape)

    fixed_img = template.read_padded(0, entry.pad_origin, shape, normalise=True)

    frac = tissue_fraction(fixed_img)
    if frac < cfg.levels.min_tissue_fraction:
        return {
            "field": seed.astype(np.float32, copy=False),
            "skipped": True,
            "reason": "tissue",
            "tissue_fraction": frac,
            "folds": 0.0,
            "iters": [],
            "residual_hist": None,
            "retries": 0,
        }

    moving_img = warped_subject_block(
        subject, level, entry.pad_origin, shape, seed, spacing, normalise=True
    )

    fixed = extractor(fixed_img, spacing)
    moving = extractor(moving_img, spacing)

    d_max = manifest.d_max_mm
    retries = 0
    result = None
    total = seed
    folds = 0.0

    ladder = [None, *cfg.retry.ladder]
    for step in ladder:
        run_stages = stages
        limit = d_max
        if step == "sigma_w_x2":
            run_stages = [
                replace(s, smooth_warp_sigma=s.smooth_warp_sigma * 2)
                if s.is_deformable
                else s
                for s in stages
            ]
        elif step == "clamp_x0.5":
            limit = None if d_max is None else d_max * 0.5
        elif step == "emit_seed":
            return {
                "field": seed.astype(np.float32, copy=False),
                "skipped": True,
                "reason": "folds",
                "tissue_fraction": frac,
                "folds": folds,
                "iters": [] if result is None else result.iters_per_scale,
                "residual_hist": None,
                "retries": retries,
            }

        result = engine.register(fixed, moving, spacing, run_stages)
        residual = _fields.clamp(result.disp_mm, limit)
        total = _fields.compose(residual, seed, spacing)
        folds = _fields.fold_fraction(total, spacing)
        if folds <= cfg.retry.fold_frac:
            break
        retries += 1

    # The *residual* is the convergence signal, not the total. The total field
    # is the full correspondence and never shrinks; what has to fall below the
    # next level's clamp is the part this pass could not already explain.
    # Reported as a histogram so the pipeline can take an exact percentile over
    # the whole level rather than an outlier-driven maximum over tasks.
    residual_hist = _stats.histogram(_fields.magnitude(residual))

    return {
        "field": total,
        "skipped": False,
        "reason": None,
        "tissue_fraction": frac,
        "folds": folds,
        "residual_hist": residual_hist.tolist(),
        "iters": [] if result is None else list(result.iters_per_scale),
        "converged": True if result is None else bool(result.converged),
        "loss_start": None if not result or not result.loss_curve else result.loss_curve[0],
        "loss_end": None if not result or not result.loss_curve else result.loss_curve[-1],
        "retries": retries,
    }


def run_register_task(
    cfg: RunConfig,
    manifest: Manifest,
    task_id: int,
    engine=None,
    extractor=None,
) -> dict[str, Any]:
    """Run one register task. Idempotent: a completed task array is skipped."""
    plan = manifest.tasks[task_id]
    path = cfg.task_path(manifest.level, manifest.iteration, task_id)

    if TaskArray.is_complete(path, cfg.backend):
        return {"task_id": task_id, "skipped": True, "entries": len(plan)}

    t0 = time.perf_counter()
    engine = engine or get_engine("demons")
    extractor = extractor or get_extractor(cfg.profile.features)
    extractor.setup()

    template = Volume.open(cfg.template_path(manifest.level), cfg.backend)
    subjects: dict[str, Volume] = {}

    array = TaskArray.create(
        path,
        plan.entries,
        cfg.profile,
        backend=cfg.backend,
        level=manifest.level,
        iteration=manifest.iteration,
    )

    records = []
    for i, entry in enumerate(plan.entries):
        if entry.subject not in subjects:
            subjects[entry.subject] = Volume.open(
                cfg.subject_path(entry.subject), cfg.backend
            )
        rec = register_chunk(
            cfg,
            manifest,
            entry,
            template,
            subjects[entry.subject],
            engine,
            extractor,
        )
        lat_origin, lat_shape = entry.lattice(cfg.profile.lattice_factor)
        array.write(i, _to_lattice(rec.pop("field"), lat_shape, cfg.profile.lattice_factor))
        records.append({"subject": entry.subject, "chunk_id": entry.chunk_id, **rec})

    array.meta["records"] = records
    array.mark_complete()
    return {
        "task_id": task_id,
        "skipped": False,
        "entries": len(plan),
        "seconds": time.perf_counter() - t0,
        "records": records,
    }


def _to_lattice(u: np.ndarray, lat_shape, factor: int) -> np.ndarray:
    """Mean-pool a level-resolution field onto its storage lattice."""
    a = np.asarray(u, dtype=np.float32)
    target = tuple(int(n) for n in lat_shape)
    need = tuple(n * factor for n in target)
    pad = [(0, 0)] + [(0, max(0, n - s)) for n, s in zip(need, a.shape[1:])]
    if any(p[1] for p in pad):
        a = np.pad(a, pad, mode="edge")
    a = a[:, : need[0], : need[1], : need[2]]
    return a.reshape(
        3, target[0], factor, target[1], factor, target[2], factor
    ).mean(axis=(2, 4, 6))

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
from .. import xp as _xp
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
        return _xp.zeros((3,) + tuple(shape))
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
    fixed_cache: dict | None = None,
) -> dict[str, Any]:
    """Solve one chunk. Returns the total field and a QC record.

    ``fixed_cache`` carries the template chunk's tissue fraction and features
    between calls. Every subject of a chunk registers against the same
    template chunk, so a task that holds several subjects of one chunk
    extracts those features once. Only the most recent chunk is kept.
    """
    level = manifest.level
    spacing = manifest.grid.spacing_mm
    shape = tuple(entry.pad_shape)
    stages = list(cfg.levels.stages(level))

    if seed is None:
        seed = _seed_block(cfg, manifest, entry, shape)

    key = (entry.chunk_id, tuple(entry.pad_origin), shape)
    cached = None if fixed_cache is None else fixed_cache.get(key)
    if cached is None:
        fixed_img = template.read_padded(0, entry.pad_origin, shape, normalise=True)
        frac = tissue_fraction(fixed_img)
        fixed = (
            extractor(fixed_img, spacing)
            if frac >= cfg.levels.min_tissue_fraction
            else None
        )
        del fixed_img
        cached = (frac, fixed)
        if fixed_cache is not None:
            fixed_cache.clear()  # one chunk's features at a time
            fixed_cache[key] = cached
    frac, fixed = cached

    if frac < cfg.levels.min_tissue_fraction:
        return {
            "field": _xp.to_float32(seed),
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

    moving = extractor(moving_img, spacing)
    del moving_img

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
                "field": _xp.to_float32(seed),
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
        # The records are what the level's stopping rule reads, so a task
        # finished before a restart still has to report them.
        done = TaskArray.open(path, cfg.backend)
        # Only if it holds this task's entries: a task array written under a
        # manifest that grouped the work differently has the same path but
        # different contents, and taking it as done would lose that work.
        if [e.to_json() for e in done.entries] == [e.to_json() for e in plan.entries]:
            return {
                "task_id": task_id,
                "skipped": True,
                "entries": len(plan),
                "records": done.meta.get("records") or [],
            }

    t0 = time.perf_counter()
    engine = engine or get_engine(cfg.engine_for(_xp.device_name()))
    extractor = extractor or get_extractor(cfg.profile.features, seed=cfg.seed)
    extractor.setup()
    # A sampling extractor registers a different channel subset each pass. The
    # subset is a function of the pass alone, so every chunk of it, in every
    # task and on every worker, registers the same channels.
    if hasattr(extractor, "select"):
        extractor.select(manifest.level, manifest.iteration, cfg.seed)

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
    fixed_cache: dict = {}
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
            fixed_cache=fixed_cache,
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


def _to_lattice(u, lat_shape, factor: int):
    """Mean-pool a level-resolution field onto its storage lattice."""
    return _fields.pool_field(u, factor, tuple(int(n) for n in lat_shape))

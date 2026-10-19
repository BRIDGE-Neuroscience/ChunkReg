"""The blend pass: partition-of-unity merge, plus the template accumulators.

Parallel over output shards, where a shard is a chunk core. For its own core a
task gathers the residuals of every chunk whose padded box reaches in, weights
them by their Hann windows, divides by the analytic window sum and writes the
subject's field shard. In the same read it warps the subject through that field
into the intensity accumulator and adds the field to the displacement
accumulator, so the template average costs no extra pass over the data.

The denominator is never stored. Cores tile the level exactly, so every voxel
lies in exactly one core whose own window is 1 there, and the sum is at least 1
everywhere by construction.
"""

from __future__ import annotations

import math
import time
from itertools import product
from typing import Any

import numpy as np
from scipy import ndimage

from .. import backend as _backend
from .. import fields as _fields
from .. import xp as _xp
from ..config import RunConfig
from ..grid import chunks_touching, window_on
from ..store import Field, TaskArray, Volume, read_padded, write_block
from ._common import warped_subject_block
from .manifest import Manifest

__all__ = ["run_blend_task", "ensure_accumulators", "blend_core", "blend_depends", "blend_margin", "blend_box"]


def blend_depends(cfg: RunConfig, manifest: Manifest) -> dict[int, set[int]]:
    """For each core, the register tasks whose results its blend reads.

    A core's blend reads the residuals of every chunk whose padded box reaches
    into it -- itself plus at most its 26 neighbours -- for every subject, and
    those sit in whichever task arrays the manifest grouped them into. Stating
    that before the pass runs is what lets a runner start one core's blend
    while the rest of the register pass is still going, instead of holding
    every worker at a barrier that only the last few register tasks are still
    keeping shut.

    Candidates come from the chunk lattice rather than from testing every
    chunk against every other, which is quadratic and at production chunk
    counts costs more than the answer is worth. A padded box overruns its core
    by at most the halo, so only chunks within ``ceil(halo / core)`` index
    steps can reach in; each candidate is then checked against the same
    overlap predicate the blend itself uses, so the result is exactly
    :meth:`Manifest.chunks_touching_core`.
    """
    margin = blend_margin(cfg, manifest.level)
    radius = -(-(int(cfg.profile.halo) + margin) // int(cfg.profile.core))
    offsets = list(product(range(-radius, radius + 1), repeat=3))
    by_index = {tuple(c.index): c for c in manifest.chunks}
    deps: dict[int, set[int]] = {}
    for own in manifest.chunks:
        box_lo, box_shape = blend_box(manifest.grid, own, margin)
        tasks: set[int] = set()
        for dz, dy, dx in offsets:
            other = by_index.get(
                (own.index[0] + dz, own.index[1] + dy, own.index[2] + dx)
            )
            if other is None or not other.intersects(box_lo, box_shape):
                continue
            for subject in manifest.subjects:
                tasks.add(manifest.locate(subject, other.id)[0])
        deps[own.id] = tasks
    return deps


def ensure_accumulators(cfg: RunConfig, manifest: Manifest) -> tuple[Any, Any]:
    """Create or open the intensity and displacement accumulators for a pass.

    Writing an accumulator is owner-computes, but *creating* one is shared
    state: every blend task of a pass needs the same two objects. The pipeline
    calls this once before dispatching, and each task calls it again and
    tolerates having lost the race, so the pass is correct whether tasks start
    together on a scheduler or one at a time in a pool.
    """
    b = _backend.get_backend(cfg.backend)
    grid = manifest.grid
    p = cfg.profile_at(manifest.level)
    lat_shape = p.lattice_shape(grid.shape)
    isum_path = cfg.accumulator_path(manifest.level, manifest.iteration, "isum")
    wsum_path = cfg.accumulator_path(manifest.level, manifest.iteration, "wsum")

    inner = max(1, p.inner_chunk // p.lattice_factor)
    core = max(inner, p.core // p.lattice_factor)
    i_chunks, i_shards = _backend.shard_layout(grid.shape, p.inner_chunk, p.core)
    w_chunks, w_shards = _backend.shard_layout(lat_shape, inner, core)
    _create_if_absent(b, isum_path, grid.shape, chunks=i_chunks, shards=i_shards)
    _create_if_absent(
        b,
        wsum_path,
        (3,) + tuple(lat_shape),
        chunks=(3,) + w_chunks,
        shards=(3,) + w_shards,
    )
    return b.open(isum_path, "r+"), b.open(wsum_path, "r+")


def _create_if_absent(b, path, shape, chunks, shards) -> None:
    if b.exists(path):
        return
    try:
        b.create(path, shape, np.float32, chunks=chunks, shards=shards)
    except FileExistsError:
        # Another task of the same pass created it between the check and the
        # call. Harmless: the geometry is a function of the manifest, so both
        # tasks would have created the same object.
        pass


def blend_margin(cfg: RunConfig, level: int) -> int:
    """Voxels read past a core so its smoothing sees the neighbours' field.

    The blended field is smoothed by half a lattice voxel. Done within the
    core alone, with the core's own edge replicated, every core ended with a
    kink in the field's derivative at its boundary; with large displacements
    those kinks were folds along every chunk seam. Reading the support of the
    kernel past the core (the neighbours' residuals cover it, since the halo
    is wider) and cropping after smoothing removes them.
    """
    sigma = 0.5 * cfg.profile_at(level).lattice_factor
    return int(math.ceil(4.0 * sigma)) if sigma > 0 else 0


def blend_box(grid, chunk, margin: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """A chunk's core widened by ``margin``, clipped to the level."""
    lo = np.maximum(np.asarray(chunk.core_origin, dtype=np.int64) - margin, 0)
    hi = np.minimum(
        np.asarray(chunk.core_origin, dtype=np.int64) + np.asarray(chunk.core_shape, dtype=np.int64) + margin,
        np.asarray(grid.shape, dtype=np.int64),
    )
    return tuple(int(v) for v in lo), tuple(int(v) for v in hi - lo)


def blend_core(
    cfg: RunConfig, manifest: Manifest, subject: str, chunk_id: int
) -> np.ndarray:
    """Partition-of-unity merge over one core, at level resolution.

    Accumulated over the core plus :func:`blend_margin`, smoothed there, and
    cropped to the core, so the smoothing never replicates a core's own edge.
    """
    own = manifest.chunk(chunk_id)
    grid = manifest.grid
    core_shape = tuple(own.core_shape)
    margin = blend_margin(cfg, manifest.level)
    box_lo, box_shape = blend_box(grid, own, margin)
    num = _xp.zeros((3,) + tuple(box_shape), dtype=np.float64)
    den = _xp.zeros(tuple(box_shape), dtype=np.float64)

    lo = np.asarray(box_lo, dtype=np.int64)
    hi = lo + np.asarray(box_shape, dtype=np.int64)

    cache: dict[int, TaskArray] = {}
    for other in chunks_touching(manifest.chunks, box_lo, box_shape):
        task_id, index = manifest.locate(subject, other.id)
        if task_id not in cache:
            cache[task_id] = TaskArray.open(
                cfg.task_path(manifest.level, manifest.iteration, task_id), cfg.backend
            )
        plo = np.asarray(other.pad_origin, dtype=np.int64)
        phi = plo + np.asarray(other.pad_shape, dtype=np.int64)
        a = np.maximum(lo, plo)
        bnd = np.minimum(hi, phi)
        if np.any(bnd <= a):
            continue
        src = tuple(slice(int(s), int(e)) for s, e in zip(a - plo, bnd - plo))
        dst = tuple(slice(int(s), int(e)) for s, e in zip(a - lo, bnd - lo))

        # Only the overlap with this core is ever read, decompressed, upsampled
        # or weighted. The lattice read covers just that overlap, and the
        # upsample takes the box relative to what was read.
        f = cache[task_id].factor  # the lattice the residual was stored on
        llo = [b.start // f for b in src]
        lhi = [-(-b.stop // f) for b in src]
        lat = cache[task_id].read_box(index, llo, lhi)
        local = tuple(slice(b.start - l * f, b.stop - l * f) for b, l in zip(src, llo))
        if _xp.uses_torch():
            from .. import gpu_ops
            from ..grid import axis_windows

            dense = gpu_ops.upsample_box(_xp.put(lat), f, local)
            wz, wy, wx = (
                _xp.put(w[box]) for w, box in zip(axis_windows(other, grid, cfg.profile), src)
            )
            wq = gpu_ops.outer3(wz, wy, wx)
        else:
            dense = _upsample_box(lat, f, local)
            wq = window_on(other, grid, cfg.profile, src)
        num[(slice(None),) + dst] += dense * wq
        den[dst] += wq

    if float(den.min()) <= 0:
        raise RuntimeError(
            f"blend denominator reached zero on chunk {chunk_id}; cores should "
            f"tile the level exactly, so this means the manifest is inconsistent"
        )
    out = _xp.to_float32(num / den)

    sigma = 0.5 * cfg.profile_at(manifest.level).lattice_factor  # half a lattice voxel
    if sigma > 0:
        out = _fields.smooth(out, sigma, mode="nearest")
    off = np.asarray(own.core_origin, dtype=np.int64) - lo
    crop = (slice(None),) + tuple(
        slice(int(o), int(o + n)) for o, n in zip(off, core_shape)
    )
    out = out[crop]
    return out.contiguous() if _xp.is_tensor(out) else np.ascontiguousarray(out)


def _upsample_box(
    lat: np.ndarray, factor: int, box: tuple[slice, slice, slice]
) -> np.ndarray:
    """Upsample a chunk's stored residual over one sub-box of its padded box.

    Level voxel ``i`` reads lattice voxel ``i // factor``, so only the lattice
    range covering the box has to be expanded. Expanding the whole padded box
    to pick one overlap out of it cost 522 MB per neighbour at the production
    profile.
    """
    a = np.asarray(lat, dtype=np.float32)
    if factor == 1:
        return np.ascontiguousarray(a[(slice(None),) + box])
    lo = [b.start // factor for b in box]
    hi = [-(-b.stop // factor) for b in box]
    sub = a[:, lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    up = np.repeat(
        np.repeat(np.repeat(sub, factor, axis=1), factor, axis=2), factor, axis=3
    )
    inner = (slice(None),) + tuple(
        slice(b.start - lo[d] * factor, b.start - lo[d] * factor + (b.stop - b.start))
        for d, b in enumerate(box)
    )
    return np.ascontiguousarray(up[inner])


def run_blend_task(cfg: RunConfig, manifest: Manifest, chunk_id: int) -> dict[str, Any]:
    """Blend one shard for every subject and fold it into the accumulators."""
    t0 = time.perf_counter()
    own = manifest.chunk(chunk_id)
    grid = manifest.grid
    spacing = grid.spacing_mm
    # The accumulators feed the update pass, which estimates the template. A
    # template held fixed is never estimated, so for it the warped subject
    # would be read and a float32 copy of it written every pass for nothing:
    # at 20 um that was a second read of the moving volume and 1.6 TB of
    # writes per pass.
    accumulate = cfg.template_subject is None
    isum, wsum = ensure_accumulators(cfg, manifest) if accumulate else (None, None)

    core_origin = tuple(own.core_origin)
    core_shape = tuple(own.core_shape)
    acc_i = _xp.zeros(core_shape) if accumulate else None
    lat_origin, lat_shape = None, None
    acc_w = None
    stats = []

    for subject in manifest.subjects:
        u = blend_core(cfg, manifest, subject, chunk_id)

        field = Field.open(cfg.field_path(manifest.level, subject), cfg.backend)
        field.write_dense(core_origin, u)

        if accumulate:
            vol = Volume.open(cfg.subject_path(subject), cfg.backend)
            warped = warped_subject_block(
                vol, manifest.level, core_origin, core_shape, u, spacing, normalise=True
            )
            acc_i += warped

            factor = cfg.profile_at(manifest.level).lattice_factor
            lat = _pool(u, factor)
            if acc_w is None:
                acc_w = _xp.zeros_like(lat)
                lat_origin = tuple(int(o) // factor for o in core_origin)
                lat_shape = lat.shape[1:]
            acc_w += lat

        stats.append(
            {
                "subject": subject,
                "field_d99_mm": _fields.percentile_magnitude(u, 99.9),
                "folds": _fields.fold_fraction(u, spacing),
            }
        )

    if accumulate:
        write_block(isum, core_origin, acc_i, lead=0)
        if acc_w is not None:
            write_block(wsum, lat_origin, acc_w, lead=1)

    return {
        "chunk_id": chunk_id,
        "subjects": len(manifest.subjects),
        "seconds": time.perf_counter() - t0,
        "stats": stats,
    }


def _pool(u, factor: int):
    return _fields.pool_field(u, factor)

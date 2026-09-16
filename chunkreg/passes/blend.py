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

import time
from typing import Any

import numpy as np
from scipy import ndimage

from .. import backend as _backend
from .. import fields as _fields
from ..config import RunConfig
from ..grid import window_on
from ..store import Field, TaskArray, Volume, read_padded, write_block
from ._common import warped_subject_block
from .manifest import Manifest

__all__ = ["run_blend_task", "ensure_accumulators", "blend_core"]


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
    p = cfg.profile
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


def blend_core(
    cfg: RunConfig, manifest: Manifest, subject: str, chunk_id: int
) -> np.ndarray:
    """Partition-of-unity merge over one core, at level resolution."""
    own = manifest.chunk(chunk_id)
    grid = manifest.grid
    core_shape = tuple(own.core_shape)
    num = np.zeros((3,) + core_shape, dtype=np.float64)
    den = np.zeros(core_shape, dtype=np.float64)

    lo = np.asarray(own.core_origin, dtype=np.int64)
    hi = lo + np.asarray(core_shape, dtype=np.int64)

    cache: dict[int, TaskArray] = {}
    for other in manifest.chunks_touching_core(chunk_id):
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

        # Only the overlap with this core is ever read, so neither the window
        # nor the upsampled residual is built outside it.
        lat = cache[task_id].read(index)
        dense = _upsample_box(lat, cfg.profile.lattice_factor, src)
        wq = window_on(other, grid, cfg.profile, src)
        num[(slice(None),) + dst] += dense * wq
        den[dst] += wq

    if den.min() <= 0:
        raise RuntimeError(
            f"blend denominator reached zero on chunk {chunk_id}; cores should "
            f"tile the level exactly, so this means the manifest is inconsistent"
        )
    out = (num / den).astype(np.float32)

    sigma = cfg.profile.lattice_factor  # one lattice voxel, in level voxels
    if sigma > 0:
        out = np.stack(
            [ndimage.gaussian_filter(out[d], sigma * 0.5, mode="nearest") for d in range(3)]
        )
    return out


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
    isum, wsum = ensure_accumulators(cfg, manifest)

    core_origin = tuple(own.core_origin)
    core_shape = tuple(own.core_shape)
    acc_i = np.zeros(core_shape, dtype=np.float32)
    lat_origin, lat_shape = None, None
    acc_w = None
    stats = []

    for subject in manifest.subjects:
        u = blend_core(cfg, manifest, subject, chunk_id)

        field = Field.open(cfg.field_path(manifest.level, subject), cfg.backend)
        field.write_dense(core_origin, u)

        vol = Volume.open(cfg.subject_path(subject), cfg.backend)
        warped = warped_subject_block(
            vol, manifest.level, core_origin, core_shape, u, spacing, normalise=True
        )
        acc_i += warped

        lat = _pool(u, cfg.profile.lattice_factor)
        if acc_w is None:
            acc_w = np.zeros_like(lat)
            lat_origin = tuple(int(o) // cfg.profile.lattice_factor for o in core_origin)
            lat_shape = lat.shape[1:]
        acc_w += lat

        stats.append(
            {
                "subject": subject,
                "field_d99_mm": _fields.percentile_magnitude(u, 99.9),
                "folds": _fields.fold_fraction(u, spacing),
            }
        )

    write_block(isum, core_origin, acc_i, lead=0)
    if acc_w is not None:
        write_block(wsum, lat_origin, acc_w, lead=1)

    return {
        "chunk_id": chunk_id,
        "subjects": len(manifest.subjects),
        "seconds": time.perf_counter() - t0,
        "stats": stats,
    }


def _pool(u: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return np.asarray(u, dtype=np.float32)
    a = np.asarray(u, dtype=np.float32)
    n = [(-(-s // factor)) for s in a.shape[1:]]
    pad = [(0, 0)] + [(0, t * factor - s) for t, s in zip(n, a.shape[1:])]
    if any(p[1] for p in pad):
        a = np.pad(a, pad, mode="edge")
    return a.reshape(3, n[0], factor, n[1], factor, n[2], factor).mean(axis=(2, 4, 6))

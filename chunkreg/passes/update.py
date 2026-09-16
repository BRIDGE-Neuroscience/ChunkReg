"""The update pass: intensity average, then the unbiasing recentre.

Parallel over template shards. The new template is the mean of the warped
subjects, moved by the negative mean displacement:

    T <- (isum / N) . (id - eps * wsum / N)

The recentre is what makes the template unbiased. Without it the template drifts
toward whichever subject the initial mean happened to resemble, and every field
inherits that bias. With it, the mean displacement is driven toward zero and the
template converges on the population's mean shape, with the bias decaying as
``(1 - eps)^k`` per pass.

The step is also the level's convergence signal: once ``eps * ||u_bar||`` falls
below a voxel, further passes move the template by less than it can represent.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy import ndimage

from .. import backend as _backend
from .. import fields as _fields
from .. import stats as _stats
from ..config import RunConfig
from ..grid import GridSpec
from ..store import Field, Volume, read_padded, write_block
from .manifest import Manifest

__all__ = ["run_update_task", "ensure_recentre"]


def ensure_recentre(cfg: RunConfig, manifest: Manifest) -> Field:
    """Create or open the store that records this pass's template motion.

    Like the accumulators, creating it is shared state while writing it is
    owner-computes, so the pipeline makes it once before dispatching.
    """
    path = cfg.recentre_path(manifest.level)
    if Field.exists(path, cfg.backend):
        return Field.open(path, cfg.backend)
    try:
        return Field.create(
            path, manifest.grid, cfg.profile, backend=cfg.backend, subject=None
        )
    except FileExistsError:
        return Field.open(path, cfg.backend)


def run_update_task(cfg: RunConfig, manifest: Manifest, chunk_id: int) -> dict[str, Any]:
    """Write one template shard from the accumulators."""
    t0 = time.perf_counter()
    b = _backend.get_backend(cfg.backend)
    own = manifest.chunk(chunk_id)
    grid = manifest.grid
    spacing = grid.spacing_mm
    n = max(len(manifest.subjects), 1)
    eps = cfg.levels.shape_update_step

    isum = b.open(cfg.accumulator_path(manifest.level, manifest.iteration, "isum"), "r")
    wsum = b.open(cfg.accumulator_path(manifest.level, manifest.iteration, "wsum"), "r")

    core_origin = tuple(own.core_origin)
    core_shape = tuple(own.core_shape)

    mean_warp = _mean_warp_block(
        wsum, core_origin, core_shape, cfg.profile.lattice_factor, grid, n, eps
    )

    # Reading the average with a margin sized from the recentring step keeps
    # the warp exact at the shard edge instead of pulling in zeros.
    margin = _fields.required_margin(mean_warp, spacing)
    src_origin = tuple(o - margin for o in core_origin)
    src_shape = tuple(s + 2 * margin for s in core_shape)
    block = read_padded(isum, src_origin, src_shape, mode="constant") / n
    updated = _fields.warp_offset(block, (margin, margin, margin), mean_warp, spacing)

    if manifest.level in cfg.levels.sharpen_laplacian_levels:
        updated = _sharpen(updated)

    template = Volume.open(cfg.template_path(manifest.level), cfg.backend)
    template.write_block(0, core_origin, updated)

    # Publish the motion so the next pass can fold it into every seed. Without
    # this the template moves but the fields do not, so the fields simply chase
    # the template and the mean displacement never decays.
    Field.open(cfg.recentre_path(manifest.level), cfg.backend).write_dense(
        core_origin, mean_warp
    )

    # mean_warp is -eps * u_bar, so its magnitude histogram is the recentring
    # step's distribution over this shard.
    return {
        "chunk_id": chunk_id,
        "step_hist": _stats.histogram(_fields.magnitude(mean_warp)).tolist(),
        "seconds": time.perf_counter() - t0,
    }


def _mean_warp_block(
    wsum, core_origin, core_shape, factor: int, grid: GridSpec, n: int, eps: float
) -> np.ndarray:
    """``-eps * mean(u)`` over a core, at level resolution."""
    lat_origin = tuple(int(o) // factor for o in core_origin)
    lat_shape = tuple(int(-(-s // factor)) for s in core_shape)
    margin = 2
    block = read_padded(
        wsum,
        tuple(o - margin for o in lat_origin),
        tuple(s + 2 * margin for s in lat_shape),
        lead=1,
        mode="edge",
    )
    block = block * np.float32(-eps / n)

    lat = grid.coarsened(factor)
    sub_lat = GridSpec(
        tuple(block.shape[1:]),
        lat.spacing_mm,
        tuple(lat.world(tuple(o - margin for o in lat_origin))),
    )
    sub_level = GridSpec(core_shape, grid.spacing_mm, tuple(grid.world(core_origin)))
    return _fields.resample_field(block, sub_lat, sub_level)


def _sharpen(img: np.ndarray, amount: float = 0.5, sigma: float = 1.0) -> np.ndarray:
    """Unsharp mask, countering the blur that averaging N subjects introduces.

    Applied only at coarse levels, where the averaging blur is a large fraction
    of the voxel and the template is a shape prior rather than a deliverable.
    """
    blurred = ndimage.gaussian_filter(img, sigma, mode="nearest")
    return np.clip(img + amount * (img - blurred), 0.0, None).astype(np.float32)


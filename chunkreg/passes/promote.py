"""The promote pass: carry a solved level down to the next one.

Upsampling the template is ordinary resampling. Upsampling the fields is the
step the whole design rests on: because displacements are stored in
millimetres, moving them to a finer lattice changes where they are sampled and
not what they say. The seed a finer level starts from is therefore the coarse
solution exactly, not an approximation of it, and the residual that remains is
genuinely only what the finer resolution newly reveals.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .. import fields as _fields
from ..config import RunConfig
from ..grid import GridSpec, tile
from ..store import Field, Volume
from .manifest import Manifest

__all__ = ["promote_level", "seed_level_zero"]


def promote_level(
    cfg: RunConfig, from_level: int, from_grid: GridSpec, to_grid: GridSpec
) -> dict[str, Any]:
    """Create the next level's template and seed fields from a solved level."""
    t0 = time.perf_counter()
    to_level = from_level + 1

    src_template = Volume.open(cfg.template_path(from_level), cfg.backend)
    dst_template = _template_store(cfg, to_level, to_grid)
    src_block = src_template.read_padded(0, (0, 0, 0), from_grid.shape)
    dst_template.write_block(
        0, (0, 0, 0), _fields.resample_scalar(src_block, from_grid, to_grid)
    )
    dst_template.set_normalisation(*_normalisation(src_template))

    for subject in cfg.subject_ids:
        src = Field.open(cfg.field_path(from_level, subject), cfg.backend)
        dst = Field.create(
            cfg.field_path(to_level, subject),
            to_grid,
            cfg.profile,
            backend=cfg.backend,
            overwrite=True,
            subject=subject,
        )
        for chunk in tile(to_grid, cfg.profile):
            block = src.read_dense_on(
                to_grid, chunk.core_origin, chunk.core_shape
            )
            dst.write_dense(chunk.core_origin, block)

    return {
        "from_level": from_level,
        "to_level": to_level,
        "subjects": len(cfg.subject_ids),
        "seconds": time.perf_counter() - t0,
    }


def _normalisation(vol: Volume) -> tuple[float, float]:
    norm = vol.normalisation
    return norm if norm is not None else (0.0, 1.0)


def _template_store(cfg: RunConfig, level: int, grid: GridSpec) -> Volume:
    """A template is a single-level volume store at its level's grid."""
    path = cfg.template_path(level)
    if Volume.exists(path, cfg.backend):
        return Volume.open(path, cfg.backend)
    return Volume.create(
        path,
        grid,
        cfg.profile,
        dtype="float32",
        backend=cfg.backend,
        overwrite=True,
        n_levels=1,
        provenance={"config": cfg.fingerprint(), "level": level},
    )


def seed_level_zero(cfg: RunConfig, grid: GridSpec, volumes: dict) -> Volume:
    """Build the initial template as the voxelwise mean of the subjects.

    This is the most biased the template ever is: it carries the full
    population shape variance, which is why level 0 budgets the most passes of
    any level to recentre it away.
    """
    template = _template_store(cfg, 0, grid)
    acc = np.zeros(grid.shape, dtype=np.float64)
    lo = np.inf
    hi = -np.inf
    for subject in cfg.subject_ids:
        vol = volumes[subject]
        block = vol.read_padded(0, (0, 0, 0), grid.shape, normalise=True)
        acc += block
        lo = min(lo, float(block.min()))
        hi = max(hi, float(block.max()))
    acc /= max(len(cfg.subject_ids), 1)
    template.write_block(0, (0, 0, 0), acc.astype(np.float32))
    template.set_normalisation(0.0, 1.0)
    for subject in cfg.subject_ids:
        Field.create(
            cfg.field_path(0, subject),
            grid,
            cfg.profile,
            backend=cfg.backend,
            overwrite=True,
            subject=subject,
        )
    return template

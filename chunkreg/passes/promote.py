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

__all__ = ["promote_level", "seed_level_zero", "settle_fields"]


def _recentre(cfg: RunConfig, level: int) -> Field | None:
    """The template motion a level's last pass published, if there was one.

    Within a level the next pass folds this into every seed. The last pass of
    a level has no next pass, so whatever it published is still outstanding
    when the level ends, and both :func:`promote_level` and
    :func:`settle_fields` have to compose it or the fields are left pointing at
    where the template used to be.
    """
    path = cfg.recentre_path(level)
    return Field.open(path, cfg.backend) if Field.exists(path, cfg.backend) else None


def _carry_field(
    cfg: RunConfig,
    src: Field,
    dst: Field,
    grid: GridSpec,
    motion: Field | None,
) -> None:
    """Write ``src`` onto ``dst``'s grid, composing ``motion`` if there is any.

    ``compose(motion, seed)`` is ``motion(x) + seed(x + motion(x))``, the same
    composition a seeded pass applies, so a field promoted through here is the
    one the next level would have been handed had the level run another pass.
    """
    for chunk in tile(grid, cfg.profile):
        origin, shape = chunk.core_origin, chunk.core_shape
        if motion is None:
            dst.write_dense(origin, src.read_dense_on(grid, origin, shape))
            continue
        step = motion.read_dense_on(grid, origin, shape)
        # The composition samples the seed at x + motion(x), so the read has to
        # reach that far past the core or the edge would replicate instead.
        margin = min(
            _fields.required_margin(step, grid.spacing_mm), int(cfg.profile.halo)
        )
        pad_origin = tuple(int(v) - margin for v in origin)
        pad_shape = tuple(int(v) + 2 * margin for v in shape)
        seed = src.read_dense_on(grid, pad_origin, pad_shape)
        wide = motion.read_dense_on(grid, pad_origin, pad_shape)
        composed = _fields.compose(wide, seed, grid.spacing_mm)
        core = (slice(None),) + tuple(
            slice(margin, margin + int(n)) for n in shape
        )
        dst.write_dense(origin, composed[core])


def settle_fields(cfg: RunConfig, level: int, grid: GridSpec) -> dict[str, Field]:
    """Fold the last pass's template motion into the deliverable fields.

    The finest level has no promote to carry its outstanding recentring step,
    so without this the fields a run hands back are stale by one step. Written
    to a separate store rather than in place: the composition reads a margin
    past each core, which would otherwise see cores this same pass had already
    rewritten.
    """
    motion = _recentre(cfg, level)
    out: dict[str, Field] = {}
    for subject in cfg.registered_ids:
        src = Field.open(cfg.field_path(level, subject), cfg.backend)
        # Written even when there is no motion to fold in, so a finished run
        # always leaves its fields in the same place.
        dst = Field.create(
            cfg.final_field_path(subject),
            grid,
            cfg.profile,
            backend=cfg.backend,
            overwrite=True,
            subject=subject,
        )
        _carry_field(cfg, src, dst, grid, motion)
        out[subject] = dst
    return out


def promote_level(
    cfg: RunConfig, from_level: int, from_grid: GridSpec, to_grid: GridSpec
) -> dict[str, Any]:
    """Create the next level's template and seed fields from a solved level."""
    t0 = time.perf_counter()
    to_level = from_level + 1

    if cfg.template_subject is None:
        src_template = Volume.open(cfg.template_path(from_level), cfg.backend)
        dst_template = _template_store(cfg, to_level, to_grid)
        src_block = src_template.read_padded(0, (0, 0, 0), from_grid.shape)
        dst_template.write_block(
            0, (0, 0, 0), _fields.resample_scalar(src_block, from_grid, to_grid)
        )
        dst_template.set_normalisation(*_normalisation(src_template))
    else:
        # A held-fixed template is a particular anatomy, and that anatomy is
        # already stored at every level. Reading its own level beats upsampling
        # the coarse template, which would hand the finer level a blurred
        # target it could never match.
        template_from_subject(cfg, to_level, to_grid)

    motion = _recentre(cfg, from_level)
    for subject in cfg.registered_ids:
        src = Field.open(cfg.field_path(from_level, subject), cfg.backend)
        dst = Field.create(
            cfg.field_path(to_level, subject),
            to_grid,
            cfg.profile,
            backend=cfg.backend,
            overwrite=True,
            subject=subject,
        )
        _carry_field(cfg, src, dst, to_grid, motion)

    return {
        "from_level": from_level,
        "to_level": to_level,
        "subjects": len(cfg.registered_ids),
        "recentre_carried": motion is not None,
        "seconds": time.perf_counter() - t0,
    }


def template_from_subject(cfg: RunConfig, level: int, grid: GridSpec) -> Volume:
    """Write one subject's own level as the template for that level."""
    src = Volume.open(cfg.subject_path(cfg.template_subject), cfg.backend)
    if level >= src.n_levels:
        raise ValueError(
            f"template subject {cfg.template_subject!r} has {src.n_levels} "
            f"levels but the run has at least {level + 1}; re-ingest it with "
            f"the profile this run uses so the pyramids match"
        )
    dst = _template_store(cfg, level, grid)
    for chunk in tile(grid, cfg.profile):
        dst.write_block(
            0,
            chunk.core_origin,
            src.read_padded(
                level, chunk.core_origin, chunk.core_shape, normalise=True
            ),
        )
    dst.set_normalisation(0.0, 1.0)
    return dst


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
    """Build the initial template and the empty fields that seed level 0.

    With no fixed template this is the voxelwise mean of the cohort, which is
    the most biased the template ever is: it carries the full population shape
    variance, which is why level 0 budgets the most passes of any level to
    recentre it away. With ``template_subject`` set it is that subject.
    """
    if cfg.template_subject is not None:
        template = template_from_subject(cfg, 0, grid)
        for subject in cfg.registered_ids:
            Field.create(
                cfg.field_path(0, subject),
                grid,
                cfg.profile,
                backend=cfg.backend,
                overwrite=True,
                subject=subject,
            )
        return template

    template = _template_store(cfg, 0, grid)
    acc = np.zeros(grid.shape, dtype=np.float64)
    sources = cfg.registered_ids
    for subject in sources:
        vol = volumes[subject]
        block = vol.read_padded(0, (0, 0, 0), grid.shape, normalise=True)
        acc += block
    acc /= max(len(sources), 1)
    template.write_block(0, (0, 0, 0), acc.astype(np.float32))
    template.set_normalisation(0.0, 1.0)
    for subject in cfg.registered_ids:
        Field.create(
            cfg.field_path(0, subject),
            grid,
            cfg.profile,
            backend=cfg.backend,
            overwrite=True,
            subject=subject,
        )
    return template

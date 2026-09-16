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
from .. import xp as _xp
from ..config import RunConfig
from ..grid import GridSpec, tile
from ..store import Field, Volume
from .manifest import Manifest

__all__ = ["promote_level", "seed_level_zero", "settle_fields", "carry_task"]


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


def _carry_core(
    cfg: RunConfig,
    src: Field,
    dst: Field,
    grid: GridSpec,
    motion: Field | None,
    origin,
    shape,
) -> None:
    """Write one core of ``src`` onto ``dst``'s grid, composing ``motion``.

    ``compose(motion, seed)`` is ``motion(x) + seed(x + motion(x))``, the same
    composition a seeded pass applies, so a field carried through here is the
    one the next level would have been handed had the level run another pass.
    """
    if motion is None:
        dst.write_dense(origin, src.read_dense_on(grid, origin, shape))
        return
    step = motion.read_dense_on(grid, origin, shape)
    # The composition samples the seed at x + motion(x), so the read has to
    # reach that far past the core or the edge would replicate instead.
    margin = min(_fields.required_margin(step, grid.spacing_mm), int(cfg.profile.halo))
    pad_origin = tuple(int(v) - margin for v in origin)
    pad_shape = tuple(int(v) + 2 * margin for v in shape)
    seed = src.read_dense_on(grid, pad_origin, pad_shape)
    wide = motion.read_dense_on(grid, pad_origin, pad_shape)
    composed = _fields.compose(wide, seed, grid.spacing_mm)
    core = (slice(None),) + tuple(slice(margin, margin + int(n)) for n in shape)
    dst.write_dense(origin, composed[core])


def _resample_template_core(
    src: Volume, src_grid: GridSpec, dst: Volume, dst_grid: GridSpec, chunk
) -> None:
    """Resample the template into one destination core.

    Reads only the source voxels the core's samples fall between, plus a
    margin, with edges replicated exactly as a whole-volume resample would.
    """
    from ..store import read_padded

    margin = 2
    origin = np.asarray(chunk.core_origin, dtype=np.int64)
    shape = np.asarray(chunk.core_shape, dtype=np.int64)
    lo = np.floor(src_grid.voxel(dst_grid.world(origin))).astype(np.int64) - margin
    hi = np.ceil(src_grid.voxel(dst_grid.world(origin + shape - 1))).astype(np.int64)
    ext = hi + margin + 1 - lo
    block = _xp.put(read_padded(src.array(0), lo, ext, mode="edge"))
    sub_src = GridSpec(
        tuple(int(v) for v in ext), src_grid.spacing_mm, tuple(src_grid.world(lo))
    )
    sub_dst = GridSpec(
        tuple(int(v) for v in shape), dst_grid.spacing_mm, tuple(dst_grid.world(origin))
    )
    dst.write_block(0, chunk.core_origin, _fields.resample_scalar(block, sub_src, sub_dst))


# --------------------------------------------------------------------------- #
# Carrying as tasks: one core per task, so a multi-GPU runner can spread it
# --------------------------------------------------------------------------- #
def _grid_json(g: GridSpec) -> dict:
    return {"shape": list(g.shape), "spacing_mm": g.spacing_mm, "origin_mm": list(g.origin_mm)}


def _grid_from(d: dict) -> GridSpec:
    return GridSpec(tuple(d["shape"]), d["spacing_mm"], tuple(d["origin_mm"]))


def carry_task(cfg: RunConfig, spec: dict, task_id: int) -> dict[str, Any]:
    """Carry one core. The unit of work a GPU worker runs for promote and settle.

    ``spec["kind"]`` is ``"field"`` or ``"template"``. For fields the task id
    is ``subject_index * n_chunks + chunk_id``; for the template it is the
    chunk id. Every task writes only its own core, so tasks never collide.
    """
    dst_grid = _grid_from(spec["dst_grid"])
    chunks = tile(dst_grid, cfg.profile)
    if spec["kind"] == "template":
        chunk = chunks[int(task_id)]
        src = Volume.open(cfg.template_path(spec["src_level"]), cfg.backend)
        dst = Volume.open(cfg.template_path(spec["dst_level"]), cfg.backend)
        _resample_template_core(src, _grid_from(spec["src_grid"]), dst, dst_grid, chunk)
        return {"task_id": int(task_id)}

    subject = spec["subjects"][int(task_id) // len(chunks)]
    chunk = chunks[int(task_id) % len(chunks)]
    src = Field.open(cfg.field_path(spec["src_level"], subject), cfg.backend)
    dst_path = (
        cfg.final_field_path(subject)
        if spec["final"]
        else cfg.field_path(spec["dst_level"], subject)
    )
    dst = Field.open(dst_path, cfg.backend)
    motion = (
        None
        if spec["motion_level"] is None
        else Field.open(cfg.recentre_path(spec["motion_level"]), cfg.backend)
    )
    _carry_core(cfg, src, dst, dst_grid, motion, chunk.core_origin, chunk.core_shape)
    return {"task_id": int(task_id)}


def _spreads(runner) -> bool:
    """Can this runner run promotion work on its own workers?"""
    return bool(getattr(runner, "runs_carry", False))


def _run_carry(cfg: RunConfig, runner, spec: dict, n_tasks: int) -> None:
    """Run carry tasks on the runner's workers, or here, core by core."""
    if _spreads(runner):
        report = runner.run("carry", None, range(n_tasks), spec=spec)
        report.raise_for_failures()
        return
    for task_id in range(n_tasks):
        carry_task(cfg, spec, task_id)


def _carry_fields(
    cfg: RunConfig,
    src_level: int,
    dst_level: int,
    grid: GridSpec,
    final: bool,
    runner=None,
) -> dict[str, Field]:
    """Create each subject's destination field, then fill it core by core."""
    motion = _recentre(cfg, src_level)
    out: dict[str, Field] = {}
    for subject in cfg.registered_ids:
        out[subject] = Field.create(
            cfg.final_field_path(subject) if final else cfg.field_path(dst_level, subject),
            grid,
            cfg.profile,
            backend=cfg.backend,
            overwrite=True,
            subject=subject,
        )
    spec = {
        "kind": "field",
        "src_level": int(src_level),
        "dst_level": int(dst_level),
        "dst_grid": _grid_json(grid),
        "subjects": list(cfg.registered_ids),
        "final": bool(final),
        "motion_level": None if motion is None else int(src_level),
    }
    _run_carry(cfg, runner, spec, len(cfg.registered_ids) * len(tile(grid, cfg.profile)))
    return out


def settle_fields(
    cfg: RunConfig, level: int, grid: GridSpec, runner=None
) -> dict[str, Field]:
    """Fold the last pass's template motion into the deliverable fields.

    The finest level has no promote to carry its outstanding recentring step,
    so without this the fields a run hands back are stale by one step. Written
    to a separate store rather than in place: the composition reads a margin
    past each core, which would otherwise see cores this same pass had already
    rewritten. Written even when there is no motion to fold in, so a finished
    run always leaves its fields in the same place.

    With a runner that spreads work over GPUs, each core is a task there.
    """
    return _carry_fields(cfg, level, level, grid, final=True, runner=runner)


def promote_level(
    cfg: RunConfig,
    from_level: int,
    from_grid: GridSpec,
    to_grid: GridSpec,
    to_level: int | None = None,
    runner=None,
) -> dict[str, Any]:
    """Create a finer level's template and seed fields from a solved level.

    ``to_level`` defaults to the next level. It can be further down when
    ``levels.run`` skips levels: fields are millimetres, so carrying them
    across a gap is the same resampling as carrying them one level.

    With a runner that spreads work over GPUs, every core of the template and
    of each field is a task there, instead of a loop in this process.
    """
    t0 = time.perf_counter()
    to_level = from_level + 1 if to_level is None else int(to_level)
    if to_level <= from_level:
        raise ValueError(
            f"cannot promote level {from_level} to level {to_level}; promotion "
            f"only goes to finer levels"
        )

    if cfg.template_subject is None:
        src_template = Volume.open(cfg.template_path(from_level), cfg.backend)
        dst_template = _template_store(cfg, to_level, to_grid)
        dst_template.set_normalisation(*_normalisation(src_template))
        spec = {
            "kind": "template",
            "src_level": int(from_level),
            "dst_level": int(to_level),
            "src_grid": _grid_json(from_grid),
            "dst_grid": _grid_json(to_grid),
        }
        _run_carry(cfg, runner, spec, len(tile(to_grid, cfg.profile)))
    else:
        # A held-fixed template is a particular anatomy, and that anatomy is
        # already stored at every level. Reading its own level beats upsampling
        # the coarse template, which would hand the finer level a blurred
        # target it could never match.
        template_from_subject(cfg, to_level, to_grid)

    motion = _recentre(cfg, from_level)
    _carry_fields(cfg, from_level, to_level, to_grid, final=False, runner=runner)

    return {
        "from_level": from_level,
        "to_level": to_level,
        "subjects": len(cfg.registered_ids),
        "recentre_carried": motion is not None,
        "seconds": time.perf_counter() - t0,
    }


def _resample_template(
    cfg: RunConfig,
    src: Volume,
    src_grid: GridSpec,
    dst: Volume,
    dst_grid: GridSpec,
) -> None:
    """Resample a template onto a finer grid, one destination core at a time."""
    for chunk in tile(dst_grid, cfg.profile):
        _resample_template_core(src, src_grid, dst, dst_grid, chunk)


def _carry_field(
    cfg: RunConfig,
    src: Field,
    dst: Field,
    grid: GridSpec,
    motion: Field | None,
) -> None:
    """Write ``src`` onto ``dst``'s grid core by core, composing ``motion``."""
    for chunk in tile(grid, cfg.profile):
        _carry_core(cfg, src, dst, grid, motion, chunk.core_origin, chunk.core_shape)


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
    acc = _xp.zeros(grid.shape, dtype=np.float64)
    sources = cfg.registered_ids
    for subject in sources:
        vol = volumes[subject]
        block = vol.read_padded(0, (0, 0, 0), grid.shape, normalise=True)
        acc += block
    acc /= max(len(sources), 1)
    template.write_block(0, (0, 0, 0), _xp.to_float32(acc))
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

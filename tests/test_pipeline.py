"""End to end: does the architecture actually recover a cohort?

Three subjects are built as known smooth deformations of one hidden volume.
A correct pipeline must drive each subject's field toward the inverse of the
deformation that made it, agree between chunks across their shared boundaries,
and stop on its own once the residual fits the next level's halo.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from chunkreg import fields
from chunkreg.config import (
    LevelPolicy,
    Resources,
    RunConfig,
    SlurmConfig,
    StageSpec,
    SubjectSpec,
)
from chunkreg.engines import get_engine
from chunkreg.grid import GridSpec, Profile, pyramid, tile
from chunkreg.passes import build_manifest, run_blend_task, run_register_task
from chunkreg.passes.promote import seed_level_zero
from chunkreg.pipelines import build_template
from chunkreg.runners import LocalRunner
from chunkreg.store import Field, Volume, ingest_array

from .conftest import blob

SPACING = 0.05
SHAPE = (64, 64, 64)


def smooth_field(shape, amp_mm, seed) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(0, 1, (3,) + tuple(shape)).astype(np.float32)
    u = np.stack([ndimage.gaussian_filter(raw[d], 8.0) for d in range(3)])
    return (u * (amp_mm / max(np.abs(u).max(), 1e-8))).astype(np.float32)


@pytest.fixture
def cohort(tmp_path, small_profile):
    """Three subjects, each a known deformation of one hidden volume."""
    truth_vol = blob(SHAPE, seed=42, smooth=2.0)
    warps = {
        "s01": smooth_field(SHAPE, 5 * SPACING, 1),
        "s02": smooth_field(SHAPE, 5 * SPACING, 2),
        "s03": smooth_field(SHAPE, 5 * SPACING, 3),
    }
    for sid, w in warps.items():
        data = fields.warp(truth_vol, w, SPACING)
        ingest_array(
            (data * 4000).astype(np.uint16),
            f"subjects/{sid}.zarr",
            SPACING,
            small_profile,
        )
    return truth_vol, warps


def make_cfg(profile, **kw) -> RunConfig:
    base = dict(
        root=".",
        subjects=(
            SubjectSpec("s01", "subjects/s01.zarr"),
            SubjectSpec("s02", "subjects/s02.zarr"),
            SubjectSpec("s03", "subjects/s03.zarr"),
        ),
        profile=profile,
        profile_name="test",
        backend="memory",
        gpu_mem_gb=1000.0,
        levels=LevelPolicy(
            caps=(3, 2, 2),
            level0_stages=(
                StageSpec(kind="greedy", scales=(4, 2, 1), iterations=(40, 25, 15)),
            ),
            seeded_stages=(
                StageSpec(kind="greedy", scales=(2, 1), iterations=(25, 15)),
            ),
            sharpen_laplacian_levels=(),
        ),
        slurm=SlurmConfig(register=Resources(gpus=1, chunks_per_task=4)),
    )
    base.update(kw)
    return RunConfig(**base)


# --------------------------------------------------------------------------- #
# The pieces, before the whole
# --------------------------------------------------------------------------- #
def test_level_zero_is_a_single_chunk(cohort, small_profile):
    """The top of the pyramid is where the global component is estimated."""
    levels = pyramid(GridSpec(SHAPE, SPACING), small_profile)
    assert len(tile(levels[0], small_profile)) == 1
    assert len(levels) > 1, "test needs a real pyramid"


def test_manifest_covers_every_subject_and_chunk(cohort, small_profile):
    cfg = make_cfg(small_profile)
    grid = pyramid(GridSpec(SHAPE, SPACING), small_profile)[-1]
    m = build_manifest(cfg, 2, 0, grid)
    assert m.n_entries == len(m.chunks) * 3
    seen = {(e.subject, e.chunk_id) for t in m.tasks for e in t.entries}
    assert len(seen) == m.n_entries


def test_blend_read_set_is_local(cohort, small_profile):
    cfg = make_cfg(small_profile)
    grid = pyramid(GridSpec(SHAPE, SPACING), small_profile)[-1]
    m = build_manifest(cfg, 2, 0, grid)
    interior = max(m.chunks, key=lambda c: min(c.index))
    touching = m.chunks_touching_core(interior.id)
    assert interior in touching
    assert len(touching) <= 27


def test_register_task_is_idempotent(cohort, small_profile):
    cfg = make_cfg(small_profile)
    levels = pyramid(GridSpec(SHAPE, SPACING), small_profile)
    seed_level_zero(cfg, levels[0], {s: Volume.open(f"subjects/{s}.zarr") for s in cfg.subject_ids})
    m = build_manifest(cfg, 0, 0, levels[0])
    first = run_register_task(cfg, m, 0, engine=get_engine("demons"))
    second = run_register_task(cfg, m, 0, engine=get_engine("demons"))
    assert not first["skipped"]
    assert second["skipped"], "a completed task array must short-circuit a rerun"


def test_blend_agrees_with_its_neighbours_across_a_core_boundary(cohort, small_profile):
    """The partition of unity must not leave a seam at a chunk edge."""
    cfg = make_cfg(small_profile)
    levels = pyramid(GridSpec(SHAPE, SPACING), small_profile)
    vols = {s: Volume.open(f"subjects/{s}.zarr") for s in cfg.subject_ids}
    seed_level_zero(cfg, levels[0], vols)
    from chunkreg.passes.promote import promote_level

    for k in range(len(levels) - 1):
        promote_level(cfg, k, levels[k], levels[k + 1])

    level = len(levels) - 1
    grid = levels[level]
    m = build_manifest(cfg, level, 0, grid)
    for tid in range(m.n_tasks):
        run_register_task(cfg, m, tid, engine=get_engine("demons"))
    for cid in range(len(m.chunks)):
        run_blend_task(cfg, m, cid)

    f = Field.open(cfg.field_path(level, "s01"), "memory")
    dense = f.read_dense((0, 0, 0), grid.shape)
    # A seam would show as a step at a core boundary far above the field's own
    # smooth variation. Compare the jump across boundaries to the typical jump.
    core = small_profile.core
    grad = np.abs(np.diff(dense, axis=1))
    boundaries = [i for i in range(core, grid.shape[0], core)]
    assert boundaries, "test needs at least one interior boundary"
    at_edge = np.mean([grad[:, i - 1].mean() for i in boundaries])
    typical = np.median(grad)
    assert at_edge < 5 * max(typical, 1e-9), (
        f"seam at chunk boundary: {at_edge:.2e} vs typical {typical:.2e}"
    )


# --------------------------------------------------------------------------- #
# The whole loop
# --------------------------------------------------------------------------- #
def test_build_template_recovers_the_cohort(cohort, small_profile):
    truth_vol, warps = cohort
    cfg = make_cfg(small_profile)
    result = build_template(cfg, runner=LocalRunner())

    assert set(result.fields) == set(warps)
    assert result.levels[0].level == 0
    assert result.n_passes >= len(result.levels)

    grid = result.template.grid(0)
    inner = (slice(None),) + tuple(slice(16, -16) for _ in range(3))
    for sid, truth in warps.items():
        u = result.fields[sid].read_dense((0, 0, 0), grid.shape)
        leftover = fields.compose(u, truth, SPACING)
        before = np.abs(truth[inner]).mean()
        after = np.abs(leftover[inner]).mean()
        assert after < before, (
            f"{sid}: composing the recovered field with the truth left "
            f"{after:.4f} mm, no better than the original {before:.4f} mm"
        )


def test_template_is_sharper_than_the_subject_average(cohort, small_profile):
    """The point of registering before averaging: detail survives."""
    truth_vol, warps = cohort
    cfg = make_cfg(small_profile)
    result = build_template(cfg, runner=LocalRunner())

    grid = result.template.grid(0)
    final = result.template.read_padded(0, (0, 0, 0), grid.shape)
    naive = np.mean(
        [
            Volume.open(f"subjects/{s}.zarr").read_padded(
                small_profile and 0 or 0, (0, 0, 0), grid.shape, normalise=True
            )
            for s in cfg.subject_ids
        ],
        axis=0,
    ) if False else None  # naive mean is built below at the right level

    subs = []
    for s in cfg.subject_ids:
        v = Volume.open(f"subjects/{s}.zarr")
        subs.append(v.read_padded(v.n_levels - 1, (0, 0, 0), grid.shape, normalise=True))
    naive = np.mean(subs, axis=0)

    box = tuple(slice(16, -16) for _ in range(3))
    sharp = lambda a: float(np.abs(np.gradient(a[box])).mean())
    assert sharp(final) > sharp(naive)


def test_stopping_predicate_reads_both_conditions():
    """A level exits only when the residual fits the next halo *and* the
    template has stopped moving; either alone is not enough."""
    from chunkreg.pipelines.groupwise import PassReport, _should_stop

    cfg = make_cfg(small_profile_for_predicate())
    grid = GridSpec((64, 64, 64), 0.05)
    threshold = 0.20

    def report(d99, step):
        return PassReport(0, 0, d99, 0.0, step, 0.0, 0, 0, 0.0)

    stop, why = _should_stop(cfg, report(0.05, 0.01), threshold, grid, 0, 9)
    assert stop and why == "residual fits the next halo"

    # Residual small but template still moving: keep going.
    assert not _should_stop(cfg, report(0.05, 0.9), threshold, grid, 0, 9)[0]
    # Template settled but residual too large for the next halo: keep going.
    assert not _should_stop(cfg, report(9.0, 0.01), threshold, grid, 0, 9)[0]
    # The finest level has no successor, so only the template matters.
    stop, why = _should_stop(cfg, report(9.0, 0.01), 0.0, grid, 0, 9)
    assert stop and why == "template converged"
    # The cap is always a backstop.
    assert _should_stop(cfg, report(9.0, 9.0), threshold, grid, 8, 9) == (
        True,
        "cap reached",
    )


def small_profile_for_predicate() -> Profile:
    return Profile(
        core=32, halo=12, inner_chunk=8, lattice_factor=2, channels=1,
        features="intensity", r_f=0, k=3, sigma_g=0.5, sigma_w=0.5, scales=(2, 1),
    )


def test_the_loop_stops_early_when_the_thresholds_are_met(cohort, small_profile):
    """The mechanism must actually exit the loop, not just report a number.

    Thresholds are loosened here because the reference engine is a stand-in:
    the production thresholds are calibrated for FireANTs, and what this test
    has to prove is that the loop honours them.
    """
    from dataclasses import replace

    cfg = make_cfg(small_profile)
    cfg = replace(
        cfg,
        levels=replace(cfg.levels, caps=(5, 5), residual_frac=50.0, ubar_vox=100.0),
    )
    result = build_template(cfg, runner=LocalRunner())
    for lv in result.levels:
        assert lv.final.stopped
        assert lv.final.reason != "cap reached", (
            f"level {lv.level} ran to its cap despite thresholds being met"
        )
        assert lv.n_passes == 1, "a satisfied level should exit after one pass"


def test_recentring_shrinks_the_template_step(cohort, small_profile):
    """Without recentring the fields chase the template and the step grows."""
    cfg = make_cfg(small_profile)
    result = build_template(cfg, runner=LocalRunner())
    fine = result.levels[-1].passes
    assert len(fine) >= 2, "need at least two passes to see a trend"
    assert fine[1].step_mm < fine[0].step_mm, (
        f"template step did not fall between passes: "
        f"{[p.step_mm for p in fine]}"
    )


def test_fields_stay_diffeomorphic(cohort, small_profile):
    cfg = make_cfg(small_profile)
    result = build_template(cfg, runner=LocalRunner())
    grid = result.template.grid(0)
    for sid, f in result.fields.items():
        u = f.read_dense((0, 0, 0), grid.shape)
        folds = fields.fold_fraction(u, grid.spacing_mm)
        assert folds < 0.01, f"{sid} folded on {folds:.2%} of voxels"


def test_scratch_is_cleaned_up(cohort, small_profile):
    from chunkreg import backend as _backend

    cfg = make_cfg(small_profile)
    build_template(cfg, runner=LocalRunner())
    # Matched against the run's own scratch directory, not the substring
    # "scratch": store keys are absolute, so a temporary directory that happens
    # to carry the word (this test's own, for one) would match every path.
    scratch = str(cfg.scratch_root.resolve())
    left = [p for p in _backend.get_backend("memory").paths() if p.startswith(scratch)]
    assert left == [], f"retention left {len(left)} transient objects behind"


def test_parallel_workers_give_the_same_answer(cohort, small_profile):
    """Owner-computes means task order cannot matter."""
    cfg = make_cfg(small_profile)
    serial = build_template(cfg, runner=LocalRunner(workers=1))
    grid = serial.template.grid(0)
    a = serial.template.read_padded(0, (0, 0, 0), grid.shape).copy()
    fa = {s: f.read_dense((0, 0, 0), grid.shape).copy() for s, f in serial.fields.items()}

    from chunkreg import backend as _backend

    _backend.get_backend("memory").clear()
    _backend._MEM_JSON.clear()
    # Pass records and seed markers are real files, and a run that finds them
    # resumes instead of starting over; this comparison needs a fresh run.
    import shutil

    shutil.rmtree(cfg.root_path / "levels")
    truth_vol, warps = cohort  # re-ingest into the cleared store
    for sid, w in warps.items():
        data = fields.warp(truth_vol, w, SPACING)
        ingest_array((data * 4000).astype(np.uint16), f"subjects/{sid}.zarr", SPACING, small_profile)

    par = build_template(cfg, runner=LocalRunner(workers=4))
    b = par.template.read_padded(0, (0, 0, 0), grid.shape)
    assert np.allclose(a, b, atol=1e-5)
    for s, f in par.fields.items():
        assert np.allclose(fa[s], f.read_dense((0, 0, 0), grid.shape), atol=1e-5)


def test_a_failing_task_is_reported_not_swallowed():
    runner = LocalRunner()
    report = runner.run("register", lambda tid: 1 / 0, [0, 1])
    assert not report.ok
    assert len(report.failed) == 2
    with pytest.raises(RuntimeError, match="register tasks failed"):
        report.raise_for_failures()

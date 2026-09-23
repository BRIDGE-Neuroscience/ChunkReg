"""The residual map and the per-pass slice sheet.

The pipeline used to report where it failed only as a percentile. These
check that the register pass now leaves a spatial record, that a chunk which
gave up says so honestly (zero for no tissue, NaN for no acceptable solution),
and that every pass leaves a sheet a person can open without pulling a
volume.
"""

from __future__ import annotations

import numpy as np
import pytest

from chunkreg import qc
from chunkreg.grid import GridSpec, pyramid
from chunkreg.passes import build_manifest
from chunkreg.pipelines import build_template
from chunkreg.runners import LocalRunner
from chunkreg.store import Volume

from .test_pipeline import SHAPE, SPACING, cohort, make_cfg  # noqa: F401 - fixture

PIL = pytest.importorskip("PIL")


def _level_and_manifest(cfg, small_profile, level=1):
    grids = pyramid(GridSpec(SHAPE, SPACING), small_profile)
    return grids[level], build_manifest(cfg, level, 0, grids[level])


# --------------------------------------------------------------------------- #
# Residual stores
# --------------------------------------------------------------------------- #
def test_one_residual_store_per_subject_on_the_lattice(cohort, small_profile):
    cfg = make_cfg(small_profile)
    grid, m = _level_and_manifest(cfg, small_profile)
    qc.ensure_residual_stores(cfg, m)
    lattice = grid.coarsened(small_profile.lattice_factor)
    for subject in cfg.registered_ids:
        vol = Volume.open(qc.residual_path(cfg, m.level, subject), cfg.backend)
        assert vol.n_levels == 1
        assert tuple(vol.native_grid.shape) == tuple(lattice.shape)
        assert vol.native_grid.spacing_mm == pytest.approx(lattice.spacing_mm)
    qc.ensure_residual_stores(cfg, m)  # idempotent


def test_a_registered_chunk_writes_its_residual_magnitude(cohort, small_profile):
    cfg = make_cfg(small_profile)
    _, m = _level_and_manifest(cfg, small_profile)
    qc.ensure_residual_stores(cfg, m)
    entry = m.tasks[0].entries[0]
    chunk = m.chunk(entry.chunk_id)

    # A residual of a known constant magnitude: (3, 4, 0) mm everywhere.
    residual = np.zeros((3,) + tuple(entry.pad_shape), dtype=np.float32)
    residual[0], residual[1] = 3.0, 4.0
    qc.write_residual_core(cfg, m, entry, residual, None)

    f = small_profile.lattice_factor
    lo = tuple(o // f for o in chunk.core_origin)
    shape = tuple(-(-s // f) for s in chunk.core_shape)
    got = Volume.open(qc.residual_path(cfg, m.level, entry.subject), cfg.backend)
    block = np.asarray(got.array(0)[tuple(slice(o, o + s) for o, s in zip(lo, shape))])
    np.testing.assert_allclose(block, 5.0, atol=1e-5)


def test_a_chunk_that_gave_up_is_written_honestly(cohort, small_profile):
    """Zero means nothing to register; NaN means nothing acceptable found.

    Writing zero for both would make a chunk that exhausted the fold ladder
    look converged, which is the one reading the map must never invite.
    """
    cfg = make_cfg(small_profile)
    _, m = _level_and_manifest(cfg, small_profile)
    qc.ensure_residual_stores(cfg, m)
    air, folded = m.tasks[0].entries[0], m.tasks[-1].entries[-1]
    assert air.chunk_id != folded.chunk_id or air.subject != folded.subject

    qc.write_residual_core(cfg, m, air, None, "tissue")
    qc.write_residual_core(cfg, m, folded, None, "folds")

    def core_of(entry):
        chunk = m.chunk(entry.chunk_id)
        f = small_profile.lattice_factor
        lo = tuple(o // f for o in chunk.core_origin)
        shape = tuple(-(-s // f) for s in chunk.core_shape)
        vol = Volume.open(qc.residual_path(cfg, m.level, entry.subject), cfg.backend)
        return np.asarray(vol.array(0)[tuple(slice(o, o + s) for o, s in zip(lo, shape))])

    assert np.all(core_of(air) == 0.0)
    assert np.all(np.isnan(core_of(folded)))


# --------------------------------------------------------------------------- #
# The sheet
# --------------------------------------------------------------------------- #
def test_every_pass_leaves_a_sheet(cohort, small_profile):
    from PIL import Image

    cfg = make_cfg(small_profile)
    result = build_template(cfg, runner=LocalRunner())
    for lv in result.levels:
        for p in lv.passes:
            png = qc.slices_path(cfg, lv.level, p.iteration)
            assert png.exists(), f"no sheet for level {lv.level} pass {p.iteration}"
            im = Image.open(png)
            assert im.mode == "RGB"
            # Three panels across, two rows down, plus labels and margins.
            assert im.width > 3 * 32 and im.height > 2 * 32
            assert im.width > im.height


def test_a_chunk_some_subjects_gave_up_on_is_tinted_not_blanked(cohort, small_profile):
    """The panel has to show both things at once.

    Where a subject gave up matters, and so does the residual the others left
    there. Colouring the voxel magenta outright answers the first question by
    destroying the second.
    """
    from PIL import Image

    cfg = make_cfg(small_profile)
    grid, m = _level_and_manifest(cfg, small_profile, level=0)
    _seeded_level_zero(cfg, grid)
    qc.ensure_residual_stores(cfg, m)

    entries = [e for t in m.tasks for e in t.entries]
    for e in entries:
        small = np.full((3,) + tuple(e.pad_shape), 0.1, dtype=np.float32)
        qc.write_residual_core(cfg, m, e, small, None)
    qc.write_residual_core(cfg, m, entries[-1], None, "folds")  # one of several

    png = qc.render_pass(cfg, 0, 0, grid, 1.0,
                         report=_seeded_report(len(entries), 1))
    a = np.asarray(Image.open(png)).astype(np.int16)
    lower = a[a.shape[0] // 2:]
    r, g, b = lower[..., 0], lower[..., 1], lower[..., 2]

    assert ((r > g) & (b > g)).any(), "the subject that gave up must show"
    assert not ((r > 250) & (g < 10) & (b > 250)).any(), (
        "the others solved here, so the voxel must not be fully magenta"
    )


def _magentaness(rgb) -> float:
    a = np.asarray(rgb, dtype=np.int32)
    return float((a[..., 0] + a[..., 2] - 2 * a[..., 1]).mean())


def test_a_partial_give_up_marks_a_region_without_hiding_its_value():
    """Full magenta is reserved for "no subject solved this".

    A blend straight by the fraction put a third of the way to magenta across
    a whole panel and buried the residual under it, which is the opposite of
    what the panel is for.
    """
    from chunkreg.qc import _heat

    value = np.full((8, 8), 0.5, dtype=np.float32)
    plain = _heat(value, 1.0, None)
    partial = _heat(value, 1.0, np.full((8, 8), 1 / 3, dtype=np.float32))
    total = _heat(np.full((8, 8), np.nan, np.float32),
                  1.0, np.ones((8, 8), np.float32))

    assert _magentaness(plain) < _magentaness(partial) < _magentaness(total)
    assert not np.array_equal(partial, total), "a partial give-up is not a total one"
    # The value still drives the colour: a bigger residual still reads bigger.
    bigger = _heat(np.full((8, 8), 0.9, np.float32), 1.0,
                   np.full((8, 8), 1 / 3, dtype=np.float32))
    assert bigger[..., 1].mean() > partial[..., 1].mean()


def test_a_missing_residual_store_does_not_stop_the_sheet(cohort, small_profile):
    """A sheet with the template alone still beats no sheet."""
    cfg = make_cfg(small_profile)
    grid, _ = _level_and_manifest(cfg, small_profile, level=0)
    from chunkreg.passes.promote import seed_level_zero

    seed_level_zero(cfg, grid, {s: Volume.open(cfg.subject_path(s), cfg.backend)
                                for s in cfg.subject_ids})
    png = qc.render_pass(cfg, 0, 0, grid, d_max_mm=None)
    assert png.exists()


def test_a_resumed_run_that_adds_a_pass_writes_that_pass_s_qc(cohort, small_profile):
    """Raising a cap on a finished level is how a solved run is asked for a
    residual map it predates.

    The finished passes replay from their records and do no work, so they
    render nothing; the pass the raised cap allows runs under the current
    code and writes both the sheet and the residual stores.
    """
    from chunkreg.config import LevelPolicy, StageSpec

    def cfg_with(caps):
        return make_cfg(
            small_profile,
            levels=LevelPolicy(
                caps=caps,
                level0_stages=(
                    StageSpec(kind="greedy", scales=(4, 2, 1), iterations=(8, 6, 4)),
                ),
                seeded_stages=(
                    StageSpec(kind="greedy", scales=(2, 1), iterations=(6, 4)),
                ),
                sharpen_laplacian_levels=(),
            ),
        )

    first = cfg_with((1, 1))
    result = build_template(first, runner=LocalRunner())
    last = result.levels[-1].level
    assert last > 0, "test needs a seeded level"
    assert qc.slices_path(first, last, 0).exists()
    assert not qc.slices_path(first, last, 1).exists()

    second = cfg_with((1, 2))
    build_template(second, runner=LocalRunner())

    assert qc.slices_path(second, last, 1).exists(), (
        "the pass the raised cap allowed must leave a sheet"
    )
    for subject in second.registered_ids:
        assert Volume.exists(qc.residual_path(second, last, subject), second.backend)


# --------------------------------------------------------------------------- #
# Telling collapse apart from "nothing was computed"
# --------------------------------------------------------------------------- #
def _seeded_report(entries: int, gave_up: int, written: bool = True):
    from chunkreg.pipelines.groupwise import PassReport

    return PassReport(
        level=0, iteration=0, d99_mm=0.0, ubar_mm=0.0, step_mm=0.0,
        fold_frac=0.0, retries=0, skipped=gave_up, seconds=0.0,
        entries=entries, seeded_folds=gave_up, residual_written=written,
    )


def _seeded_level_zero(cfg, grid):
    from chunkreg.passes.promote import seed_level_zero

    seed_level_zero(cfg, grid, {s: Volume.open(cfg.subject_path(s), cfg.backend)
                                for s in cfg.subject_ids})


def test_a_level_where_every_chunk_gave_up_says_so_on_the_sheet(cohort, small_profile):
    """An all-magenta panel is ambiguous without the count beside it.

    It reads the same whether the level collapsed or nothing was computed, and
    those call for opposite responses.
    """
    from PIL import Image

    cfg = make_cfg(small_profile)
    grid, m = _level_and_manifest(cfg, small_profile, level=0)
    _seeded_level_zero(cfg, grid)
    qc.ensure_residual_stores(cfg, m)
    entries = [e for t in m.tasks for e in t.entries]
    for e in entries:
        qc.write_residual_core(cfg, m, e, None, "folds")

    png = qc.render_pass(cfg, 0, 0, grid, 1.0,
                         report=_seeded_report(len(entries), len(entries)))
    a = np.asarray(Image.open(png))
    magenta = (a[..., 0] > 200) & (a[..., 1] < 60) & (a[..., 2] > 200)
    assert magenta.sum() > 1000, "a collapsed level must still be drawn"

    from chunkreg.qc import _residual_label

    label = _residual_label(_seeded_report(len(entries), len(entries)),
                            [np.zeros((4, 4))], False, 3, 1.0, 0.0)
    assert f"0/{len(entries)} subject-chunks solved" in label
    assert f"{len(entries)} gave up on folds" in label


def test_a_pass_that_reused_its_tasks_is_not_drawn_as_this_pass_s(cohort, small_profile):
    """A reused register task does not re-solve, so it writes no residual.

    Drawing the stores anyway would present an earlier pass's map, or an
    empty one, as the current state.
    """
    from PIL import Image

    cfg = make_cfg(small_profile)
    grid, m = _level_and_manifest(cfg, small_profile, level=0)
    _seeded_level_zero(cfg, grid)
    qc.ensure_residual_stores(cfg, m)
    for e in (e for t in m.tasks for e in t.entries):
        qc.write_residual_core(cfg, m, e, None, "folds")  # would be magenta

    png = qc.render_pass(cfg, 0, 0, grid, 1.0,
                         report=_seeded_report(4, 4, written=False))
    a = np.asarray(Image.open(png))
    magenta = (a[..., 0] > 200) & (a[..., 1] < 60) & (a[..., 2] > 200)
    assert not magenta.any(), "stale data must not be drawn as this pass's"

    from chunkreg.qc import _residual_label

    label = _residual_label(_seeded_report(4, 4, written=False), None, True, 3, 1.0, 0.0)
    assert "NOT recomputed" in label and "reused" in label


def test_a_reused_register_pass_reports_that_it_wrote_no_residual(cohort, small_profile):
    """The flag has to come from the run, not be set by hand in a test."""
    cfg = make_cfg(small_profile)
    result = build_template(cfg, runner=LocalRunner())
    for lv in result.levels:
        for p in lv.passes:
            assert p.residual_written, "a pass that ran must report its map written"
            assert p.entries > 0, "a pass that ran must count its chunks"


def test_one_subject_giving_up_does_not_erase_what_the_others_solved(cohort, small_profile):
    """np.maximum propagates NaN; at a level where half the pairs gave up that
    turned a partly-working map into a uniform magenta field."""
    from PIL import Image

    cfg = make_cfg(small_profile)
    grid, m = _level_and_manifest(cfg, small_profile, level=0)
    _seeded_level_zero(cfg, grid)
    qc.ensure_residual_stores(cfg, m)

    entries = [e for t in m.tasks for e in t.entries]
    assert len({e.subject for e in entries}) > 1, "test needs several subjects"
    for e in entries:
        if e.subject == entries[0].subject:
            qc.write_residual_core(cfg, m, e, None, "folds")   # one gave up
        else:
            u = np.zeros((3,) + tuple(e.pad_shape), dtype=np.float32)
            u[0] = 0.5
            qc.write_residual_core(cfg, m, e, u, None)         # the rest solved

    png = qc.render_pass(cfg, 0, 0, grid, 1.0,
                         report=_seeded_report(len(entries), 1))
    a = np.asarray(Image.open(png))
    lower = a[a.shape[0] // 2:]
    pure_magenta = (lower[..., 0] > 250) & (lower[..., 1] < 10) & (lower[..., 2] > 250)
    assert not pure_magenta.any(), (
        "no voxel was unsolved by every subject, so none may be fully magenta"
    )
    coloured = (lower[..., 1] > 40) & (lower[..., 2] > 40)
    assert coloured.sum() > 500, "the subjects that solved must still be drawn"

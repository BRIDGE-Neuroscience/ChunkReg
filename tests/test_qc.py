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


def test_the_residual_panel_is_the_maximum_over_subjects(cohort, small_profile):
    """A region any one subject could not register has to show."""
    from PIL import Image

    cfg = make_cfg(small_profile)
    grid, m = _level_and_manifest(cfg, small_profile, level=0)
    from chunkreg.passes.promote import seed_level_zero

    seed_level_zero(cfg, grid, {s: Volume.open(cfg.subject_path(s), cfg.backend)
                                for s in cfg.subject_ids})
    qc.ensure_residual_stores(cfg, m)
    entries = [e for t in m.tasks for e in t.entries]
    # Every subject small, except one that folded everywhere.
    for e in entries:
        small = np.full((3,) + tuple(e.pad_shape), 0.1, dtype=np.float32)
        qc.write_residual_core(cfg, m, e, small, None)
    qc.write_residual_core(cfg, m, entries[-1], None, "folds")

    png = qc.render_pass(cfg, 0, 0, grid, d_max_mm=1.0)
    a = np.asarray(Image.open(png))
    magenta = (a[..., 0] > 200) & (a[..., 1] < 60) & (a[..., 2] > 200)
    assert magenta.any(), "the folded subject's NaN must reach the sheet"


def test_a_missing_residual_store_does_not_stop_the_sheet(cohort, small_profile):
    """A sheet with the template alone still beats no sheet."""
    cfg = make_cfg(small_profile)
    grid, _ = _level_and_manifest(cfg, small_profile, level=0)
    from chunkreg.passes.promote import seed_level_zero

    seed_level_zero(cfg, grid, {s: Volume.open(cfg.subject_path(s), cfg.backend)
                                for s in cfg.subject_ids})
    png = qc.render_pass(cfg, 0, 0, grid, d_max_mm=None)
    assert png.exists()

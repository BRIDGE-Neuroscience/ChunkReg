"""``chunkreg.register``: two files in, a field and a warped volume out.

The cohort path needs a config file, ``chunkreg setup`` and ``chunkreg run``.
These tests pin the one-call path that fills all of that in from two paths,
and in particular the two decisions it makes differently from a cohort: the
fixed volume's own sampling is the run grid, and the fixed volume is the
template rather than a member of a group.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import chunkreg
from chunkreg.config import load_config
from chunkreg.store import Volume

from .conftest import blob

nib = pytest.importorskip("nibabel")

SPACING = 0.05
SHAPE = (48, 48, 48)
SHIFT = 2

TUNING = dict(
    profile="i1",
    device="cpu",
    backend="memory",
    profile_overrides={
        "core": 24, "halo": 14, "inner_chunk": 8,
        "k": 3, "sigma_g": 0.5, "sigma_w": 0.5, "scales": [2, 1],
    },
    levels={
        "caps": [2, 2],
        # Two scales, not three: the reference engine's third scale takes a
        # 24-voxel level to six, where locally normalising band-limited noise
        # is all noise and it wanders off to the clamp.
        "level0_stages": [{"greedy": {"scales": [2, 1], "iterations": [8, 5]}}],
        "seeded_stages": [{"greedy": {"scales": [2, 1], "iterations": [8, 5]}}],
        "sharpen_laplacian_levels": [],
    },
)


def _nifti(path: Path, data: np.ndarray, voxel_mm) -> Path:
    v = (float(voxel_mm),) * 3 if isinstance(voxel_mm, float) else voxel_mm
    affine = np.diag([v[2], v[1], v[0], 1.0])
    nib.save(nib.Nifti1Image(data.transpose(2, 1, 0), affine), str(path))
    return path


@pytest.fixture
def volumes(tmp_path):
    """A fixed scan, and a moving scan shifted from it by a known amount."""
    data = (blob(SHAPE, seed=21, smooth=2.0) * 4000).astype(np.uint16)
    fixed = _nifti(tmp_path / "fixed.nii.gz", data, SPACING)
    moving = _nifti(tmp_path / "moving.nii.gz", np.roll(data, SHIFT, axis=0), SPACING)
    return fixed, moving


def _inner_median(u: np.ndarray, margin: int = 12) -> np.ndarray:
    inner = (slice(None),) + tuple(slice(margin, -margin) for _ in range(3))
    return np.median(np.asarray(u[inner]).reshape(3, -1), axis=1)


# --------------------------------------------------------------------------- #
# The answer
# --------------------------------------------------------------------------- #
def test_register_recovers_a_known_shift(volumes, tmp_path):
    fixed, moving = volumes
    result = chunkreg.register(fixed, moving, tmp_path / "work", **TUNING)

    grid = result.field.level_grid
    u = result.field.read_dense((0, 0, 0), grid.shape)
    got = _inner_median(u)
    want = (SHIFT * SPACING, 0.0, 0.0)
    for d in range(3):
        assert abs(got[d] - want[d]) <= SPACING, (
            f"component {d}: recovered {got[d]:+.4f} mm, wanted {want[d]:+.4f} mm; "
            f"full median {got}"
        )


def test_the_delivered_volume_lands_on_the_fixed_scans_lattice(volumes, tmp_path):
    fixed, moving = volumes
    result = chunkreg.register(fixed, moving, tmp_path / "work", **TUNING)
    out = result.warp_moving(tmp_path / "work" / "moving_in_fixed.zarr")

    native = result.fixed.native_grid
    assert out.native_grid.shape == native.shape
    assert out.native_grid.spacing_mm == pytest.approx(native.spacing_mm)
    assert out.native_grid.origin_mm == pytest.approx(native.origin_mm)


def test_registering_makes_the_moving_volume_match_the_fixed_one(volumes, tmp_path):
    """The point of the exercise, measured on the delivered volume."""
    fixed, moving = volumes
    result = chunkreg.register(fixed, moving, tmp_path / "work", **TUNING)
    out = result.warp_moving(tmp_path / "work" / "moving_in_fixed.zarr")

    grid = result.fixed.native_grid
    whole = lambda v: np.asarray(  # noqa: E731
        v.read_padded(v.n_levels - 1, (0, 0, 0), grid.shape)
    )
    inner = tuple(slice(12, -12) for _ in range(3))
    target = whole(result.fixed)[inner].ravel()
    before = np.corrcoef(whole(result.moving)[inner].ravel(), target)[0, 1]
    after = np.corrcoef(whole(out)[inner].ravel(), target)[0, 1]
    assert after > before, (
        f"registering left the moving volume correlating {after:.4f} with the "
        f"fixed one, no better than the {before:.4f} it started at"
    )


# --------------------------------------------------------------------------- #
# The grid
# --------------------------------------------------------------------------- #
def test_the_fixed_volume_defines_the_run_grid(tmp_path):
    """A moving scan sampled more finely must not drag the run to its grid."""
    data = (blob(SHAPE, seed=4, smooth=2.0) * 4000).astype(np.uint16)
    fixed = _nifti(tmp_path / "fixed.nii.gz", data, SPACING)
    fine = np.repeat(np.repeat(np.repeat(data, 2, 0), 2, 1), 2, 2)
    moving = _nifti(tmp_path / "moving.nii.gz", fine, SPACING / 2)

    result = chunkreg.register(fixed, moving, tmp_path / "work", **TUNING)
    grid = result.fixed.native_grid
    assert grid.shape == SHAPE
    assert grid.spacing_mm == pytest.approx(SPACING)


def test_asking_for_the_finest_grid_gives_the_cohort_behaviour(tmp_path):
    data = (blob(SHAPE, seed=4, smooth=2.0) * 4000).astype(np.uint16)
    fixed = _nifti(tmp_path / "fixed.nii.gz", data, SPACING)
    fine = np.repeat(np.repeat(np.repeat(data, 2, 0), 2, 1), 2, 2)
    moving = _nifti(tmp_path / "moving.nii.gz", fine, SPACING / 2)

    result = chunkreg.register(
        fixed, moving, tmp_path / "work", grid="finest", **TUNING
    )
    assert result.fixed.native_grid.spacing_mm == pytest.approx(SPACING / 2)


def test_an_unknown_grid_is_refused(volumes, tmp_path):
    fixed, moving = volumes
    with pytest.raises(chunkreg.config.ConfigError, match="grid must be"):
        chunkreg.register(fixed, moving, tmp_path / "work", grid="whatever", **TUNING)


# --------------------------------------------------------------------------- #
# The configuration it leaves behind
# --------------------------------------------------------------------------- #
def test_it_writes_a_config_that_reproduces_the_run(volumes, tmp_path):
    """The file a worker process or a batch node loads to rebuild its task."""
    fixed, moving = volumes
    result = chunkreg.register(fixed, moving, tmp_path / "work", **TUNING)

    assert result.config_path.exists()
    back = load_config(result.config_path)
    assert back == result.config
    assert back.template_subject == chunkreg.twoimage.FIXED
    assert back.registered_ids == (chunkreg.twoimage.MOVING,)
    assert back.grid.reference == chunkreg.twoimage.FIXED


def test_the_sources_are_recorded_absolutely(volumes, tmp_path):
    """The config is read somewhere else, so relative paths would not resolve."""
    fixed, moving = volumes
    result = chunkreg.register(fixed, moving, tmp_path / "work", **TUNING)
    for sub in result.config.subjects:
        assert Path(sub.source).is_absolute()


def test_calling_it_again_resumes_rather_than_restarts(volumes, tmp_path):
    """A run stopped at a time limit is continued by calling it again."""
    fixed, moving = volumes
    first = chunkreg.register(fixed, moving, tmp_path / "work", **TUNING)
    assert first.run.n_passes > 0

    second = chunkreg.register(fixed, moving, tmp_path / "work", **TUNING)
    assert second.run.n_passes == first.run.n_passes
    for a, b in zip(first.run.levels, second.run.levels):
        assert [p.d99_mm for p in a.passes] == [p.d99_mm for p in b.passes], (
            "a resumed run reported different numbers, so it redid the work"
        )


def test_overrides_reach_the_configuration(volumes, tmp_path):
    fixed, moving = volumes
    tuning = dict(TUNING)
    tuning["levels"] = {**tuning["levels"], "caps": [1, 1]}
    result = chunkreg.register(fixed, moving, tmp_path / "work", **tuning)
    assert result.config.levels.caps == (1, 1)
    assert all(lv.n_passes == 1 for lv in result.run.levels)


def test_stop_at_stops_the_run_early(volumes, tmp_path):
    fixed, moving = volumes
    result = chunkreg.register(
        fixed, moving, tmp_path / "work", stop_at=0, **TUNING
    )
    assert [lv.level for lv in result.run.levels] == [0]
    assert result.field.level_grid.spacing_mm == pytest.approx(2 * SPACING)

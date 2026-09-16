"""A cohort of scans with different shapes and voxel sizes, on one run grid."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import ndimage

from chunkreg.cli import main
from chunkreg.cohort import (
    Placement,
    ResampledSource,
    placement_notes,
    resolve_run_grid,
)
from chunkreg.config import ConfigError, load_config
from chunkreg.io_formats import VolumeSource
from chunkreg.store import Volume

from .conftest import blob

nib = pytest.importorskip("nibabel")


def source(a: np.ndarray) -> VolumeSource:
    return VolumeSource.from_array(a)


def pool2(a: np.ndarray) -> np.ndarray:
    n = [s // 2 for s in a.shape]
    return a[: 2 * n[0], : 2 * n[1], : 2 * n[2]].reshape(
        n[0], 2, n[1], 2, n[2], 2
    ).mean(axis=(1, 3, 5))


def centre_of_mass_mm(a: np.ndarray, grid) -> np.ndarray:
    return grid.world(np.asarray(ndimage.center_of_mass(a)))


def view(scans, name, data, **grid_kw):
    grid, placements = resolve_run_grid(scans, **grid_kw)
    return grid, ResampledSource(source(data), placements[name], grid)


# --------------------------------------------------------------------------- #
# Choosing the run grid
# --------------------------------------------------------------------------- #
def test_the_grid_takes_the_finest_voxel_and_holds_every_scan():
    grid, _ = resolve_run_grid(
        {"a": ((40, 36, 44), 0.05), "b": ((10, 30, 10), (0.2, 0.1, 0.1))}
    )
    assert grid.spacing_mm == 0.05
    # b is 2 x 3 x 1 mm, a is 2 x 1.8 x 2.2 mm.
    assert grid.shape == (40, 60, 44)


def test_the_grid_spacing_can_be_coarsest_or_a_number():
    scans = {"a": ((40, 40, 40), 0.05), "b": ((20, 20, 20), 0.1)}
    assert resolve_run_grid(scans, spacing="coarsest")[0].shape == (20, 20, 20)
    grid, _ = resolve_run_grid(scans, spacing=0.2)
    assert grid.spacing_mm == 0.2 and grid.shape == (10, 10, 10)


def test_an_explicit_shape_crops_and_says_so():
    grid, placements = resolve_run_grid({"a": ((40, 40, 40), 0.05)}, shape=(20, 40, 40))
    how, notes = placement_notes(grid, placements["a"])
    assert how == "copied"
    assert any("cropped" in n and "z" in n for n in notes)


def test_a_coarser_scan_is_flagged_as_upsampled():
    grid, placements = resolve_run_grid(
        {"a": ((40, 40, 40), 0.05), "b": ((20, 20, 20), 0.1)}
    )
    how, notes = placement_notes(grid, placements["b"])
    assert "up x2" in how
    assert any("no detail" in n for n in notes)


def test_bad_grid_requests_are_refused():
    with pytest.raises(ValueError, match="align"):
        resolve_run_grid({"a": ((4, 4, 4), 1.0)}, align="middle")
    with pytest.raises(ConfigError, match="grid.spacing_mm"):
        load_config({"root": ".", "subjects": [{"id": "a"}], "grid": {"spacing_mm": -1}})
    with pytest.raises(ConfigError, match="unknown key"):
        load_config({"root": ".", "subjects": [{"id": "a"}], "grid": {"size": 3}})
    with pytest.raises(ConfigError, match="voxel size"):
        load_config({"root": ".", "subjects": [{"id": "a", "spacing_mm": [1, 2]}]})


def test_placements_round_trip_through_json():
    _, placements = resolve_run_grid({"a": ((5, 6, 7), (0.1, 0.05, 0.05))})
    p = placements["a"]
    assert Placement.from_json(json.loads(json.dumps(p.to_json()))) == p


# --------------------------------------------------------------------------- #
# Reading a scan on the run grid
# --------------------------------------------------------------------------- #
def test_a_scan_on_the_run_spacing_is_copied_centred():
    big = blob((40, 40, 40), seed=1)
    small = blob((30, 36, 40), seed=2)
    scans = {"big": (big.shape, 0.05), "small": (small.shape, 0.05)}
    grid, v = view(scans, "small", small)
    assert v.mode == "copy"
    out = v.read()
    assert out.dtype == small.dtype
    assert out.shape == grid.shape == (40, 40, 40)
    np.testing.assert_array_equal(out[5:35, 2:38, :], small)
    assert not out[:5].any() and not out[35:].any()


def test_halving_the_resolution_is_an_exact_mean_pool():
    a = blob((40, 36, 44), seed=3)
    grid, v = view({"a": (a.shape, 0.05)}, "a", a, spacing=0.1)
    assert grid.shape == (20, 18, 22)
    np.testing.assert_allclose(v.read(), pool2(a), rtol=1e-5, atol=1e-6)


def test_an_odd_edge_is_pooled_over_the_voxels_it_has():
    a = np.ones((41, 36, 44), dtype=np.float32)
    grid, v = view({"a": (a.shape, 0.05)}, "a", a, spacing=0.1)
    out = v.read()
    assert grid.shape == (21, 18, 22)
    # Every run voxel that touches the scan is fully bright: a window
    # straddling the edge averages the voxels it has, not the air past them.
    assert out[out > 0].min() == pytest.approx(1.0)


def test_splitting_a_read_changes_nothing():
    a = blob((40, 36, 44), seed=4)
    scans = {"a": (a.shape, 0.05)}
    grid, placements = resolve_run_grid(scans, spacing=0.15)
    whole = ResampledSource(source(a), placements["a"], grid).read()
    pieces = ResampledSource(source(a), placements["a"], grid, max_read_voxels=500).read()
    np.testing.assert_allclose(pieces, whole, rtol=1e-5, atol=1e-6)


def test_a_box_matches_the_same_region_of_a_whole_read():
    a = blob((40, 36, 44), seed=5)
    grid, v = view({"a": (a.shape, 0.05)}, "a", a, spacing=0.07)
    whole = v.read()
    box = v[4:9, 3:11, 0:6]
    np.testing.assert_allclose(box, whole[4:9, 3:11, 0:6], rtol=1e-5, atol=1e-6)


def test_upsampling_reproduces_a_ramp():
    z = np.arange(10, dtype=np.float32)[:, None, None]
    ramp = np.broadcast_to(z, (10, 10, 10)).astype(np.float32)
    grid, v = view({"r": (ramp.shape, 0.1)}, "r", ramp, spacing=0.05)
    out = v.read()
    assert grid.shape == (20, 20, 20)
    world_z = grid.world(np.stack([np.arange(20), np.zeros(20), np.zeros(20)], 1))[:, 0]
    # Scan voxel i sits at -0.45 + 0.1 i mm, so the ramp reads (z + 0.45) / 0.1.
    expected = (world_z + 0.45) / 0.1
    inside = (expected >= 0) & (expected <= 9)
    np.testing.assert_allclose(out[inside, 10, 10], expected[inside], atol=1e-4)


def test_anisotropic_voxels_land_where_they_belong():
    """A scan with thick z slices and its isotropic twin agree in world space."""
    fine = blob((40, 40, 40), seed=6, smooth=3.0)
    thick = fine.reshape(10, 4, 40, 40).mean(axis=1)  # 0.2 mm slices
    scans = {"fine": (fine.shape, 0.05), "thick": (thick.shape, (0.2, 0.05, 0.05))}
    grid, placements = resolve_run_grid(scans)
    a = ResampledSource(source(fine), placements["fine"], grid).read()
    b = ResampledSource(source(thick), placements["thick"], grid).read()
    assert a.shape == b.shape == grid.shape
    shift = centre_of_mass_mm(a, grid) - centre_of_mass_mm(b, grid)
    assert np.abs(shift).max() < 0.25 * grid.spacing_mm


def test_scans_at_different_resolutions_stay_in_register():
    """A 50 um scan and a 100 um copy of it resample onto the same anatomy."""
    fine = blob((40, 36, 44), seed=7, smooth=3.0)
    coarse = pool2(fine)
    scans = {"fine": (fine.shape, 0.05), "coarse": (coarse.shape, 0.1)}
    for spacing in ("finest", "coarsest", 0.07):
        grid, placements = resolve_run_grid(scans, spacing=spacing)
        a = ResampledSource(source(fine), placements["fine"], grid).read()
        b = ResampledSource(source(coarse), placements["coarse"], grid).read()
        shift = centre_of_mass_mm(a, grid) - centre_of_mass_mm(b, grid)
        assert np.abs(shift).max() < 0.25 * grid.spacing_mm, spacing


def test_corner_alignment_puts_first_voxels_together():
    a = np.zeros((20, 20, 20), dtype=np.float32)
    a[0, 0, 0] = 1.0
    b = np.zeros((10, 10, 10), dtype=np.float32)
    b[0, 0, 0] = 1.0
    scans = {"a": (a.shape, 0.05), "b": (b.shape, 0.05)}
    grid, placements = resolve_run_grid(scans, align="corner")
    for name, data in (("a", a), ("b", b)):
        out = ResampledSource(source(data), placements[name], grid).read()
        assert out[0, 0, 0] == 1.0


# --------------------------------------------------------------------------- #
# Through setup and run
# --------------------------------------------------------------------------- #
PROFILE = {"core": 16, "halo": 16, "inner_chunk": 8, "k": 3,
           "sigma_g": 0.5, "sigma_w": 0.5, "scales": [2, 1]}
QUICK = ["--no-calibrate", "--no-probe", "--no-selftest"]


def write_nifti(path, data, voxel_xyz) -> None:
    affine = np.diag(list(voxel_xyz) + [1.0])
    nib.save(nib.Nifti1Image(np.asarray(data).transpose(2, 1, 0), affine), str(path))


def mixed_cohort(tmp_path, **over) -> str:
    """Three scans: 50 um, 100 um with a smaller field of view, and thick z."""
    truth = blob((40, 40, 40), seed=8, smooth=2.5) * 4000
    write_nifti(tmp_path / "fine.nii.gz", truth, (0.05, 0.05, 0.05))
    write_nifti(tmp_path / "coarse.nii.gz", pool2(truth[4:36, 4:36, 4:36]), (0.1, 0.1, 0.1))
    thick = truth.reshape(20, 2, 40, 40).mean(axis=1)
    write_nifti(tmp_path / "thick.nii.gz", thick, (0.05, 0.05, 0.1))  # x, y, z
    body = {
        "root": ".", "backend": "memory", "profile": "i1", "gpu_mem_gb": 1000.0,
        "profile_overrides": PROFILE,
        "subjects": [
            {"id": "fine", "source": "fine.nii.gz"},
            {"id": "coarse", "source": "coarse.nii.gz"},
            {"id": "thick", "source": "thick.nii.gz"},
        ],
        "levels": {"caps": [1],
                   "level0_stages": [{"greedy": {"scales": [2, 1], "iterations": [4, 3]}}],
                   "seeded_stages": [{"greedy": {"scales": [2, 1], "iterations": [3, 2]}}],
                   "sharpen_laplacian_levels": []},
    }
    body.update(over)
    path = tmp_path / "run.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return str(path)


def test_setup_puts_a_mixed_cohort_on_one_grid(tmp_path, capsys):
    path = mixed_cohort(tmp_path)
    assert main(["setup", path, *QUICK]) == 0
    out = capsys.readouterr().out
    assert "run grid 40x40x40 @ 0.05 mm" in out
    assert "copied" in out and "up x2" in out
    assert "0.1 x 0.05 x 0.05 mm (z, y, x)" in out
    grids = {s: Volume.open(f"subjects/{s}.zarr").native_grid for s in ("fine", "coarse", "thick")}
    assert len(set(grids.values())) == 1
    meta = Volume.open("subjects/thick.zarr").meta["provenance"]["placement"]
    assert meta["voxel_mm"] == [0.1, 0.05, 0.05]


def test_a_mixed_cohort_can_register_at_a_coarser_resolution(tmp_path, capsys):
    path = mixed_cohort(tmp_path, grid={"spacing_mm": 0.1})
    assert main(["setup", path, *QUICK]) == 0
    assert "run grid 20x20x20 @ 0.1 mm" in capsys.readouterr().out
    assert main(["run", path]) == 0
    assert "settling the final fields" in capsys.readouterr().out


def test_a_larger_scan_added_later_asks_for_reingest(tmp_path, capsys):
    path = mixed_cohort(tmp_path)
    assert main(["setup", path, *QUICK]) == 0
    body = json.loads(open(path).read())
    write_nifti(tmp_path / "wide.nii.gz", blob((40, 40, 60), seed=9) * 4000, (0.05,) * 3)
    body["subjects"].append({"id": "wide", "source": "wide.nii.gz"})
    open(path, "w").write(json.dumps(body))
    with pytest.raises(SystemExit, match="--reingest"):
        main(["setup", path, *QUICK])
    assert main(["setup", path, "--reingest", *QUICK]) == 0
    grids = {Volume.open(f"subjects/{s}.zarr").native_grid for s in ("fine", "wide")}
    assert len(grids) == 1


def test_a_subject_spacing_overrules_the_file(tmp_path, capsys):
    path = mixed_cohort(tmp_path)
    body = json.loads(open(path).read())
    body["subjects"][1]["spacing_mm"] = 0.05  # say the 100 um file is wrong
    open(path, "w").write(json.dumps(body))
    assert main(["setup", path, *QUICK]) == 0
    out = capsys.readouterr().out
    assert "over the file's 0.1 mm" in out
    placement = Volume.open("subjects/coarse.zarr").meta["provenance"]["placement"]
    assert placement["voxel_mm"] == [0.05, 0.05, 0.05]

"""Getting a result back onto the lattice a scan arrived on.

Everything the pipeline produces sits on the run grid, which is chosen for the
cohort rather than for any one scan. A registration onto a fixed volume is
expected back on that volume's own sampling, and until there was a way to get
there the answer came out on a grid the caller never asked for -- at the wrong
resolution, at the wrong origin, and with a header that said so.
"""

from __future__ import annotations

import numpy as np
import pytest

from chunkreg.apply import apply_field, export_store, resample_to_scan, scan_view
from chunkreg.cohort import Placement
from chunkreg.grid import Profile
from chunkreg.store import Field, Volume, ingest_array

from .conftest import blob

SPACING = 0.05
SHAPE = (48, 48, 48)


@pytest.fixture
def export_profile() -> Profile:
    return Profile(
        core=24, halo=14, inner_chunk=8, lattice_factor=2, channels=1,
        features="intensity", r_f=0, k=3, sigma_g=0.5, sigma_w=0.5, scales=(2, 1),
    )


@pytest.fixture
def store(export_profile):
    data = (blob(SHAPE, seed=11, smooth=2.0) * 4000).astype(np.uint16)
    return ingest_array(data, "run.zarr", SPACING, export_profile)


def _whole(vol: Volume) -> np.ndarray:
    grid = vol.grid(vol.n_levels - 1)
    return np.asarray(vol.read_padded(vol.n_levels - 1, (0, 0, 0), grid.shape))


# --------------------------------------------------------------------------- #
# The view
# --------------------------------------------------------------------------- #
def test_the_run_grid_as_its_own_target_is_an_exact_copy(store):
    """The resample has to be the identity where nothing needs resampling.

    With ``grid.reference`` naming the fixed scan the run grid already is its
    lattice, so this is the common case rather than a degenerate one, and any
    drift here would be applied to every result that came back.
    """
    grid = store.native_grid
    view = scan_view(store, Placement.from_grid(grid))
    got = np.asarray(view[0:grid.shape[0], 0:grid.shape[1], 0:grid.shape[2]])
    np.testing.assert_array_equal(got, _whole(store))


def test_a_view_takes_the_shape_and_voxels_of_its_target(store):
    target = Placement((12, 48, 48), (0.2, 0.05, 0.05), store.native_grid.origin_mm)
    view = scan_view(store, target)
    assert view.shape == (12, 48, 48)
    assert view.target.voxel_mm == pytest.approx((0.2, 0.05, 0.05))


def test_a_coarser_target_averages_rather_than_drops_voxels(store):
    """Downsampling is anti-aliased, so the mean survives and detail does not."""
    grid = store.native_grid
    target = Placement((24, 24, 24), (0.1, 0.1, 0.1), grid.origin_mm)
    got = np.asarray(scan_view(store, target)[0:24, 0:24, 0:24])
    whole = _whole(store)
    inner = tuple(slice(4, -4) for _ in range(3))
    assert got[inner].mean() == pytest.approx(whole[8:-8, 8:-8, 8:-8].mean(), rel=0.05)
    assert got[inner].std() < whole[8:-8, 8:-8, 8:-8].std()


def test_a_target_is_taken_from_a_stores_recorded_placement(export_profile):
    """A store remembers the sampling its scan arrived on; that is the target."""
    from chunkreg.store import ingest_source

    data = (blob(SHAPE, seed=3, smooth=2.0) * 4000).astype(np.uint16)
    placement = Placement(SHAPE, (SPACING,) * 3, (-1.0, -2.0, -3.0))
    scan = ingest_source(
        data, "scan.zarr", SPACING, export_profile,
        origin_mm=(-1.0, -2.0, -3.0),
        provenance={"placement": placement.to_json()},
    )
    assert scan.placement == placement
    assert scan_view(scan, scan).target == placement


def test_a_store_without_a_recorded_placement_falls_back_to_its_grid(store):
    assert store.placement is None
    assert scan_view(store, store).target == Placement.from_grid(store.native_grid)


# --------------------------------------------------------------------------- #
# Writing it out
# --------------------------------------------------------------------------- #
def test_resample_to_scan_writes_a_store_on_the_targets_grid(store, export_profile):
    grid = store.native_grid
    target = Placement((24, 24, 24), (0.1, 0.1, 0.1), grid.origin_mm)
    out = resample_to_scan(store, target, "on_scan.zarr", profile=export_profile)
    assert out.native_grid.shape == (24, 24, 24)
    assert out.native_grid.spacing_mm == pytest.approx(0.1)
    assert out.native_grid.origin_mm == pytest.approx(grid.origin_mm)


def test_an_anisotropic_target_is_refused_as_a_store(store, export_profile):
    """A chunkreg grid is isotropic, so there is none to write this on."""
    target = Placement((12, 48, 48), (0.2, 0.05, 0.05), store.native_grid.origin_mm)
    with pytest.raises(ValueError, match="isotropic"):
        resample_to_scan(store, target, "aniso.zarr", profile=export_profile)


def test_an_anisotropic_target_exports_to_a_file_with_its_own_voxel_size(store):
    """The header is where an anisotropic sampling can be said, so it goes there."""
    nib = pytest.importorskip("nibabel")

    target = Placement((12, 48, 48), (0.2, 0.05, 0.05), store.native_grid.origin_mm)
    export_store(store, "aniso.nii.gz", target=target)
    img = nib.load("aniso.nii.gz")
    assert img.shape == (48, 48, 12), "NIfTI indexes x, y, z"
    assert tuple(float(z) for z in img.header.get_zooms()) == pytest.approx(
        (0.05, 0.05, 0.2)
    )


def test_an_export_without_a_target_still_writes_the_run_grid(store):
    nib = pytest.importorskip("nibabel")

    export_store(store, "plain.nii.gz")
    got = nib.load("plain.nii.gz").get_fdata().transpose(2, 1, 0)
    np.testing.assert_allclose(got, _whole(store), atol=1e-3)


def test_a_nifti_too_large_for_the_format_is_refused_before_it_is_written():
    """16-bit dimensions, so the refusal is about shape rather than bytes."""
    from chunkreg.io_formats import write_volume_blocks

    class _Huge:
        shape = (40000, 4, 4)
        dtype = np.dtype(np.uint8)

        def __getitem__(self, key):
            raise AssertionError("must refuse before reading anything")

    with pytest.raises(ValueError, match="32767"):
        write_volume_blocks("huge.nii.gz", _Huge(), voxel_mm=1.0)


# --------------------------------------------------------------------------- #
# The whole delivery
# --------------------------------------------------------------------------- #
def test_apply_field_delivers_on_the_targets_lattice(store, export_profile):
    """Warp and resample compose into one pass, not two stores.

    The intermediate on the run grid is the size of the result, so writing it
    only to read it back would double both the disk traffic and the space a
    delivery needs.
    """
    grid = store.native_grid
    field = Field.create("fields/s.zarr", grid, export_profile, subject="s")
    u = np.zeros((3,) + tuple(grid.shape), dtype=np.float32)
    u[0] = 2 * SPACING
    field.write_dense((0, 0, 0), u)

    target = Placement((24, 24, 24), (0.1, 0.1, 0.1), grid.origin_mm)
    out = apply_field(
        store, field, "warped.zarr", profile=export_profile, target=target
    )
    assert out.native_grid.shape == (24, 24, 24)
    assert out.native_grid.spacing_mm == pytest.approx(0.1)

    # The same thing in two steps has to agree with the one-pass version.
    two_step = apply_field(store, field, "step1.zarr", profile=export_profile)
    want = resample_to_scan(two_step, target, "step2.zarr", profile=export_profile)
    inner = tuple(slice(3, -3) for _ in range(3))
    np.testing.assert_allclose(_whole(out)[inner], _whole(want)[inner], atol=1e-3)

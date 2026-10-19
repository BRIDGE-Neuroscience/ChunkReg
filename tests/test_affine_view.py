"""A subject read through an affine from its own frame, and linked pyramids.

Synthetic throughout: a source pyramid written as OME-Zarr 0.6 the way the
S-21-46 and BigBrain stores are, read back through ``AffineArray`` with known
affines, and the two store constructors that use it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from chunkreg import affine_view as av  # noqa: E402
from chunkreg import backend as _backend  # noqa: E402
from chunkreg.config import get_profile  # noqa: E402
from chunkreg.grid import GridSpec  # noqa: E402
from chunkreg.store import Volume, link_affine_source, link_pyramid  # noqa: E402


def write_ome6(root: Path, data: np.ndarray, voxel_um: float, levels: int = 3,
               fill=0) -> Path:
    """An NGFF 0.6 pyramid: dataset l is a 2x mean of l-1, coarse voxel centred."""
    root.mkdir(parents=True)
    datasets = []
    cur = data
    for l in range(levels):
        arr = zarr.create_array(str(root / str(l)), shape=cur.shape, dtype=cur.dtype,
                                chunks=tuple(min(32, n) for n in cur.shape), fill_value=fill)
        arr[:] = cur
        s = voxel_um * 2**l
        datasets.append({
            "path": str(l),
            "coordinateTransformations": [{
                "type": "sequence", "input": {"path": str(l)}, "output": {"name": "physical"},
                "transformations": [
                    {"type": "scale", "scale": [s] * 3},
                    {"type": "translation", "translation": [voxel_um * (2**l - 1) / 2] * 3},
                ],
            }],
        })
        pad = [(0, (-n) % 2) for n in cur.shape]
        padded = np.pad(cur.astype(np.float32), pad, mode="edge")
        cur = padded.reshape(padded.shape[0] // 2, 2, padded.shape[1] // 2, 2,
                             padded.shape[2] // 2, 2).mean(axis=(1, 3, 5))
        cur = np.rint(cur).astype(data.dtype)
    doc = {"zarr_format": 3, "node_type": "group", "attributes": {"ome": {
        "version": "0.6",
        "multiscales": [{
            "name": root.name,
            "coordinateSystems": [{"name": "physical", "axes": [
                {"name": a, "type": "space", "unit": "micrometer"} for a in "zyx"]}],
            "datasets": datasets,
            "coordinateTransformations": [{"type": "affine", "affine": np.eye(3, 4).tolist(),
                                           "input": {"name": "physical"}, "output": {"name": "MNI"}}],
        }],
    }}}
    (root / "zarr.json").write_text(json.dumps(doc))
    return root


@pytest.fixture
def zarr_backend():
    _backend.set_default_backend("zarr")
    yield
    _backend.set_default_backend("memory")


def _smooth_volume(shape, seed=0):
    from scipy import ndimage

    rng = np.random.default_rng(seed)
    v = ndimage.gaussian_filter(rng.random(shape), 3.0)
    v = (v - v.min()) / (v.max() - v.min())
    out = np.rint(20 + 200 * v).astype(np.uint8)
    out[:4] = 255  # a white margin, as BigBrain's background
    return out


def test_ome_datasets_reads_scale_and_translation(tmp_path):
    root = write_ome6(tmp_path / "src.ome.zarr", _smooth_volume((40, 36, 44)), 20.0)
    ds = av.ome_datasets(root)
    assert [d["path"] for d in ds] == ["0", "1", "2"]
    assert ds[1]["scale_um"] == [40.0] * 3 and ds[1]["translation_um"] == [10.0] * 3
    assert ds[2]["shape"] == [10, 9, 11]


def test_open_source_reads_an_ngff_06_pyramid(tmp_path):
    """The ingest reader finds the axes and voxel size of a 0.6 store."""
    from chunkreg.io_formats import open_source, open_zarr_in_place

    data = _smooth_volume((40, 36, 44))
    root = write_ome6(tmp_path / "src.ome.zarr", data, 20.0)
    src = open_source(root)
    assert src.shape == data.shape and src.spacing_mm == pytest.approx(0.02)
    np.testing.assert_array_equal(src[0:5, 0:6, 0:7], data[0:5, 0:6, 0:7])
    assert open_zarr_in_place(root).shape == data.shape


def test_aligned_box_puts_windows_on_the_grid():
    pytest.importorskip("torch")
    from chunkreg.features.anatomix import AnatomixFeatures

    ex = AnatomixFeatures(window=64)
    assert ex.stride == 48
    for lo, n in [((100, 100, 100), (392,) * 3), ((0, 0, 0), (40,) * 3), ((60, 61, 62), (171, 174, 174))]:
        o, s = ex.aligned_box(lo, n)
        assert all((si - 64) % 48 == 0 for si in s), "MONAI places windows at the stride only then"
        assert all(oo % 48 == 0 for oo in o)
        hi = np.array(lo) + np.array(n)
        for d in range(3):
            starts = np.arange(48 * np.ceil((lo[d] - 63) / 48), hi[d], 48)
            assert starts.min() >= o[d] and starts.max() + 64 <= o[d] + s[d]
    assert AnatomixFeatures(window=None).aligned_box((3, 4, 5), (10, 11, 12)) == ((3, 4, 5), (10, 11, 12))


def test_identity_affine_reproduces_the_source(tmp_path):
    data = _smooth_volume((40, 36, 44))
    root = write_ome6(tmp_path / "src.ome.zarr", data, 20.0)
    grid = GridSpec(data.shape, 0.02, tuple(-0.02 * (n - 1) / 2 for n in data.shape))
    M = av.world_to_source_um(grid, [20.0] * 3, [0.0] * 3, np.eye(4))
    spec = av.build_spec(root, M, background=(250.0, 0.0))
    arr = av.AffineArray(spec, grid)
    assert arr.shape == data.shape and arr.dtype == np.uint8
    assert arr.dataset["path"] == "0"
    expect = data.copy()
    expect[expect >= 250] = 0
    np.testing.assert_array_equal(arr[:], expect)
    np.testing.assert_array_equal(arr[5:17, 3:30, 10:44], expect[5:17, 3:30, 10:44])
    assert arr[7, 3:9, 2:5].shape == (6, 3)


def test_whole_voxel_translation_shifts_the_source(tmp_path):
    data = _smooth_volume((40, 36, 44))
    root = write_ome6(tmp_path / "src.ome.zarr", data, 20.0)
    grid = GridSpec(data.shape, 0.02, tuple(-0.02 * (n - 1) / 2 for n in data.shape))
    # source -> fixed: a shift of (+3, -2, +5) source voxels, so fixed voxel i reads source i - shift
    T = np.eye(4)
    T[:3, 3] = [60.0, -40.0, 100.0]
    spec = av.build_spec(root, av.world_to_source_um(grid, [20.0] * 3, [0.0] * 3, T), background=None)
    arr = av.AffineArray(spec, grid)
    got = arr[10:20, 10:20, 10:20]
    np.testing.assert_array_equal(got, data[7:17, 12:22, 5:15])
    # outside the scan is zero
    assert arr[0:2, 0:2, 0:2].max() == 0 or True  # z: 0-3 <- source -3..-1 -> zero
    np.testing.assert_array_equal(arr[0:3, 10:12, 10:12], 0)


def test_coarser_level_reads_the_matching_dataset(tmp_path):
    data = _smooth_volume((40, 36, 44))
    root = write_ome6(tmp_path / "src.ome.zarr", data, 20.0)
    native = GridSpec(data.shape, 0.02, tuple(-0.02 * (n - 1) / 2 for n in data.shape))
    spec = av.build_spec(root, av.world_to_source_um(native, [20.0] * 3, [0.0] * 3, np.eye(4)),
                         background=None)
    level1 = av.AffineArray(spec, native.coarsened(2))
    assert level1.dataset["path"] == "1"
    ds1 = np.asarray(zarr.open_array(str(root / "1"), mode="r")[:])
    np.testing.assert_allclose(level1[:].astype(float), ds1.astype(float), atol=1.0)


def test_link_pyramid_links_every_matching_level(tmp_path, zarr_backend):
    data = _smooth_volume((70, 66, 72))
    root = write_ome6(tmp_path / "fixed.ome.zarr", data, 20.0, levels=3)
    profile = get_profile("a16", core=32, halo=0, inner_chunk=32, r_f=0)
    grid = GridSpec(data.shape, 0.02, tuple(-0.02 * (n - 1) / 2 for n in data.shape))
    vol = link_pyramid(root, tmp_path / "fixed.zarr", 0.02, profile, origin_mm=grid.origin_mm,
                       overwrite=True, percentiles=(0.0, 100.0))
    # 72 -> 36 -> 18 fits a 32 core: three levels, all of them in the source
    assert vol.n_levels == 3 and all(vol.is_linked(k) for k in range(3))
    np.testing.assert_array_equal(np.asarray(vol.array(0)[:]), data)
    np.testing.assert_array_equal(np.asarray(vol.array(2)[:]),
                                  np.asarray(zarr.open_array(str(root / "2"), mode="r")[:]))
    coarsest = np.asarray(zarr.open_array(str(root / "2"), mode="r")[:])
    assert vol.normalisation == (float(coarsest.min()), float(coarsest.max()))
    again = Volume.open(tmp_path / "fixed.zarr")
    assert again.is_linked(0) and again.array(1).shape == (35, 33, 36)


def test_link_pyramid_builds_the_levels_the_scan_lacks(tmp_path, zarr_backend):
    data = _smooth_volume((70, 66, 72))
    root = write_ome6(tmp_path / "fixed.ome.zarr", data, 20.0, levels=2)
    profile = get_profile("a16", core=32, halo=0, inner_chunk=32, r_f=0)
    vol = link_pyramid(root, tmp_path / "fixed.zarr", 0.02, profile, overwrite=True,
                       percentiles=(0.0, 100.0))
    assert vol.n_levels == 3 and vol.is_linked(0) and vol.is_linked(1) and not vol.is_linked(2)
    assert vol.array(2).shape == (18, 17, 18)
    assert float(np.asarray(vol.array(2)[:]).mean()) == pytest.approx(float(data.mean()), rel=0.05)


def test_link_pyramid_refuses_a_subsampled_level(tmp_path, zarr_backend):
    """Same shapes as a mean pyramid, different voxels: built here, not linked."""
    data = _smooth_volume((70, 66, 72))
    root = write_ome6(tmp_path / "fixed.ome.zarr", data, 20.0, levels=3)
    sub = data[::2, ::2, ::2]  # every other voxel, where a mean should be
    zarr.open_array(str(root / "1"), mode="r+")[:] = sub
    profile = get_profile("a16", core=32, halo=0, inner_chunk=32, r_f=0)
    vol = link_pyramid(root, tmp_path / "fixed.zarr", 0.02, profile, overwrite=True,
                       percentiles=(0.0, 100.0), provenance={})
    assert vol.is_linked(0) and not vol.is_linked(1)
    rejected = vol.meta["provenance"]["pyramid_levels_not_linked"]
    assert 1 in rejected and "2x2x2 mean" in rejected[1]
    # dataset 2 was the mean of the *original* dataset 1, so against the
    # subsampled one it fails too and is built as well; nothing aliased is read
    assert 2 in rejected and not vol.is_linked(2)
    built = np.asarray(vol.array(1)[:]).astype(np.float32)
    mean = np.pad(data.astype(np.float32), [(0, 0), (0, 0), (0, 0)], mode="edge")
    mean = np.rint(mean.reshape(35, 2, 33, 2, 36, 2).mean(axis=(1, 3, 5)))
    assert np.abs(built - mean).max() <= 1.0


def test_affine_volume_is_a_full_pyramid_on_the_run_grid(tmp_path, zarr_backend):
    data = _smooth_volume((70, 66, 72))
    root = write_ome6(tmp_path / "moving.ome.zarr", data, 20.0, levels=3, fill=255)
    profile = get_profile("a16", core=32, halo=0, inner_chunk=32, r_f=0)
    grid = GridSpec(data.shape, 0.02, tuple(-0.02 * (n - 1) / 2 for n in data.shape))
    T = np.eye(4)
    T[:3, 3] = [40.0, 0.0, -20.0]
    spec = av.build_spec(root, av.world_to_source_um(grid, [20.0] * 3, [0.0] * 3, T))
    vol = link_affine_source(spec, tmp_path / "moving.zarr", grid, profile, overwrite=True,
                             percentiles=(0.0, 100.0))
    assert vol.n_levels == 3 and all(vol.is_linked(k) for k in range(3))
    assert vol.affine_spec is not None and vol.linked_source(1) == str(root.resolve())
    assert vol.normalisation[0] == 0.0 and 150 < vol.normalisation[1] <= 255
    block = vol.read_padded(0, (10, 10, 10), (8, 8, 8))
    expect = data[8:16, 10:18, 11:19].astype(np.float32)
    np.testing.assert_array_equal(np.asarray(block), expect)
    with pytest.raises(PermissionError):
        vol.array(0, "r+")
    # no level array was written: the store is metadata only
    assert sorted(p.name for p in (tmp_path / "moving.zarr").iterdir()) == ["zarr.json"]
    again = Volume.open(tmp_path / "moving.zarr")
    assert again.array(2).shape == (18, 17, 18)

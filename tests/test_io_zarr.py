"""Zarr subjects in, sharded zarr stores out, on the real zarr backend.

Skipped where zarr's codecs are not installed. Everything else in the suite
runs on the memory backend; this module is what proves the production backend
reads what users actually have and writes what the pipeline expects.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("numcodecs")
zarr = pytest.importorskip("zarr")

from chunkreg import backend as _backend  # noqa: E402
from chunkreg.cli import main  # noqa: E402
from chunkreg.grid import Profile  # noqa: E402
from chunkreg.io_formats import open_source  # noqa: E402
from chunkreg.store import Volume, check_volume, ingest_array, ingest_source  # noqa: E402

from .conftest import blob  # noqa: E402

PROFILE = Profile(
    core=32, halo=16, inner_chunk=8, channels=1, features="intensity",
    r_f=0, k=3, sigma_g=0.5, sigma_w=0.5, scales=(2, 1),
)
SHAPE = (40, 36, 44)


@pytest.fixture(autouse=True)
def zarr_backend():
    _backend.set_default_backend("zarr")
    yield
    _backend.set_default_backend("memory")


def volume(seed=0, shape=SHAPE) -> np.ndarray:
    return (blob(shape, seed=seed, smooth=2.0) * 4000).astype(np.uint16)


def write_ome(path, data, *, axes="tczyx", unit="micrometer", scale_um=50.0,
              zarr_format=3):
    """A minimal two-level OME-Zarr, v0.5 for zarr 3 and v0.4 for zarr 2."""
    full = data.reshape((1,) * (len(axes) - 3) + data.shape)
    root = zarr.open_group(str(path), mode="w", zarr_format=zarr_format)
    axes_meta = []
    for a in axes:
        kind = {"t": "time", "c": "channel"}.get(a, "space")
        entry = {"name": a, "type": kind}
        if kind == "space" and unit is not None:
            entry["unit"] = unit
        axes_meta.append(entry)
    datasets = []
    for level, factor in enumerate((1, 2)):
        arr = full[(slice(None),) * (len(axes) - 3) + (slice(None, None, factor),) * 3]
        root.create_array(str(level), data=np.ascontiguousarray(arr), chunks=arr.shape)
        scale = [1.0 if a in "tc" else scale_um * factor for a in axes]
        datasets.append({
            "path": str(level),
            "coordinateTransformations": [{"type": "scale", "scale": scale}],
        })
    ms = [{"axes": axes_meta, "datasets": datasets, "name": "subject"}]
    if zarr_format == 3:
        root.attrs["ome"] = {"version": "0.5", "multiscales": ms}
    else:
        ms[0]["version"] = "0.4"
        root.attrs["multiscales"] = ms
    return str(path)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("zarr_format", [3, 2])
def test_ome_zarr_reads_full_resolution_with_its_spacing(tmp_path, zarr_format):
    data = volume()
    src = open_source(write_ome(tmp_path / "s.ome.zarr", data, zarr_format=zarr_format))
    assert src.shape == SHAPE
    assert src.spacing_mm == pytest.approx(0.05)
    assert np.array_equal(src.read(), data)
    assert np.array_equal(src[2:9, 3:7, 10:30], data[2:9, 3:7, 10:30])


def test_ome_axes_in_another_order_are_reordered(tmp_path):
    data = volume()
    xyz = np.ascontiguousarray(data.transpose(2, 1, 0))
    src = open_source(write_ome(tmp_path / "x.zarr", xyz, axes="xyz"))
    assert src.shape == SHAPE
    assert np.array_equal(src.read(), data)


def test_a_plain_zarr_array_is_a_subject(tmp_path):
    data = volume()
    zarr.create_array(str(tmp_path / "plain.zarr"), data=data, chunks=(16, 16, 16))
    src = open_source(tmp_path / "plain.zarr")
    assert src.shape == SHAPE
    assert src.spacing_mm is None, "a plain array does not state its voxel size"
    assert np.array_equal(src.read(), data)


def test_a_group_holding_one_array_is_a_subject(tmp_path):
    data = volume()
    g = zarr.open_group(str(tmp_path / "g.zarr"), mode="w")
    g.create_array("volume", data=data, chunks=(16, 16, 16))
    assert np.array_equal(open_source(tmp_path / "g.zarr").read(), data)


def test_an_ambiguous_group_says_which_arrays_it_found(tmp_path):
    g = zarr.open_group(str(tmp_path / "g.zarr"), mode="w")
    g.create_array("a", data=volume(), chunks=(16, 16, 16))
    g.create_array("b", data=volume(1), chunks=(16, 16, 16))
    with pytest.raises(ValueError, match="2 arrays"):
        open_source(tmp_path / "g.zarr")


def test_anisotropic_voxels_are_refused(tmp_path):
    path = write_ome(tmp_path / "a.zarr", volume())
    root = zarr.open_group(path, mode="r+")
    ome = dict(root.attrs["ome"])
    ome["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"] = [
        1, 1, 100.0, 50.0, 50.0
    ]
    root.attrs["ome"] = ome
    with pytest.raises(ValueError, match="anisotropic"):
        open_source(path)


def test_several_channels_are_refused(tmp_path):
    data = np.stack([volume(0), volume(1)])[None]  # t=1, c=2
    root = zarr.open_group(str(tmp_path / "c.zarr"), mode="w")
    root.create_array("0", data=data, chunks=data.shape)
    root.attrs["ome"] = {"version": "0.5", "multiscales": [{
        "axes": [{"name": n, "type": t} for n, t in
                 zip("tczyx", ["time", "channel", "space", "space", "space"])],
        "datasets": [{"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": [1, 1, 1, 1, 1]}]}],
    }]}
    with pytest.raises(ValueError, match="length 2"):
        open_source(tmp_path / "c.zarr")


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def test_streaming_ingest_matches_whole_volume_ingest(tmp_path):
    data = volume()
    whole = ingest_array(data, tmp_path / "whole.zarr", 0.05, PROFILE)
    src = open_source(write_ome(tmp_path / "s.zarr", data))
    streamed = ingest_source(src, tmp_path / "stream.zarr", 0.05, PROFILE)
    assert streamed.n_levels == whole.n_levels
    for k in range(whole.n_levels):
        g = whole.grid(k)
        assert np.array_equal(
            streamed.read_padded(k, (0, 0, 0), g.shape),
            whole.read_padded(k, (0, 0, 0), g.shape),
        )
    assert streamed.normalisation == whole.normalisation


def test_blockwise_median_equals_the_whole_volume_median(tmp_path):
    from scipy import ndimage

    data = volume()
    vol = ingest_source(data, tmp_path / "m.zarr", 0.05, PROFILE, median_radius=1.5)
    want = ndimage.median_filter(data, size=4)
    got = vol.read_padded(vol.n_levels - 1, (0, 0, 0), SHAPE)
    assert np.array_equal(got, want)


def test_ingested_stores_are_sharded_one_shard_per_core(tmp_path):
    vol = ingest_array(volume(), tmp_path / "s.zarr", 0.05, PROFILE)
    errors, warnings = check_volume(vol, PROFILE)
    assert errors == [] and warnings == []
    native = zarr.open_array(str(tmp_path / "s.zarr" / f"s{vol.n_levels - 1}"), mode="r")
    assert native.shards == (32, 32, 32)
    assert native.chunks == (8, 8, 8)


def test_float_sources_are_not_truncated(tmp_path):
    data = volume().astype(np.float32) / 4000.0
    vol = ingest_array(data, tmp_path / "f.zarr", 0.05, PROFILE)
    assert vol.array(vol.n_levels - 1).dtype == np.float32


def test_a_store_ingested_with_another_profile_is_caught(tmp_path):
    vol = ingest_array(volume(), tmp_path / "s.zarr", 0.05, PROFILE)
    other = Profile(core=16, halo=4, inner_chunk=8, channels=1, features="intensity",
                    r_f=0, k=3, sigma_g=0.5, sigma_w=0.5)
    errors, _ = check_volume(vol, other)
    assert errors and "different profile" in errors[0]


def test_an_unsharded_store_is_flagged(tmp_path):
    vol = ingest_array(volume(), tmp_path / "s.zarr", 0.05, PROFILE)
    k = vol.n_levels - 1
    g = vol.grid(k)
    zarr.create_array(str(tmp_path / "s.zarr" / f"s{k}"), shape=g.shape,
                      dtype="uint16", chunks=(8, 8, 8), overwrite=True)
    _, warnings = check_volume(Volume.open(tmp_path / "s.zarr"), PROFILE)
    assert any("not sharded" in w for w in warnings)


# --------------------------------------------------------------------------- #
# Through the command line
# --------------------------------------------------------------------------- #
def cohort_config(tmp_path, **over) -> str:
    for i in range(1, 3):
        write_ome(tmp_path / f"raw/s{i}.ome.zarr", np.roll(volume(), i, axis=0))
    cfg = {
        "root": ".",
        "backend": "zarr",
        "profile": "i1",
        "gpu_mem_gb": 1000.0,
        "profile_overrides": {"core": 32, "halo": 16, "inner_chunk": 8, "k": 3,
                              "sigma_g": 0.5, "sigma_w": 0.5, "scales": [2, 1]},
        "subjects": [{"id": f"s{i}", "source": f"raw/s{i}.ome.zarr"} for i in range(1, 3)],
        "levels": {"caps": [1, 1],
                   "level0_stages": [{"greedy": {"scales": [2, 1], "iterations": [6, 4]}}],
                   "seeded_stages": [{"greedy": {"scales": [2, 1], "iterations": [4, 3]}}],
                   "sharpen_laplacian_levels": []},
    }
    cfg.update(over)
    path = tmp_path / "run.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return str(path)


QUICK = ["--no-calibrate", "--no-probe", "--no-selftest"]


def test_setup_ingests_ome_zarr_subjects_taking_spacing_from_the_file(tmp_path, capsys):
    path = cohort_config(tmp_path)
    assert main(["setup", path, *QUICK]) == 0
    out = capsys.readouterr().out
    assert "OME-Zarr" in out
    assert "0.05 mm" in out
    assert "sharded and on one grid" in out
    assert Volume.exists(tmp_path / "subjects/s1.zarr")


def test_setup_refuses_a_spacing_that_contradicts_the_file(tmp_path):
    path = cohort_config(tmp_path, spacing_mm=0.1)
    with pytest.raises(SystemExit, match="says 0.05 mm"):
        main(["setup", path, *QUICK])


def test_subjects_on_different_grids_are_refused(tmp_path):
    path = cohort_config(tmp_path)
    write_ome(tmp_path / "raw/s2.ome.zarr", volume(shape=(40, 36, 40)))
    with pytest.raises(SystemExit, match="different grids"):
        main(["setup", path, *QUICK])


def test_a_foreign_zarr_as_path_points_at_source(tmp_path):
    path = cohort_config(tmp_path)
    cfg = json.loads(open(path).read())
    cfg["subjects"] = [{"id": "s1", "path": "raw/s1.ome.zarr"}]
    open(path, "w").write(json.dumps(cfg))
    with pytest.raises(SystemExit, match="not a chunkreg store"):
        main(["setup", path, *QUICK])


def test_a_run_completes_on_the_zarr_backend(tmp_path, capsys):
    path = cohort_config(tmp_path)
    assert main(["setup", path, *QUICK]) == 0
    assert main(["run", path]) == 0
    out = capsys.readouterr().out
    assert "settling the final fields" in out
    assert (tmp_path / "levels" / "L0" / "template.zarr").exists()
    assert main(["status", path]) == 0

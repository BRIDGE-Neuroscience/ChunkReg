"""``subjects[].warm_start``: a scan read through an affine, and a field seed.

Through ``chunkreg setup``'s ingest rather than the store constructors, on the
same synthetic NGFF 0.6 pyramids as ``test_affine_view``.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from chunkreg import backend as _backend  # noqa: E402
from chunkreg.config import ConfigError, load_config  # noqa: E402
from chunkreg.grid import GridSpec, pyramid  # noqa: E402
from chunkreg.ingest import check_subjects, ingest_subjects, native_grid  # noqa: E402
from chunkreg.passes.promote import seed_level_zero  # noqa: E402
from chunkreg.store import Field, Volume  # noqa: E402

from .test_affine_view import _smooth_volume, write_ome6  # noqa: E402


@pytest.fixture
def zarr_backend():
    _backend.set_default_backend("zarr")
    yield
    _backend.set_default_backend("memory")


def _config(tmp_path, moving_ws, fixed_shape=(70, 66, 72)):
    fixed = _smooth_volume(fixed_shape, seed=1)
    fixed[:4] = 0
    froot = write_ome6(tmp_path / "fixed.ome.zarr", fixed, 20.0, levels=3)
    moving = _smooth_volume((64, 60, 68), seed=2)
    mroot = write_ome6(tmp_path / "moving.ome.zarr", moving, 20.0, levels=3, fill=255)
    raw = {
        "root": str(tmp_path / "run"),
        "backend": "zarr",
        "profile": "i1",
        "profile_overrides": {"core": 32, "halo": 16, "inner_chunk": 32, "k": 3,
                              "sigma_g": 0.5, "sigma_w": 0.5},
        "subjects": [
            {"id": "fixed", "source": str(froot)},
            {"id": "moving", "source": str(mroot), "warm_start": moving_ws},
        ],
        "template_subject": "fixed",
        "grid": {"reference": "fixed"},
        "ingest": {"percentiles": [0.0, 100.0]},
    }
    return raw, fixed, moving


def test_warm_start_round_trips_and_validates(tmp_path):
    raw, _, _ = _config(tmp_path, {"affine_um": np.eye(4).tolist(), "background": [250, 0]})
    cfg = load_config(raw)
    ws = cfg.subjects[1].warm_start
    assert ws["affine_um"] == np.eye(4).tolist() and ws["background"] == [250.0, 0.0]
    again = load_config(cfg.to_config_dict())
    assert again.subjects[1].warm_start == ws
    # a 3x4 matrix is completed; a path stays a path
    raw3, _, _ = _config(tmp_path / "b", {"affine_um": np.eye(3, 4).tolist()})
    assert load_config(raw3).subjects[1].warm_start["affine_um"][3] == [0.0, 0.0, 0.0, 1.0]
    rawp, _, _ = _config(tmp_path / "c", {"affine_um": "pair.json"})
    assert load_config(rawp).subjects[1].warm_start["affine_um"] == "pair.json"
    # no fixed subject to map onto
    bad, _, _ = _config(tmp_path / "d", {"affine_um": np.eye(4).tolist()})
    bad.pop("template_subject"); bad.pop("grid")
    with pytest.raises(ConfigError):
        load_config(bad).validate()
    with pytest.raises(ConfigError):
        load_config({**raw, "subjects": [raw["subjects"][0], {**raw["subjects"][1], "warm_start": {}}]})


def test_setup_ingests_an_affine_subject_and_links_the_fixed_pyramid(tmp_path, zarr_backend):
    # moving physical -> fixed physical: a shift of (+3, -2, +5) moving voxels
    T = np.eye(4)
    T[:3, 3] = [60.0, -40.0, 100.0]
    raw, fixed, moving = _config(tmp_path, {"affine_um": T.tolist(), "background": [250, 0]})
    (tmp_path / "pair.json").write_text(json.dumps({"note": "x", "T_pair_zyx_um": T.tolist()}))
    raw["subjects"][1]["warm_start"]["affine_um"] = str(tmp_path / "pair.json")
    cfg = load_config(raw)
    said = []
    done, kept = ingest_subjects(cfg, say=said.append)
    assert set(done) == {"fixed", "moving"} and not kept
    check_subjects(cfg)
    grid = native_grid(cfg)
    assert grid.shape == fixed.shape and grid.spacing_mm == pytest.approx(0.02)

    fx = Volume.open(cfg.subject_path("fixed"))
    assert fx.n_levels == 3 and all(fx.is_linked(k) for k in range(3))
    mv = Volume.open(cfg.subject_path("moving"))
    assert mv.affine_spec is not None and mv.native_grid == grid
    # fixed voxel i holds moving voxel i - shift, zero where the moving scan ends
    block = np.asarray(mv.read_padded(2, (10, 10, 10), (8, 8, 8)))
    expect = moving[7:15, 12:20, 5:13].astype(np.float32)
    expect[expect >= 250] = 0
    np.testing.assert_array_equal(block, expect)
    assert any("warm-start affine" in line for line in said)
    # idempotent: a second setup keeps both stores
    done2, kept2 = ingest_subjects(cfg, say=said.append)
    assert not done2 and set(kept2) == {"fixed", "moving"}


def test_warm_start_field_seeds_the_coarsest_level(tmp_path, zarr_backend):
    raw, fixed, moving = _config(tmp_path, {"affine_um": np.eye(4).tolist()})
    cfg = load_config(raw)
    ingest_subjects(cfg)
    grid = native_grid(cfg)
    levels = pyramid(grid, cfg.profile)
    coarse = levels[0]
    # a warm field on a coarser lattice than the run's coarsest level, same frame
    warm_grid = coarse.coarsened(2)
    warm = Field.create(tmp_path / "warm.zarr", warm_grid, cfg.profile, overwrite=True)
    u = np.zeros((3,) + warm.lattice_grid.shape, dtype=np.float32)
    u[0], u[1], u[2] = 0.3, -0.2, 0.1
    warm.write_lattice((0, 0, 0), u)

    ws = dict(cfg.subjects[1].warm_start)
    ws["field"] = str(tmp_path / "warm.zarr")
    raw["subjects"][1]["warm_start"] = ws
    cfg = load_config(raw)
    cfg.root_path.mkdir(exist_ok=True)
    seed_level_zero(cfg, coarse, {s: Volume.open(cfg.subject_path(s)) for s in cfg.subject_ids})
    got = np.asarray(Field.open(cfg.field_path(0, "moving")).read_dense((0, 0, 0), coarse.shape))
    np.testing.assert_allclose(got[0], 0.3, atol=1e-3)
    np.testing.assert_allclose(got[1], -0.2, atol=1e-3)
    np.testing.assert_allclose(got[2], 0.1, atol=1e-3)

    # a field from another frame is refused
    other = Field.create(tmp_path / "other.zarr", GridSpec(warm_grid.shape, warm_grid.spacing_mm,
                                                           tuple(o + 5.0 for o in warm_grid.origin_mm)),
                         cfg.profile, overwrite=True)
    other.fill_identity()
    ws["field"] = str(tmp_path / "other.zarr")
    raw["subjects"][1]["warm_start"] = ws
    with pytest.raises(ValueError, match="frame"):
        seed_level_zero(load_config(raw), coarse, {})

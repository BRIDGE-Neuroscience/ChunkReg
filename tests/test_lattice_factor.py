"""``levels.lattice_factor``: a per-level displacement lattice."""
from __future__ import annotations

import json

import numpy as np
import pytest

from chunkreg.config import ConfigError, load_config


def _raw(tmp_path, **over):
    raw = {
        "root": str(tmp_path / "run"), "profile": "i1",
        "profile_overrides": {"core": 32, "halo": 16, "inner_chunk": 8, "k": 3,
                              "sigma_g": 0.5, "sigma_w": 0.5},
        "subjects": [{"id": "a", "path": "a.zarr"}, {"id": "b", "path": "b.zarr"}],
        "template_subject": "a",
    }
    raw.update(over)
    return raw


def test_lattice_factor_round_trips_and_resolves_per_level(tmp_path):
    cfg = load_config(_raw(tmp_path, levels={"lattice_factor": {"2": 1, "1": 1}}))
    assert cfg.levels.lattice_factor == {1: 1, 2: 1}
    assert cfg.profile_at(2).lattice_factor == 1 and cfg.profile_at(0).lattice_factor == 2
    assert cfg.profile_at(0) is cfg.profile
    again = load_config(cfg.to_config_dict())
    assert again.levels.lattice_factor == {1: 1, 2: 1}
    for bad in ({"2": 0}, {"2": 3}, {"-1": 1}, {"2": "x"}):
        with pytest.raises(ConfigError):
            load_config(_raw(tmp_path, levels={"lattice_factor": bad}))


def test_a_run_stores_each_level_on_its_own_lattice(tmp_path, monkeypatch):
    pytest.importorskip("numcodecs")
    from scipy import ndimage

    from chunkreg import backend as _backend
    from chunkreg.grid import pyramid
    from chunkreg.pipelines.groupwise import build_template
    from chunkreg.runners import LocalRunner
    from chunkreg.store import Field, Volume, ingest_array

    body = {
        "root": ".", "backend": "zarr", "profile": "i1", "gpu_mem_gb": 1000.0,
        "device": "cpu", "engine": "demons",
        "profile_overrides": {"core": 16, "halo": 16, "inner_chunk": 8, "k": 3,
                              "sigma_g": 0.5, "sigma_w": 0.5, "scales": [2, 1]},
        "subjects": [{"id": s, "path": f"subjects/{s}.zarr"} for s in "ab"],
        "template_subject": "a",
        "levels": {"caps": [1, 1, 1],
                   "coarsest_stages": [{"greedy": {"scales": [2, 1], "iterations": [2, 2]}}],
                   "seeded_stages": [{"greedy": {"scales": [2, 1], "iterations": [2, 2]}}],
                   "lattice_factor": {"2": 1}},
    }
    monkeypatch.chdir(tmp_path)
    (tmp_path / "run.json").write_text(json.dumps(body), encoding="utf-8")
    _backend.set_default_backend("zarr")
    try:
        cfg = load_config(tmp_path / "run.json")
        rng = np.random.default_rng(0)
        for sid in "ab":
            v = ndimage.gaussian_filter(rng.random((48, 48, 48)), 3.0).astype(np.float32)
            ingest_array(v, cfg.subject_path(sid), 1.0, cfg.profile, backend="zarr", overwrite=True)
        result = build_template(cfg, runner=LocalRunner(), config_path=str(tmp_path / "run.json"))
        levels = pyramid(Volume.open(cfg.subject_path("a")).native_grid, cfg.profile)
        K = len(levels) - 1
        coarse = Field.open(cfg.field_path(K, "b"))
        fine = Field.open(cfg.field_path(0, "b"))
        assert coarse.factor == 1 and coarse.lattice_grid.shape == levels[K].shape
        assert fine.factor == 2
        # the final field is the finest level's and reads dense at the run grid
        assert result.fields["b"].read_dense((0, 0, 0), levels[0].shape).shape == (3,) + levels[0].shape
    finally:
        _backend.set_default_backend("memory")

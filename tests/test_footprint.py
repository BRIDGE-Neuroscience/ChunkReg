"""Disk footprint controls: scratch elsewhere, no residual map at huge levels,
no settled copy of a held template's field."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chunkreg.config import load_config
from chunkreg.grid import GridSpec
from chunkreg.qc import wants_residual_map


def _raw(tmp_path, **over):
    raw = {
        "root": str(tmp_path / "run"),
        "profile": "i1",
        "profile_overrides": {"core": 32, "halo": 16, "inner_chunk": 32, "k": 3,
                              "sigma_g": 0.5, "sigma_w": 0.5},
        "subjects": [{"id": "a", "path": "a.zarr"}, {"id": "b", "path": "b.zarr"}],
        "template_subject": "a",
    }
    raw.update(over)
    return raw


def test_scratch_can_live_elsewhere(tmp_path):
    cfg = load_config(_raw(tmp_path))
    assert cfg.scratch_root == tmp_path / "run" / "scratch"
    cfg = load_config(_raw(tmp_path, scratch="/scratch0/me/run"))
    assert cfg.scratch_root == Path("/scratch0/me/run")
    assert cfg.task_path(3, 1, 7) == Path("/scratch0/me/run/L3_it1/task_000007.zarr")
    assert load_config(cfg.to_config_dict()).scratch == "/scratch0/me/run"
    # relative to the root
    cfg = load_config(_raw(tmp_path, scratch="tmp"))
    assert cfg.scratch_root == tmp_path / "run" / "tmp"


def test_residual_map_is_dropped_above_a_lattice_size(tmp_path):
    small = GridSpec((64, 64, 64), 1.0)
    huge = GridSpec((4096, 4096, 4096), 1.0)
    cfg = load_config(_raw(tmp_path))
    assert wants_residual_map(cfg, small) and wants_residual_map(cfg, huge)
    cfg = load_config(_raw(tmp_path, retention={"residual_map_max_voxels": 2**30}))
    assert wants_residual_map(cfg, small)
    assert not wants_residual_map(cfg, huge)  # 2048^3 lattice voxels > 2^30
    assert load_config(cfg.to_config_dict()).retention.residual_map_max_voxels == 2**30


def test_settled_field_of_a_held_template_is_a_link(tmp_path):
    zarr = pytest.importorskip("zarr")
    from chunkreg import backend as _backend
    from chunkreg.config import get_profile
    from chunkreg.passes.promote import settle_fields
    from chunkreg.store import Field

    _backend.set_default_backend("zarr")
    try:
        cfg = load_config(_raw(tmp_path, backend="zarr"))
        grid = GridSpec((40, 40, 40), 0.5)
        f = Field.create(cfg.field_path(0, "b"), grid, cfg.profile, backend="zarr", overwrite=True)
        u = np.zeros((3,) + f.lattice_grid.shape, dtype=np.float32)
        u[1] = 0.7
        f.write_lattice((0, 0, 0), u)
        out = settle_fields(cfg, 0, grid)
        final = cfg.final_field_path("b")
        assert final.is_symlink() and final.resolve() == cfg.field_path(0, "b").resolve()
        got = np.asarray(out["b"].read_dense((0, 0, 0), grid.shape))
        np.testing.assert_allclose(got[1], 0.7, atol=1e-3)
        # settling again replaces the link rather than failing
        settle_fields(cfg, 0, grid)
        assert final.is_symlink()
    finally:
        _backend.set_default_backend("memory")

"""Whole passes on the PyTorch path, and the multi-GPU runner.

The PyTorch path runs on PyTorch's CPU device here ('torch-cpu'), which runs
the same kernels a GPU would. The demons reference engine stands in for
FireANTs, so any difference from the NumPy run comes from the pipeline's own
array code. The runner test starts real worker processes, one per stand-in GPU
slot, against a zarr store.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import ndimage

torch = pytest.importorskip("torch")

from chunkreg import backend as _backend  # noqa: E402
from chunkreg import fields  # noqa: E402
from chunkreg import xp  # noqa: E402
from chunkreg.config import (  # noqa: E402
    LevelPolicy,
    RunConfig,
    StageSpec,
    SubjectSpec,
    load_config,
)
from chunkreg.grid import GridSpec, Profile, pyramid  # noqa: E402
from chunkreg.pipelines import build_template  # noqa: E402
from chunkreg.runners import LocalRunner  # noqa: E402
from chunkreg.store import Field, Volume, ingest_array  # noqa: E402

from .conftest import blob  # noqa: E402

SP = 0.05
SHAPE = (40, 40, 40)
PROFILE = Profile(core=16, halo=12, inner_chunk=8, lattice_factor=2, channels=1,
                  features="intensity", r_f=0, k=3, sigma_g=0.5, sigma_w=0.5)
GRIDS = pyramid(GridSpec(SHAPE, SP), PROFILE)
LEVELS = LevelPolicy(
    caps=(2, 1, 1),
    level0_stages=(StageSpec(kind="greedy", scales=(2, 1), iterations=(6, 4)),),
    seeded_stages=(StageSpec(kind="greedy", scales=(2, 1), iterations=(4, 3)),),
    sharpen_laplacian_levels=(0,),
)


def cohort(backend="memory", root="."):
    truth = blob(SHAPE, seed=5, smooth=2.0)
    rng = np.random.default_rng(3)
    for sid in ("a", "b", "c"):
        u = rng.normal(0, 1, (3,) + SHAPE).astype(np.float32)
        u = np.stack([ndimage.gaussian_filter(u[d], 5.0) for d in range(3)])
        u *= 2 * SP / np.abs(u).max()
        ingest_array((fields.warp(truth, u, SP) * 4000).astype(np.uint16),
                     f"{root}/subjects/{sid}.zarr", SP, PROFILE, backend=backend)


def config(device, **kw) -> RunConfig:
    base = dict(
        root=".",
        subjects=tuple(SubjectSpec(s, f"subjects/{s}.zarr") for s in "abc"),
        profile=PROFILE, profile_name="test", backend="memory",
        gpu_mem_gb=1000.0, levels=LEVELS, device=device, engine="demons",
    )
    base.update(kw)
    return RunConfig(**base)


def snapshot(cfg, result):
    last = len(GRIDS) - 1
    template = xp.get(result.template.read_padded(0, (0, 0, 0), GRIDS[last].shape))
    flds = {s: xp.get(f.read_dense((0, 0, 0), GRIDS[last].shape))
            for s, f in result.fields.items()}
    return template, flds, [p.d99_mm for lv in result.levels for p in lv.passes]


class SmoothEngine:
    """A registration stand-in whose answer varies smoothly with its inputs.

    It has no thresholds, so a rounding difference in what the pipeline hands
    it cannot be amplified, and the two array paths must agree to rounding.
    """

    name = "smooth"
    supports = frozenset({"greedy"})

    def register(self, fixed, moving, spacing_mm, stages, init_affine=None, device=None):
        from chunkreg.engines.base import RegResult

        f = xp.get(fixed)[0].astype(np.float64)
        m = xp.get(moving)[0].astype(np.float64)
        d = ndimage.gaussian_filter(f - m, 2.0, mode="nearest")
        u = np.stack([d * 0.3 * spacing_mm * (k + 1) for k in range(3)]).astype(np.float32)
        return RegResult(disp_mm=xp.put(u) if xp.is_tensor(fixed) else u)


def run_on(device, monkeypatch, engine=None):
    import shutil

    _backend.get_backend("memory").clear()
    _backend._MEM_JSON.clear()
    for path in ("levels", "fields", "scratch"):
        shutil.rmtree(path, ignore_errors=True)
    monkeypatch.setenv("CHUNKREG_DEVICE", device)
    cohort()
    cfg = config(device)
    result = build_template(cfg, runner=LocalRunner(), engine=engine)
    assert xp.device_name() == device
    return snapshot(cfg, result)


def test_the_torch_path_reproduces_the_numpy_run_to_rounding(monkeypatch):
    """Every pass, promotion and settle, compared end to end."""
    ref_t, ref_f, ref_d = run_on("cpu", monkeypatch, SmoothEngine())
    got_t, got_f, got_d = run_on("torch-cpu", monkeypatch, SmoothEngine())
    np.testing.assert_allclose(got_t, ref_t, atol=2e-4)
    for s in ref_f:
        assert np.abs(ref_f[s]).max() > 1e-3, "the reference run moved nothing"
        np.testing.assert_allclose(got_f[s], ref_f[s], atol=1e-4)
    np.testing.assert_allclose(got_d, ref_d, rtol=1e-4)


def test_the_torch_path_agrees_with_demons_statistically(monkeypatch):
    """Demons thresholds turn rounding into scattered small differences only."""
    ref_t, ref_f, ref_d = run_on("cpu", monkeypatch)
    got_t, got_f, got_d = run_on("torch-cpu", monkeypatch)
    assert float(np.abs(got_t - ref_t).mean()) < 1e-4
    for s in ref_f:
        diff = fields.percentile_magnitude(got_f[s] - ref_f[s], 99)
        scale = fields.percentile_magnitude(ref_f[s], 99)
        assert scale > 0, "the reference run moved nothing"
        assert diff < 0.02 * scale, (s, diff, scale)
    np.testing.assert_allclose(got_d, ref_d, rtol=0.01)


def test_passes_keep_their_arrays_on_the_device(monkeypatch):
    """Reads come back as device tensors, so nothing downstream is NumPy."""
    monkeypatch.setenv("CHUNKREG_DEVICE", "torch-cpu")
    cohort()
    xp.configure("torch-cpu")
    vol = Volume.open("subjects/a.zarr")
    block = vol.read_padded(0, (0, 0, 0), GRIDS[0].shape, normalise=True)
    assert isinstance(block, torch.Tensor)
    f = Field.create("f.zarr", GRIDS[0], PROFILE)
    assert isinstance(f.read_dense((0, 0, 0), (4, 4, 4)), torch.Tensor)


def test_demons_refuses_a_gpu_device(monkeypatch):
    from chunkreg.engines import get_engine

    monkeypatch.setattr(xp, "_device", "cuda")
    with pytest.raises(RuntimeError, match="does not run on a\\s+GPU"):
        get_engine("demons").register(np.zeros((1, 4, 4, 4)), np.zeros((1, 4, 4, 4)), 1.0, [])


def test_median_ingest_matches_on_the_device(monkeypatch):
    from chunkreg.store import ingest_source

    data = (blob(SHAPE, seed=2) * 4000).astype(np.uint16)
    ref = ingest_source(data, "ref.zarr", SP, PROFILE, median_radius=1.5)
    with xp.using("torch-cpu"):
        got = ingest_source(data, "dev.zarr", SP, PROFILE, median_radius=1.5)
        g = xp.get(got.read_padded(len(GRIDS) - 1, (0, 0, 0), SHAPE))
    r = ref.read_padded(len(GRIDS) - 1, (0, 0, 0), SHAPE)
    assert np.array_equal(g, r)
    assert got.normalisation == pytest.approx(ref.normalisation)


# --------------------------------------------------------------------------- #
# Multi-GPU runner
# --------------------------------------------------------------------------- #
def test_the_multi_gpu_runner_matches_the_local_runner(tmp_path, monkeypatch):
    pytest.importorskip("numcodecs")
    from chunkreg.runners.multigpu import MultiGPURunner

    body = {
        "root": ".", "backend": "zarr", "profile": "i1", "gpu_mem_gb": 1000.0,
        "device": "cpu", "engine": "demons",
        "profile_overrides": {"core": 16, "halo": 16, "inner_chunk": 8, "k": 3,
                              "sigma_g": 0.5, "sigma_w": 0.5, "scales": [2, 1]},
        "subjects": [{"id": s, "path": f"subjects/{s}.zarr"} for s in "abc"],
        "levels": {"caps": [1, 1, 1],
                   "level0_stages": [{"greedy": {"scales": [2, 1], "iterations": [4, 3]}}],
                   "seeded_stages": [{"greedy": {"scales": [2, 1], "iterations": [3, 2]}}],
                   "sharpen_laplacian_levels": []},
    }
    results = {}
    for name in ("local", "multi"):
        root = tmp_path / name
        root.mkdir()
        monkeypatch.chdir(root)
        body_path = root / "run.json"
        body_path.write_text(json.dumps(body), encoding="utf-8")
        cfg = load_config(body_path)
        cohort(backend="zarr", root=".")
        if name == "local":
            result = build_template(cfg, runner=LocalRunner(), config_path=str(body_path))
        else:
            with MultiGPURunner(cfg, gpus=["0", "1"], engine="demons", device="cpu",
                                poll_seconds=0.5) as runner:
                result = build_template(cfg, runner=runner, config_path=str(body_path))
        grid = result.template.native_grid
        results[name] = (
            result.template.read_padded(0, (0, 0, 0), grid.shape),
            {s: f.read_dense((0, 0, 0), grid.shape) for s, f in result.fields.items()},
            [(lv.level, p.iteration, p.d99_mm) for lv in result.levels for p in lv.passes],
        )
    lt, lf, ld = results["local"]
    mt, mf, md = results["multi"]
    assert ld == md, "the stopping rule saw the same records from the workers"
    np.testing.assert_allclose(mt, lt, atol=1e-5)
    for s in lf:
        np.testing.assert_allclose(mf[s], lf[s], atol=1e-4)


def test_a_worker_that_cannot_start_fails_the_pass_with_its_reason(tmp_path):
    from chunkreg.runners.multigpu import MultiGPURunner

    runner = MultiGPURunner(gpus=["0"], engine="demons", device="cpu",
                            config_path=tmp_path / "missing.json", poll_seconds=0.2)
    try:
        report = runner.run("blend", None, [0, 1])
    finally:
        runner.close()
    assert not report.ok
    assert "missing.json" in report.failed[0].error

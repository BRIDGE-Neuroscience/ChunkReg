"""Task arrays are freed as soon as their last reader is done, not at the pass end."""
from __future__ import annotations

from chunkreg.runners.multigpu import ReleaseTracker


def test_an_upstream_item_is_freed_after_its_last_reader():
    # blend 0 reads register 0 and 1; blend 1 reads register 1 and 2
    blocked = {("blend", 0): {("register", 0), ("register", 1)},
               ("blend", 1): {("register", 1), ("register", 2)}}
    t = ReleaseTracker(blocked)
    assert t.pending() == 3
    assert t.done(("blend", 0)) == [("register", 0)]  # register 1 still has blend 1 to come
    assert t.pending() == 2
    freed = t.done(("blend", 1))
    assert sorted(freed) == [("register", 1), ("register", 2)]
    assert t.pending() == 0
    # finishing something nobody tracks is harmless
    assert t.done(("blend", 7)) == []


def test_free_upstream_runs_during_the_chain(tmp_path, monkeypatch):
    """Through the runner itself, with CPU workers: every register task is
    freed, and the first of them before the chain has finished."""
    import json

    import pytest

    pytest.importorskip("numcodecs")
    from chunkreg.runners.multigpu import MultiGPURunner

    body = {
        "root": ".", "backend": "zarr", "profile": "i1", "gpu_mem_gb": 1000.0,
        "device": "cpu", "engine": "demons",
        "profile_overrides": {"core": 16, "halo": 16, "inner_chunk": 8, "k": 3,
                              "sigma_g": 0.5, "sigma_w": 0.5, "scales": [2, 1]},
        "subjects": [{"id": s, "path": f"subjects/{s}.zarr"} for s in "ab"],
        "template_subject": "a",
        "levels": {"caps": [1, 1, 1],
                   "level0_stages": [{"greedy": {"scales": [2, 1], "iterations": [2, 2]}}],
                   "seeded_stages": [{"greedy": {"scales": [2, 1], "iterations": [2, 2]}}]},
    }
    monkeypatch.chdir(tmp_path)
    (tmp_path / "run.json").write_text(json.dumps(body), encoding="utf-8")
    from chunkreg import backend as _backend
    from chunkreg.config import load_config
    from chunkreg.grid import GridSpec
    from chunkreg.store import ingest_array
    import numpy as np

    _backend.set_default_backend("zarr")
    try:
        cfg = load_config(tmp_path / "run.json")
        rng = np.random.default_rng(0)
        from scipy import ndimage

        for sid in "ab":
            v = ndimage.gaussian_filter(rng.random((48, 48, 48)), 3.0).astype(np.float32)
            ingest_array(v, cfg.subject_path(sid), 1.0, cfg.profile, backend="zarr", overwrite=True)
        from chunkreg.pipelines.groupwise import build_template

        freed: list[int] = []
        import chunkreg.runners.multigpu as mg

        real = mg.MultiGPURunner._dispatch

        def spy(self, stages, blocked_on, params, free_upstream=None):
            def wrapped(tid):
                freed.append(tid)
                free_upstream(tid)
            return real(self, stages, blocked_on, params,
                        free_upstream=None if free_upstream is None else wrapped)

        monkeypatch.setattr(mg.MultiGPURunner, "_dispatch", spy)
        with MultiGPURunner(cfg, gpus=["0", "1"], engine="demons", device="cpu",
                            poll_seconds=0.5) as runner:
            build_template(cfg, runner=runner, config_path=str(tmp_path / "run.json"))
        assert freed, "no task array was freed during the chain"
        # nothing is left for the end-of-pass cleanup to find
        for d in (tmp_path / "scratch").glob("L*_it*"):
            assert not list(d.glob("task_*.zarr")), f"task arrays left in {d}"
    finally:
        _backend.set_default_backend("memory")

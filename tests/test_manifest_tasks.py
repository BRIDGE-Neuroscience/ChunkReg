"""A pass has at least as many tasks as the runner has workers."""
from __future__ import annotations

from chunkreg.config import load_config
from chunkreg.grid import GridSpec
from chunkreg.passes.manifest import build_manifest


def _cfg(tmp_path, subjects, per_task=16):
    return load_config({
        "root": str(tmp_path), "profile": "i1",
        "profile_overrides": {"core": 16, "halo": 16, "inner_chunk": 8, "k": 3,
                              "sigma_g": 0.5, "sigma_w": 0.5},
        "subjects": [{"id": s, "path": f"{s}.zarr"} for s in subjects],
        "slurm": {"register": {"chunks_per_task": per_task}},
    })


def test_small_level_is_split_over_the_workers(tmp_path):
    cfg = _cfg(tmp_path, ["a"])
    grid = GridSpec((32, 32, 32), 1.0)  # 2x2x2 = 8 chunks, 8 entries
    assert build_manifest(cfg, 0, 0, grid).n_tasks == 1
    m = build_manifest(cfg, 0, 0, grid, min_tasks=2)
    assert m.n_tasks == 2 and all(len(t) == 4 for t in m.tasks)
    m = build_manifest(cfg, 0, 0, grid, min_tasks=8)
    assert m.n_tasks == 8
    # more workers than entries: one entry per task, no empty tasks
    assert build_manifest(cfg, 0, 0, grid, min_tasks=50).n_tasks == 8


def test_subjects_of_a_chunk_stay_together(tmp_path):
    cfg = _cfg(tmp_path, ["a", "b", "c"], per_task=15)
    grid = GridSpec((32, 32, 32), 1.0)  # 8 chunks x 3 subjects = 24 entries
    m = build_manifest(cfg, 0, 0, grid, min_tasks=6)
    assert m.n_tasks == 8  # 24 / 6 = 4 -> rounded down to one chunk of 3 subjects per task
    for t in m.tasks:
        assert len({e.chunk_id for e in t.entries}) == 1 and len(t) == 3

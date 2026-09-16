"""Starting a run again carries on from where it stopped, never from scratch."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import ndimage

from chunkreg import fields
from chunkreg.cli import main
from chunkreg.config import LevelPolicy, RunConfig, StageSpec, SubjectSpec
from chunkreg.grid import GridSpec, Profile, pyramid
from chunkreg.passes import build_manifest, register as _register
from chunkreg.pipelines import build_template, groupwise
from chunkreg.runners import LocalRunner
from chunkreg.store import Field, ingest_array

from .conftest import blob

SPACING = 0.05
SHAPE = (48, 48, 48)
PROFILE = Profile(core=16, halo=8, inner_chunk=8, lattice_factor=2, channels=1,
                  features="intensity", r_f=0, k=3, sigma_g=0.5, sigma_w=0.5)
GRIDS = pyramid(GridSpec(SHAPE, SPACING), PROFILE)  # three levels


@pytest.fixture
def cohort():
    truth = blob(SHAPE, seed=3, smooth=2.0)
    rng = np.random.default_rng(1)
    for sid in ("a", "b"):
        u = rng.normal(0, 1, (3,) + SHAPE).astype(np.float32)
        u = np.stack([ndimage.gaussian_filter(u[d], 6.0) for d in range(3)])
        u *= 2 * SPACING / np.abs(u).max()
        ingest_array((fields.warp(truth, u, SPACING) * 4000).astype(np.uint16),
                     f"subjects/{sid}.zarr", SPACING, PROFILE)


def make_cfg(stop_at=None, caps=(2, 1, 1)) -> RunConfig:
    return RunConfig(
        root=".",
        subjects=(SubjectSpec("a", "subjects/a.zarr"), SubjectSpec("b", "subjects/b.zarr")),
        profile=PROFILE,
        profile_name="test",
        backend="memory",
        gpu_mem_gb=1000.0,
        levels=LevelPolicy(
            caps=caps,
            level0_stages=(StageSpec(kind="greedy", scales=(2, 1), iterations=(4, 3)),),
            seeded_stages=(StageSpec(kind="greedy", scales=(2, 1), iterations=(3, 2)),),
            sharpen_laplacian_levels=(),
            stop_at=stop_at,
        ),
    )


@pytest.fixture
def passes_run(monkeypatch):
    """Every (level, iteration) that actually ran a pass, in order."""
    seen = []
    real = groupwise._run_pass

    def counting(cfg, runner, engine, extractor, level, iteration, *rest):
        seen.append((level, iteration))
        return real(cfg, runner, engine, extractor, level, iteration, *rest)

    monkeypatch.setattr(groupwise, "_run_pass", counting)
    return seen


def lines(messages):
    return lambda m: messages.append(m)


def test_a_finished_run_started_again_does_no_passes(cohort, passes_run):
    cfg = make_cfg()
    first = build_template(cfg, runner=LocalRunner())
    ran = list(passes_run)
    assert ran, "the first run did nothing"
    messages = []
    again = build_template(cfg, runner=LocalRunner(), progress=lines(messages))
    assert passes_run == ran, "a finished run was redone"
    assert all("done earlier" in m for m in messages if " pass " in m)
    assert [lv.n_passes for lv in again.levels] == [lv.n_passes for lv in first.levels]


def test_starting_again_does_not_reseed_level_zero(cohort):
    cfg = make_cfg()
    build_template(cfg, runner=LocalRunner())
    before = Field.open(cfg.field_path(0, "a"), cfg.backend).read_lattice((0, 0, 0), (3, 3, 3))
    assert np.abs(before).max() > 0
    build_template(cfg, runner=LocalRunner())
    after = Field.open(cfg.field_path(0, "a"), cfg.backend).read_lattice((0, 0, 0), (3, 3, 3))
    np.testing.assert_array_equal(before, after)


def test_a_run_killed_part_way_resumes_at_the_pass_it_lost(cohort, monkeypatch):
    cfg = make_cfg()
    real = groupwise._run_pass

    def dies_at_level_one(cfg_, runner, engine, extractor, level, iteration, *rest):
        if level == 1:
            raise KeyboardInterrupt("walltime")
        return real(cfg_, runner, engine, extractor, level, iteration, *rest)

    monkeypatch.setattr(groupwise, "_run_pass", dies_at_level_one)
    with pytest.raises(KeyboardInterrupt):
        build_template(cfg, runner=LocalRunner())

    counted = []

    def counting(cfg_, runner, engine, extractor, level, iteration, *rest):
        counted.append((level, iteration))
        return real(cfg_, runner, engine, extractor, level, iteration, *rest)

    # Not monkeypatch.undo(): that would also undo the per-test working
    # directory, and the run would write its records into the repository.
    monkeypatch.setattr(groupwise, "_run_pass", counting)
    result = build_template(cfg, runner=LocalRunner())
    assert counted and all(level >= 1 for level, _ in counted), counted
    assert [lv.level for lv in result.levels] == [0, 1, 2]


def test_an_early_stop_is_continued_by_a_later_run(cohort, passes_run):
    short = build_template(make_cfg(stop_at=0), runner=LocalRunner())
    assert [lv.level for lv in short.levels] == [0]
    assert short.fields["a"].level_grid == GRIDS[0]
    level0 = list(passes_run)

    messages = []
    full = build_template(make_cfg(), runner=LocalRunner(), progress=lines(messages))
    assert [lv.level for lv in full.levels] == [0, 1, 2]
    # Level 0 may add a pass: with a finer level to hand on to, it has to meet
    # the stricter rule. What it must not do is run a pass it already has.
    assert len(set(passes_run)) == len(passes_run), "a finished pass was redone"
    assert "promote level 0 -> 1" in messages
    assert full.fields["a"].level_grid == GRIDS[2]


def test_a_stop_at_can_name_a_spacing():
    cfg = make_cfg(stop_at=0.1)
    assert cfg.levels.run_levels(GRIDS) == (0, 1)


def test_from_level_redoes_that_level_and_finer_only(cohort, passes_run):
    cfg = make_cfg()
    build_template(cfg, runner=LocalRunner())
    first = list(passes_run)
    build_template(cfg, runner=LocalRunner(), from_level=1)
    redone = passes_run[len(first):]
    assert redone and all(level >= 1 for level, _ in redone)
    assert {level for level, _ in redone} == {1, 2}


def test_a_level_handed_on_is_not_extended(cohort, passes_run):
    """Raising a cap after the next level started must not rerun the level."""
    build_template(make_cfg(caps=(1, 1, 1)), runner=LocalRunner())
    first = list(passes_run)
    build_template(make_cfg(caps=(3, 1, 1)), runner=LocalRunner())
    assert passes_run == first


def test_a_torn_pass_record_counts_as_not_done(cohort, passes_run):
    cfg = make_cfg(stop_at=0, caps=(1,))
    build_template(cfg, runner=LocalRunner())
    cfg.pass_record_path(0, 0).write_text("{", encoding="utf-8")
    build_template(cfg, runner=LocalRunner())
    assert passes_run == [(0, 0), (0, 0)]


def test_a_finished_register_task_still_reports_its_records(cohort):
    cfg = make_cfg()
    build_template(make_cfg(stop_at=0, caps=(1,)), runner=LocalRunner())
    manifest = build_manifest(cfg, 0, 5, GRIDS[0])
    first = _register.run_register_task(cfg, manifest, 0)
    second = _register.run_register_task(cfg, manifest, 0)
    assert second["skipped"]
    assert second["records"] == first["records"]


# --------------------------------------------------------------------------- #
# Through the command line
# --------------------------------------------------------------------------- #
def test_stop_at_on_the_command_line(tmp_path, capsys):
    for sid in ("a", "b"):
        ingest_array((blob(SHAPE, seed=len(sid) + 1) * 4000).astype(np.uint16),
                     f"subjects/{sid}.zarr", SPACING, PROFILE)
    body = {
        "root": ".", "backend": "memory", "profile": "i1", "gpu_mem_gb": 1000.0,
        "profile_overrides": {"core": 16, "halo": 16, "inner_chunk": 8, "k": 3,
                              "sigma_g": 0.5, "sigma_w": 0.5, "scales": [2, 1]},
        "subjects": [{"id": "a"}, {"id": "b"}],
        "levels": {"caps": [1],
                   "level0_stages": [{"greedy": {"scales": [2, 1], "iterations": [3, 2]}}],
                   "seeded_stages": [{"greedy": {"scales": [2, 1], "iterations": [2, 2]}}],
                   "sharpen_laplacian_levels": []},
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(body), encoding="utf-8")

    assert main(["run", str(path), "--stop-at", "100um"]) == 0
    out = capsys.readouterr().out
    assert "stopping at level 1 (100 um)" in out
    assert "running levels [0, 1] of 0..2" in out

    assert main(["run", str(path)]) == 0
    out = capsys.readouterr().out
    assert "level 0 pass 0: done earlier" in out
    assert "promote level 1 -> 2" in out

    assert main(["run", str(path), "--stop-at", "1mm"]) == 1
    err = capsys.readouterr().err
    assert "1000 um" in err and "0 = 200 um" in err

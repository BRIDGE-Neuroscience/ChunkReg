"""Choosing which pyramid levels run: stopping early and skipping levels."""

from __future__ import annotations

import json

import numpy as np
import pytest

from chunkreg import fields
from chunkreg.cli import main
from chunkreg.config import (
    ConfigError,
    LevelPolicy,
    RunConfig,
    StageSpec,
    SubjectSpec,
    get_profile,
    load_config,
)
from chunkreg.grid import GridSpec, Profile, pyramid
from chunkreg.pipelines import build_template
from chunkreg.planner import plan
from chunkreg.runners import LocalRunner
from chunkreg.store import Field, Volume, ingest_array

from .conftest import blob

HIPCT = GridSpec((2500, 2500, 2500), 0.05)  # levels 800, 400, 200, 100, 50 um


def load(run, **levels):
    body = {"root": ".", "profile": "a16", "subjects": [{"id": "a"}],
            "levels": dict(levels)}
    if run is not None:
        body["levels"]["run"] = run
    return load_config(body)


def resolved(run, **levels):
    cfg = load(run, **levels)
    return cfg.levels.run_levels(pyramid(HIPCT, cfg.profile))


# --------------------------------------------------------------------------- #
# Reading levels.run
# --------------------------------------------------------------------------- #
def test_without_run_every_level_runs():
    assert resolved(None) == (0, 1, 2, 3, 4)
    assert resolved("all") == (0, 1, 2, 3, 4)


def test_leaving_out_the_finest_levels_stops_early():
    assert resolved([0, 1, 2]) == (0, 1, 2)


def test_levels_can_be_named_by_spacing():
    assert resolved([0, "400um", "0.2mm"]) == (0, 1, 2)
    assert resolved(["800 um", "200µm"]) == (0, 2)


def test_a_middle_level_can_be_skipped():
    assert resolved([0, 2, 4]) == (0, 2, 4)


def test_listed_runs_level_zero_and_the_tuned_levels():
    assert resolved("listed", level_params={"2": {"lr": 0.2}}) == (0, 2)
    assert resolved(
        "listed",
        level_params={"1": {"lr": 0.2}},
        level_stages={"3": [{"greedy": {"scales": [1], "iterations": [2]}}]},
    ) == (0, 1, 3)


def test_order_and_repeats_do_not_matter():
    assert resolved([2, 0, 2, "800um"]) == (0, 2)


def test_level_zero_cannot_be_left_out():
    with pytest.raises(ConfigError, match="leaves out level 0"):
        load([1, 2])
    with pytest.raises(ConfigError, match="leaves out level 0"):
        resolved(["400um"])


def test_a_spacing_the_pyramid_lacks_lists_the_real_levels():
    with pytest.raises(ConfigError) as e:
        resolved([0, "1mm"])
    assert "1000 um" in str(e.value)
    assert "0 = 800 um" in str(e.value) and "4 = 50 um" in str(e.value)


def test_a_level_past_the_pyramid_is_refused():
    with pytest.raises(ConfigError, match="names level 7"):
        resolved([0, 7])


def test_listed_refuses_a_tuned_level_the_pyramid_lacks():
    with pytest.raises(ConfigError, match="'listed'"):
        resolved("listed", level_params={"9": {"lr": 0.2}})


@pytest.mark.parametrize(
    "bad, message",
    [
        ([0, 0.2], "ambiguous"),
        ([0, "fine"], "not a level"),
        ([0, "0um"], "not a level"),
        ([0, True], "not a level"),
        ("some", "'all', 'listed'"),
        ([], "empty"),
        ([0, -1], "negative"),
        (3, "list of levels"),
    ],
)
def test_malformed_run_entries_are_refused(bad, message):
    with pytest.raises(ConfigError, match=message):
        load(bad)


def test_the_planner_costs_only_the_levels_that_run():
    full = plan(load(None), HIPCT)
    short = plan(load([0, 1, 2]), HIPCT)
    assert [lv.level for lv in short.levels] == [0, 1, 2]
    assert short.work_ch_vox < full.work_ch_vox / 10
    assert "3 level(s) run of 5 in the pyramid" in short.format()


# --------------------------------------------------------------------------- #
# Running a subset
# --------------------------------------------------------------------------- #
SPACING = 0.05
SHAPE = (48, 48, 48)
# A 16-voxel core against a 48-voxel volume gives three levels, the smallest
# pyramid with a middle level to skip.
PROFILE = Profile(core=16, halo=8, inner_chunk=8, lattice_factor=2, channels=1,
                  features="intensity", r_f=0, k=3, sigma_g=0.5, sigma_w=0.5)
GRIDS = pyramid(GridSpec(SHAPE, SPACING), PROFILE)


@pytest.fixture
def cohort():
    truth = blob(SHAPE, seed=3, smooth=2.0)
    rng = np.random.default_rng(1)
    for sid in ("a", "b"):
        u = rng.normal(0, 1, (3,) + SHAPE).astype(np.float32)
        from scipy import ndimage

        u = np.stack([ndimage.gaussian_filter(u[d], 6.0) for d in range(3)])
        u *= 2 * SPACING / np.abs(u).max()
        ingest_array((fields.warp(truth, u, SPACING) * 4000).astype(np.uint16),
                     f"subjects/{sid}.zarr", SPACING, PROFILE)


def make_cfg(run) -> RunConfig:
    return RunConfig(
        root=".",
        subjects=(SubjectSpec("a", "subjects/a.zarr"), SubjectSpec("b", "subjects/b.zarr")),
        profile=PROFILE,
        profile_name="test",
        backend="memory",
        gpu_mem_gb=1000.0,
        levels=LevelPolicy(
            caps=(1,),
            level0_stages=(StageSpec(kind="greedy", scales=(2, 1), iterations=(4, 3)),),
            seeded_stages=(StageSpec(kind="greedy", scales=(2, 1), iterations=(3, 2)),),
            sharpen_laplacian_levels=(),
            run=run,
        ),
    )


def test_the_pyramid_has_a_middle_level():
    assert len(GRIDS) == 3


def test_a_run_can_stop_after_level_zero(cohort):
    cfg = make_cfg((0,))
    result = build_template(cfg, runner=LocalRunner())
    assert [lv.level for lv in result.levels] == [0]
    assert not Volume.exists(cfg.template_path(1), cfg.backend)
    assert result.template.native_grid.shape == GRIDS[0].shape
    for f in result.fields.values():
        assert f.level_grid == GRIDS[0], "the deliverable is on the last level run"


def test_a_run_can_skip_a_middle_level(cohort):
    cfg = make_cfg((0, 2))
    result = build_template(cfg, runner=LocalRunner())
    assert [lv.level for lv in result.levels] == [0, 2]
    assert not Volume.exists(cfg.template_path(1), cfg.backend)
    assert not Field.exists(cfg.field_path(1, "a"), cfg.backend)
    assert not (cfg.level_dir(1)).exists(), "nothing is written for a skipped level"
    assert result.template.native_grid.shape == GRIDS[2].shape
    assert result.fields["a"].level_grid == GRIDS[2]


def test_skipping_carries_the_coarse_solution_across_the_gap(cohort):
    """The finest level is seeded from level 0 directly, not from nothing."""
    cfg = make_cfg((0, 2))
    build_template(cfg, runner=LocalRunner())
    coarse = Field.open(cfg.field_path(0, "a"), cfg.backend).read_dense(
        (0, 0, 0), GRIDS[0].shape
    )
    assert fields.percentile_magnitude(coarse, 99) > 0, "level 0 moved nothing"
    seeded = Field.open(cfg.field_path(2, "a"), cfg.backend).read_dense(
        (0, 0, 0), GRIDS[2].shape
    )
    assert fields.percentile_magnitude(seeded, 99) > 0


def test_resuming_at_a_skipped_level_is_refused(cohort):
    with pytest.raises(ValueError, match="not one of the levels"):
        build_template(make_cfg((0, 2)), runner=LocalRunner(), from_level=1)


# --------------------------------------------------------------------------- #
# Through the command line
# --------------------------------------------------------------------------- #
def cli_config(tmp_path, run, **levels) -> str:
    for sid in ("a", "b"):
        ingest_array((blob(SHAPE, seed=len(sid)) * 4000).astype(np.uint16),
                     f"subjects/{sid}.zarr", SPACING, _cli_profile())
    body = {
        "root": ".", "backend": "memory", "profile": "i1", "gpu_mem_gb": 1000.0,
        "profile_overrides": {"core": 16, "halo": 16, "inner_chunk": 8, "k": 3,
                              "sigma_g": 0.5, "sigma_w": 0.5, "scales": [2, 1]},
        "subjects": [{"id": "a"}, {"id": "b"}],
        "levels": {"run": run, **levels},
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return str(path)


def _cli_profile() -> Profile:
    return get_profile("i1", core=16, halo=16, inner_chunk=8, k=3,
                       sigma_g=0.5, sigma_w=0.5, scales=(2, 1))


QUICK = ["--no-calibrate", "--no-probe", "--no-selftest"]


def test_setup_marks_the_skipped_levels(tmp_path, capsys):
    path = cli_config(tmp_path, [0, "50um"])  # levels here are 200, 100, 50 um
    assert main(["setup", path, *QUICK]) == 0
    out = capsys.readouterr().out
    ladder = out[out.index("resolved pyramid"):out.index("stages per level")]
    rows = [line.split() for line in ladder.splitlines()[2:] if line.strip()]
    assert [r[-1] for r in rows] == ["yes", "skip", "yes"]
    assert "levels skipped by levels.run: [1]" in out
    assert "2 level(s) run of 3 in the pyramid" in out


def test_setup_warns_about_tuning_a_skipped_level(tmp_path, capsys):
    path = cli_config(tmp_path, [0, 2], level_params={"1": {"lr": 0.1}})
    assert main(["setup", path, *QUICK]) == 0
    assert "levels.run leaves those levels out" in capsys.readouterr().out


def test_setup_reports_a_bad_spacing_without_a_traceback(tmp_path, capsys):
    path = cli_config(tmp_path, [0, "1mm"])
    assert main(["setup", path, *QUICK]) == 1
    err = capsys.readouterr().err
    assert "1000 um" in err and "Traceback" not in err

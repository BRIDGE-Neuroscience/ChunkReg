"""The command line, exercised through real files where it can be.

These run in-process on the memory backend, so ``setup`` and the commands that
read what it wrote have to share a process. That is a limitation of the test
environment, not of the CLI: with the zarr backend each command stands alone.

The configuration here is JSON, which is the format the two-stage workflow is
documented in. ``load_config`` reads YAML too, and one test pins that.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from chunkreg.cli import main
from chunkreg.grid import GridSpec

from .conftest import blob

nib = pytest.importorskip("nibabel")

SPACING = 0.05
SHAPE = (48, 48, 48)

# A core of 32 against a 48-voxel volume gives a two-level pyramid, which is
# the smallest thing that can show per-level tuning doing anything. Halo 16
# against kernel 3 leaves a clamp of 11 voxels, above the configured floor.
SMALL_PROFILE = {
    "core": 32,
    "halo": 16,
    "inner_chunk": 8,
    "k": 3,
    "sigma_g": 0.5,
    "sigma_w": 0.5,
    "scales": [2, 1],
}


def write_nifti(path, data, spacing=SPACING) -> str:
    affine = np.eye(4) * spacing
    affine[3, 3] = 1.0
    nib.save(nib.Nifti1Image(np.asarray(data).transpose(2, 1, 0), affine), str(path))
    return str(path)


def config_dict(**over) -> dict:
    cfg = {
        "root": ".",
        "backend": "memory",
        "profile": "i1",
        "runner": "local",
        "spacing_mm": SPACING,
        "gpu_mem_gb": 1000.0,
        "profile_overrides": dict(SMALL_PROFILE),
        "subjects": [
            {
                "id": f"s{i:02d}",
                "source": f"s{i:02d}.nii.gz",
                "path": f"subjects/s{i:02d}.zarr",
            }
            for i in range(1, 4)
        ],
        "levels": {
            "caps": [2, 1],
            "level0_stages": [{"greedy": {"scales": [2, 1], "iterations": [20, 12]}}],
            "seeded_stages": [{"greedy": {"scales": [2, 1], "iterations": [15, 10]}}],
            "sharpen_laplacian_levels": [],
        },
    }
    cfg.update(over)
    return cfg


@pytest.fixture
def sources(tmp_path):
    """Three NIfTI subjects on disk, named as the config expects."""
    truth = blob(SHAPE, seed=7, smooth=2.0)
    rng = np.random.default_rng(0)
    for i in range(1, 4):
        shifted = np.roll(truth, i, axis=0) * (1.0 + 0.02 * rng.normal())
        write_nifti(tmp_path / f"s{i:02d}.nii.gz", shifted * 4000)
    return tmp_path


def write_config(tmp_path, name="run.json", **over) -> str:
    path = tmp_path / name
    body = config_dict(**over)
    if path.suffix == ".json":
        path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return str(path)


@pytest.fixture
def ready(sources, tmp_path, capsys):
    """A config whose subjects are ingested, via ``setup``."""
    path = write_config(tmp_path)
    assert main(["setup", path, "--no-calibrate", "--no-probe", "--no-selftest"]) == 0
    capsys.readouterr()
    return path


# --------------------------------------------------------------------------- #
# The two-stage workflow
# --------------------------------------------------------------------------- #
def test_help_and_version():
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0


def test_the_workflow_is_two_commands():
    """setup and run are the workflow; the rest are utilities."""
    from chunkreg.cli import build_parser

    actions = build_parser()._subparsers._group_actions[0].choices
    assert "setup" in actions and "run" in actions
    for gone in ("ingest", "plan", "probe", "calibrate", "template"):
        assert gone not in actions, f"{gone} should be folded into setup or run"


def test_setup_ingests_every_subject_named_in_the_config(sources, tmp_path, capsys):
    path = write_config(tmp_path)
    assert main(["setup", path, "--no-calibrate", "--no-probe", "--no-selftest"]) == 0
    out = capsys.readouterr().out
    for i in range(1, 4):
        assert f"s{i:02d}" in out
    assert "window" in out
    assert "GPU-hours" in out, "setup should end with the plan"


def test_setup_is_idempotent(ready, capsys):
    assert main(["setup", ready, "--no-calibrate", "--no-probe", "--no-selftest"]) == 0
    assert "nothing to do" in capsys.readouterr().out


def test_setup_reports_the_resolved_pyramid(ready, capsys):
    assert main(["setup", ready, "--no-calibrate", "--no-probe", "--no-selftest"]) == 0
    out = capsys.readouterr().out
    assert "resolved pyramid" in out
    assert "stages per level" in out
    assert "um" in out, "spacings are reported in micrometres"


def test_setup_runs_the_probe(ready, capsys):
    assert main(["setup", ready, "--no-calibrate", "--no-selftest", "--pairs", "2"]) == 0
    out = capsys.readouterr().out
    assert "probe on 2 pair(s)" in out
    assert "verdict" in out


def test_run_builds_a_template_and_status_reports_it(ready, capsys):
    assert main(["run", ready]) == 0
    out = capsys.readouterr().out
    assert "level 0 pass 0" in out
    assert "d99" in out

    assert main(["status", ready]) == 0
    status = capsys.readouterr().out
    assert "template" in status


def test_a_subject_without_a_store_or_a_source_says_so(tmp_path, capsys):
    cfg = config_dict()
    for s in cfg["subjects"]:
        s.pop("source")
    path = tmp_path / "run.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        main(["setup", str(path), "--no-calibrate", "--no-probe", "--no-selftest"])
    assert "no 'source'" in str(e.value)


# --------------------------------------------------------------------------- #
# Per-level stage parameters
# --------------------------------------------------------------------------- #
def test_per_level_params_reach_the_stage_table(sources, tmp_path, capsys):
    """What the config says about level 1 is what level 1 will run."""
    levels = config_dict()["levels"]
    levels["level_params"] = {"1": {"iterations": [9, 6], "smooth_warp_sigma": 0.25}}
    path = write_config(tmp_path, levels=levels)
    assert main(["setup", path, "--no-calibrate", "--no-probe", "--no-selftest"]) == 0
    out = capsys.readouterr().out

    table = out[out.index("stages per level") :]
    level1 = table[table.index("level 1") :]
    assert "iters=[9, 6]" in level1
    assert "warp_sigma=0.25" in level1
    assert "level_params" in level1, "the tuned level should be marked as tuned"

    level0 = table[table.index("level 0") : table.index("level 1")]
    assert "iters=[20, 12]" in level0, "level 0 keeps its own stages"


def test_per_level_params_are_used_by_the_run(sources, tmp_path):
    """The resolved stages are what the register pass asks the engine for."""
    from chunkreg.config import load_config

    levels = config_dict()["levels"]
    levels["level_params"] = {"1": {"iterations": [9, 6], "lr": 0.125}}
    cfg = load_config(config_dict(levels=levels))
    assert cfg.levels.stages(1)[0].iterations == (9, 6)
    assert cfg.levels.stages(1)[0].lr == 0.125
    assert cfg.levels.stages(0)[0].iterations == (20, 12)


def test_an_override_for_a_level_that_does_not_exist_warns(sources, tmp_path, capsys):
    levels = config_dict()["levels"]
    levels["level_params"] = {"9": {"iterations": [9, 6]}}
    path = write_config(tmp_path, levels=levels)
    assert main(["setup", path, "--no-calibrate", "--no-probe", "--no-selftest"]) == 0
    out = capsys.readouterr().out
    assert "will not run" in out
    assert "[9]" in out


# --------------------------------------------------------------------------- #
# Formats and failures
# --------------------------------------------------------------------------- #
def test_yaml_configs_still_load(sources, tmp_path, capsys):
    path = write_config(tmp_path, name="run.yaml")
    assert main(["setup", path, "--no-calibrate", "--no-probe", "--no-selftest"]) == 0
    assert "GPU-hours" in capsys.readouterr().out


def test_nifti_round_trip_preserves_axis_order(tmp_path):
    """NIfTI is (X, Y, Z) and the pipeline is (Z, Y, X); the swap must be exact."""
    from chunkreg.io_formats import read_volume, write_volume

    data = np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE)
    p = tmp_path / "rt.nii.gz"
    write_volume(p, data, GridSpec(SHAPE, SPACING))
    assert np.allclose(read_volume(p), data)


def test_selftest_command_passes(capsys):
    assert main(["selftest"]) == 0
    assert "SELFTEST PASSED" in capsys.readouterr().out


def test_status_before_anything_ran(ready, capsys):
    assert main(["status", ready]) == 0
    assert "no levels started" in capsys.readouterr().out


def test_errors_are_reported_without_a_traceback(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"root": ".", "nonsense": 1}), encoding="utf-8")
    assert main(["setup", str(bad)]) == 1
    err = capsys.readouterr().err
    assert "unknown configuration keys" in err
    assert "Traceback" not in err


def test_malformed_json_is_reported_as_such(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text('{"root": ".",}', encoding="utf-8")
    assert main(["setup", str(bad)]) == 1
    err = capsys.readouterr().err
    assert "not valid JSON" in err


def test_a_mistyped_tuning_key_is_rejected(tmp_path, capsys):
    levels = config_dict()["levels"]
    levels["level_params"] = {"1": {"smooth_wrap_sigma": 0.3}}
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(config_dict(levels=levels)), encoding="utf-8")
    assert main(["setup", str(bad)]) == 1
    err = capsys.readouterr().err
    assert "unknown per-level stage parameter" in err
    assert "smooth_warp_sigma" in err, "the message should name the real key"


def test_run_task_without_a_manifest_says_what_to_do(ready, capsys):
    with pytest.raises(SystemExit) as e:
        main(
            ["run-task", ready, "--level", "0", "--iter", "0",
             "--pass", "register", "--id", "0"]
        )
    assert "no manifest" in str(e.value)
    assert "chunkreg run" in str(e.value)

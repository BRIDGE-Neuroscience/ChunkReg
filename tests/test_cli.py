"""The command line, exercised through real files where it can be.

These run in-process on the memory backend, so ``ingest`` and the commands that
read what it wrote have to share a process. That is a limitation of the test
environment, not of the CLI: with the zarr backend each command stands alone.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from chunkreg import fields
from chunkreg.cli import main
from chunkreg.grid import GridSpec

from .conftest import blob

nib = pytest.importorskip("nibabel")

SPACING = 0.05
SHAPE = (48, 48, 48)


def write_nifti(path, data, spacing=SPACING) -> str:
    affine = np.eye(4) * spacing
    affine[3, 3] = 1.0
    nib.save(nib.Nifti1Image(np.asarray(data).transpose(2, 1, 0), affine), str(path))
    return str(path)


@pytest.fixture
def ingested(tmp_path, capsys):
    """Three subjects ingested through the CLI from NIfTI files."""
    truth = blob(SHAPE, seed=7, smooth=2.0)
    rng = np.random.default_rng(0)
    for i in range(1, 4):
        shifted = np.roll(truth, i, axis=0) * (1.0 + 0.02 * rng.normal())
        src = write_nifti(tmp_path / f"s{i:02d}.nii.gz", (shifted * 4000))
        assert main(
            [
                "ingest", src, f"subjects/s{i:02d}.zarr",
                "--spacing", str(SPACING),
                "--profile", "i1",
                "--backend", "memory",
            ]
        ) == 0
    capsys.readouterr()

    cfg = {
        "root": ".",
        "backend": "memory",
        "profile": "i1",
        "runner": "local",
        "spacing_mm": SPACING,
        "gpu_mem_gb": 1000.0,
        "subjects": [
            {"id": f"s{i:02d}", "path": f"subjects/s{i:02d}.zarr"} for i in range(1, 4)
        ],
        "levels": {
            "caps": [2, 2],
            "level0_stages": [
                {"greedy": {"scales": [2, 1], "iterations": [20, 12]}}
            ],
            "seeded_stages": [
                {"greedy": {"scales": [2, 1], "iterations": [15, 10]}}
            ],
            "sharpen_laplacian_levels": [],
        },
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


def test_help_and_version():
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0


def test_ingest_reads_nifti_and_reports_the_window(tmp_path, capsys):
    src = write_nifti(tmp_path / "v.nii.gz", blob(SHAPE) * 4000)
    assert main(
        ["ingest", src, "v.zarr", "--spacing", "0.05", "--profile", "i1",
         "--backend", "memory"]
    ) == 0
    out = capsys.readouterr().out
    assert "ingested" in out
    assert "levels" in out
    assert "window" in out


def test_nifti_round_trip_preserves_axis_order(tmp_path):
    """NIfTI is (X, Y, Z) and the pipeline is (Z, Y, X); the swap must be exact."""
    from chunkreg.io_formats import read_volume, write_volume

    data = np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE)
    p = tmp_path / "rt.nii.gz"
    write_volume(p, data, GridSpec(SHAPE, SPACING))
    assert np.allclose(read_volume(p), data)


def test_plan_prints_the_resolved_pyramid(ingested, capsys):
    assert main(["plan", ingested]) == 0
    out = capsys.readouterr().out
    assert "halo budget" in out
    assert "GPU-hours" in out
    assert "profile i1" in out


def test_probe_reports_a_residual_and_a_verdict(ingested, capsys):
    assert main(["probe", ingested, "--pairs", "2"]) == 0
    out = capsys.readouterr().out
    assert "probe on 2 pair(s)" in out
    assert "99th percentile" in out
    assert "verdict" in out


def test_status_before_anything_ran(ingested, capsys):
    assert main(["status", ingested]) == 0
    assert "no levels started" in capsys.readouterr().out


def test_template_runs_and_status_then_reports_progress(ingested, capsys):
    assert main(["template", ingested]) == 0
    out = capsys.readouterr().out
    assert "level 0 pass 0" in out
    assert "d99" in out

    assert main(["status", ingested]) == 0
    status = capsys.readouterr().out
    assert "template" in status
    assert "yes" in status


def test_selftest_command_passes(capsys):
    assert main(["selftest"]) == 0
    assert "SELFTEST PASSED" in capsys.readouterr().out


def test_errors_are_reported_without_a_traceback(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"root": ".", "nonsense": 1}), encoding="utf-8")
    assert main(["plan", str(bad)]) == 1
    err = capsys.readouterr().err
    assert "unknown configuration keys" in err
    assert "Traceback" not in err


def test_run_task_without_a_manifest_says_what_to_do(ingested, capsys):
    with pytest.raises(SystemExit) as e:
        main(
            ["run-task", ingested, "--level", "0", "--iter", "0",
             "--pass", "register", "--id", "0"]
        )
    assert "no manifest" in str(e.value)
    assert "chunkreg template" in str(e.value)

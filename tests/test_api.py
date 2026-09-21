"""The Python interface: everything the CLI does, callable from a script.

``chunkreg run`` and ``chunkreg apply`` were reachable only through argparse,
which left a caller able to import a GridSpec but not to build a template or
use one. These tests pin the surface, and pin that reaching for it does not
cost every importer of the package the whole pipeline.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import chunkreg
from chunkreg import fields
from chunkreg.store import Field, Volume, ingest_array

from .conftest import blob

EXPECTED = {
    "load_config", "RunConfig", "get_profile",
    "plan", "Plan", "calibrate", "probe", "selftest",
    "build_template", "register_pair", "RunResult",
    "apply_field", "export_store",
    "Volume", "Field", "get_runner",
}


def test_the_package_exports_the_api_a_script_needs():
    for name in EXPECTED:
        assert callable(getattr(chunkreg, name)), name
    assert EXPECTED <= set(chunkreg.__all__) <= set(dir(chunkreg))


def test_an_unknown_attribute_is_still_an_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'nonesuch'"):
        chunkreg.nonesuch


def test_importing_the_package_does_not_load_the_pipeline():
    """The exports are lazy, so `import chunkreg` stays a grid-geometry import.

    Eager re-exports would pull the passes, engines and feature extractors into
    every importer, including `chunkreg --version` and anything that only wants
    to compute a pyramid.
    """
    code = (
        "import sys, chunkreg; "
        "print(any(m.startswith(('chunkreg.pipelines', 'chunkreg.passes', "
        "'chunkreg.engines')) for m in sys.modules))"
    )
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root))
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, check=True, env=env, cwd=str(root),
    )
    assert out.stdout.strip() == "False", out.stdout


def test_apply_field_matches_a_whole_volume_warp(small_profile):
    """The blockwise path has to agree with warping the volume in one piece.

    What it is really checking is the margin: each output block reads its
    source with a margin sized from the field, and getting that wrong shows up
    only at block edges, which is exactly where a whole-volume warp differs.
    """
    spacing = 0.05
    data = (blob((64, 64, 64), seed=7, smooth=2.0) * 4000).astype(np.uint16)
    volume = ingest_array(data, "subjects/s01.zarr", spacing, small_profile)
    grid = volume.native_grid

    field = Field.create("fields/s01.zarr", grid, small_profile, subject="s01")
    u = np.zeros((3,) + tuple(grid.shape), dtype=np.float32)
    u[0] = 3 * spacing
    u[2] = -2 * spacing
    field.write_dense((0, 0, 0), u)

    out = chunkreg.apply_field(volume, field, "warped.zarr", profile=small_profile)
    got = np.asarray(out.read_padded(out.n_levels - 1, (0, 0, 0), grid.shape))
    # Against the field as *stored*, not as written: a field lives on a coarser
    # lattice in float16, so comparing with `u` would measure that quantisation
    # rather than the thing under test, which is whether reading each block
    # with its own margin gives the same answer as warping the volume whole.
    stored = np.asarray(field.read_dense((0, 0, 0), grid.shape))
    want = fields.warp(data.astype(np.float32), stored, spacing)
    # The outermost layer is excluded because the two disagree there by
    # design, not by accident: a whole-volume warp treats a sample outside the
    # array as exactly zero, while a block read pads with zeros and then
    # interpolates against them, so an edge sample lands between the two. The
    # interior is where a margin bug would show, and there they must agree
    # exactly, including across the core boundaries at every multiple of 32.
    edge = 4
    inner = (slice(edge, -edge),) * 3
    np.testing.assert_allclose(got[inner], want[inner], atol=1e-3)
    assert np.any(np.abs(np.diff(got, axis=0)) > 0), "a warp that did nothing"


def test_apply_field_refuses_a_level_that_is_not_on_the_fields_grid(small_profile):
    spacing = 0.05
    data = (blob((64, 64, 64), seed=1, smooth=2.0) * 4000).astype(np.uint16)
    volume = ingest_array(data, "subjects/s01.zarr", spacing, small_profile)
    field = Field.create(
        "fields/s01.zarr", volume.native_grid, small_profile, subject="s01"
    )
    assert volume.n_levels > 1, "test needs a pyramid"
    with pytest.raises(ValueError, match="is on"):
        chunkreg.apply_field(volume, field, "warped.zarr", level=0,
                             profile=small_profile)


def test_export_can_pull_a_coarse_level_out_of_a_pyramid(small_profile, tmp_path):
    """How a subject is eyeballed beside a template of the same level."""
    nibabel = pytest.importorskip("nibabel")

    from chunkreg.cli import main

    data = (blob((64, 64, 64), seed=3, smooth=2.0) * 4000).astype(np.uint16)
    vol = ingest_array(data, "subjects/s01.zarr", 0.05, small_profile)
    assert vol.n_levels > 1, "test needs a pyramid"

    coarse, fine = tmp_path / "coarse.nii.gz", tmp_path / "fine.nii.gz"
    assert main(["export", "subjects/s01.zarr", str(coarse), "--level", "0"]) == 0
    assert main(["export", "subjects/s01.zarr", str(fine)]) == 0

    got = nibabel.load(str(coarse)).shape
    assert sorted(got) == sorted(vol.grid(0).shape)
    assert got != nibabel.load(str(fine)).shape, "the default is the finest level"


def test_export_rejects_a_level_the_store_does_not_have(small_profile, tmp_path, capsys):
    """Reported as a message and a failing exit code, never a traceback."""
    from chunkreg.cli import main

    data = (blob((64, 64, 64), seed=4, smooth=2.0) * 4000).astype(np.uint16)
    ingest_array(data, "subjects/s01.zarr", 0.05, small_profile)
    out = tmp_path / "x.nii.gz"
    assert main(["export", "subjects/s01.zarr", str(out), "--level", "9"]) == 1
    assert "out of range" in capsys.readouterr().err
    assert not out.exists()

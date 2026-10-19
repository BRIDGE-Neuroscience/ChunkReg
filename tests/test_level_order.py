"""Levels are numbered as OME-Zarr numbers its datasets: 0 is full resolution.

Level ``l`` of a run is dataset ``l`` of every subject store, ``levels/L<l>``
of the run root, and ``2**l`` times the native spacing. The loop runs from the
coarsest level, ``K``, down to ``0``. Everything a user configures, reads or
finds on disk uses that one number.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import ndimage

from chunkreg import fields
from chunkreg.config import (
    ConfigError,
    LevelPolicy,
    RunConfig,
    StageSpec,
    SubjectSpec,
    load_config,
)
from chunkreg.grid import GridSpec, Profile, pyramid
from chunkreg.passes import build_manifest
from chunkreg.pipelines import build_template
from chunkreg.runners import LocalRunner
from chunkreg.store import Field, Volume, _ome_level_order, ingest_array

from .conftest import blob

SPACING = 0.05
SHAPE = (48, 48, 48)
PROFILE = Profile(core=16, halo=8, inner_chunk=8, lattice_factor=2, channels=1,
                  features="intensity", r_f=0, k=3, sigma_g=0.5, sigma_w=0.5)
GRIDS = pyramid(GridSpec(SHAPE, SPACING), PROFILE)  # 0: 48^3 @ 50 um ... 2: 12^3 @ 200 um
TOP = len(GRIDS) - 1


def _stage(iterations):
    # Kernel and smoothing inside the small profile's halo budget.
    return StageSpec(kind="greedy", scales=(2, 1), iterations=iterations,
                     cc_kernel=3, smooth_grad_sigma=0.5, smooth_warp_sigma=0.5)


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


def make_cfg(**levels) -> RunConfig:
    policy = dict(
        caps=(1,),
        coarsest_stages=(_stage((4, 3)),),
        seeded_stages=(_stage((3, 2)),),
        sharpen_laplacian_levels=(),
    )
    policy.update(levels)
    return RunConfig(
        root=".",
        subjects=(SubjectSpec("a", "subjects/a.zarr"), SubjectSpec("b", "subjects/b.zarr")),
        profile=PROFILE,
        profile_name="test",
        backend="memory",
        gpu_mem_gb=1000.0,
        levels=LevelPolicy(**policy),
    )


# --------------------------------------------------------------------------- #
# Stores
# --------------------------------------------------------------------------- #
def test_volume_array_zero_is_full_resolution(volume_data, small_profile):
    vol = ingest_array(volume_data, "v.zarr", 0.05, small_profile)
    top = vol.n_levels - 1
    assert top >= 1, "test needs a pyramid"
    assert tuple(vol.array(0).shape) == volume_data.shape
    np.testing.assert_array_equal(np.asarray(vol.array(0)[:]), volume_data)
    assert vol.grid(0).spacing_mm == pytest.approx(0.05)
    for level in range(vol.n_levels):
        assert vol.grid(level).spacing_mm == pytest.approx(0.05 * 2**level)
        assert tuple(vol.array(level).shape) == vol.grid(level).shape
    assert vol.coarsest == top
    assert vol.array(top).shape[0] < vol.array(0).shape[0]
    assert vol.meta["level_order"] == "ome"


def test_level_l_is_ome_dataset_l_on_disk(tmp_path, volume_data, small_profile):
    zarr = pytest.importorskip("zarr")
    vol = ingest_array(volume_data, tmp_path / "v.zarr", 0.05, small_profile,
                       backend="zarr")
    root = zarr.open_group(str(tmp_path / "v.zarr"), mode="r")
    for level in range(vol.n_levels):
        assert vol.level_path(level) == tmp_path / "v.zarr" / str(level)
        assert root[str(level)].shape == vol.grid(level).shape
    datasets = root.attrs["ome"]["multiscales"][0]["datasets"]
    assert [d["path"] for d in datasets] == [str(k) for k in range(vol.n_levels)]


def test_a_store_from_the_coarse_first_numbering_is_read_the_ome_way():
    """Its datasets were always OME ordered; only the linked-level key flips."""
    old = {"kind": "volume", "n_levels": 3, "linked": {"2": "/scans/s1.zarr"}}
    new = _ome_level_order(old)
    assert new["linked"] == {"0": "/scans/s1.zarr"}, "the linked level is full resolution"
    assert new["level_order"] == "ome"
    assert _ome_level_order(new) == new, "a store in the OME order is left alone"


def test_field_components_are_labelled_in_array_order(small_profile):
    f = Field.create("f.zarr", GridSpec((32, 32, 32), 0.05), small_profile)
    assert f.meta["components"].startswith("z, y, x displacement")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE = {"root": ".", "profile": "i1", "subjects": [{"id": "s01"}]}


@pytest.mark.parametrize("old, new", [
    ("level0_stages", "coarsest_stages"),
    ("level0_max_disp_frac", "coarsest_max_disp_frac"),
])
def test_the_coarse_first_key_names_are_refused(old, new):
    value = ["moments"] if old == "level0_stages" else 0.25
    with pytest.raises(ConfigError) as e:
        load_config({**BASE, "levels": {old: value}})
    msg = str(e.value)
    assert new in msg and "level 0 is full resolution" in msg
    # The new name loads.
    assert load_config({**BASE, "levels": {new: value}})


def test_caps_index_from_full_resolution():
    policy = load_config({**BASE, "levels": {"caps": [1, 2, 7]}}).levels
    assert [policy.cap(level) for level in range(5)] == [1, 2, 7, 7, 7]


def test_the_coarsest_stages_go_to_the_highest_level():
    policy = LevelPolicy()
    assert policy.stages(4, coarsest=4) == LevelPolicy.coarsest_stages
    assert policy.stages(0, coarsest=4) == LevelPolicy.seeded_stages
    assert policy.stages(0, coarsest=0) == LevelPolicy.coarsest_stages, (
        "a volume that fits one chunk has only level 0, which is the coarsest"
    )


def test_sharpening_defaults_to_the_two_coarsest_levels():
    policy = LevelPolicy()
    assert [policy.sharpens(level, coarsest=4) for level in range(5)] == [
        False, False, False, True, True,
    ]
    named = LevelPolicy(sharpen_laplacian_levels=(0,))
    assert [named.sharpens(level, coarsest=4) for level in range(5)] == [
        True, False, False, False, False,
    ]


# --------------------------------------------------------------------------- #
# A run
# --------------------------------------------------------------------------- #
def test_a_run_writes_its_levels_as_L_ell_with_L0_at_full_resolution(cohort):
    cfg = make_cfg()
    messages = []
    result = build_template(cfg, runner=LocalRunner(), progress=messages.append)
    assert [lv.level for lv in result.levels] == [2, 1, 0]
    for level in range(TOP + 1):
        assert cfg.level_dir(level).name == f"L{level}"
        template = Volume.open(cfg.template_path(level), cfg.backend)
        assert template.native_grid.shape == GRIDS[level].shape
        assert template.native_grid.spacing_mm == pytest.approx(SPACING * 2**level)
        record = json.loads(cfg.pass_record_path(level, 0).read_text())
        assert record["level"] == level
        manifest = json.loads((cfg.level_dir(level) / "manifest_it0.json").read_text())
        assert manifest["level"] == level and manifest["coarsest"] == TOP
        seeded = json.loads(cfg.seed_marker_path(level).read_text())
        assert seeded["from_level"] == (None if level == TOP else level + 1)
    assert cfg.template_path(0).as_posix().endswith("levels/L0/template.zarr")
    assert result.fields["a"].level_grid == GRIDS[0]
    assert "promote level 2 -> 1" in messages and "promote level 1 -> 0" in messages
    assert json.loads(cfg.level_order_path.read_text())["level_order"] == "ome"


def test_per_level_settings_apply_at_the_spacing_their_number_names(cohort):
    """level_params[0] and anneal[1] reach 50 um and 100 um, and nothing else."""
    from chunkreg.engines import get_engine

    real = get_engine("demons")
    calls = []

    class Spy:
        def __getattr__(self, name):
            return getattr(real, name)

        def register(self, fixed, moving, spacing_mm, stages, *a, **kw):
            calls.append((round(spacing_mm, 6), stages[0].iterations,
                          stages[0].smooth_warp_sigma))
            return real.register(fixed, moving, spacing_mm, stages, *a, **kw)

    cfg = make_cfg(
        level_params={0: {"iterations": (2, 1)}},
        anneal={1: {"smooth_warp_sigma": [0.25, 0.1], "passes": 2}},
    )
    build_template(cfg, runner=LocalRunner(), engine=Spy())
    by_spacing = {}
    for s, it, sw in calls:  # the first attempt of each level, before any retry
        by_spacing.setdefault(s, (it, sw))
    assert by_spacing[0.05] == ((2, 1), 0.5), "level 0 is full resolution"
    assert by_spacing[0.1] == ((3, 2), 0.25), "anneal[1] is level 1, 100 um"
    assert by_spacing[0.2] == ((4, 3), 0.5), "level 2 is the coarsest here"


def test_manifests_know_which_level_is_the_coarsest():
    cfg = make_cfg()
    top = build_manifest(cfg, TOP, 0, GRIDS[TOP])
    assert top.coarsest == TOP and top.is_coarsest and not top.seeded
    assert len(top.chunks) == 1
    fine = build_manifest(cfg, 0, 0, GRIDS[0])
    assert fine.coarsest == TOP and not fine.is_coarsest and fine.seeded


def test_a_run_root_from_the_coarse_first_numbering_is_refused(cohort):
    from chunkreg.status import status

    cfg = make_cfg()
    old = cfg.level_dir(0)
    old.mkdir(parents=True)
    (old / "pass_it0.json").write_text(json.dumps({"level": 0, "iteration": 0}))
    with pytest.raises(ConfigError, match="coarse-first") as e:
        build_template(cfg, runner=LocalRunner())
    assert "previous chunkreg version" in str(e.value)
    assert "new run root" in str(e.value)
    with pytest.raises(ConfigError, match="coarse-first"):
        status(cfg)
    assert not cfg.level_order_path.exists(), "the old root was not relabelled"


def test_a_fresh_run_root_is_marked_and_resumes(cohort):
    cfg = make_cfg()
    assert not (cfg.root_path / "levels").exists()
    build_template(cfg, runner=LocalRunner())
    assert cfg.level_order_path.exists()
    messages = []
    build_template(cfg, runner=LocalRunner(), progress=messages.append)
    assert any("done earlier" in m for m in messages)


def test_a_marker_naming_another_order_is_refused(cohort):
    cfg = make_cfg()
    cfg.level_order_path.parent.mkdir(parents=True)
    cfg.level_order_path.write_text(json.dumps({"level_order": "coarse-first"}))
    with pytest.raises(ConfigError, match="level_order"):
        build_template(cfg, runner=LocalRunner())

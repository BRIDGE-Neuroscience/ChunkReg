"""Configuration validation and the planner's arithmetic."""

from __future__ import annotations

import pytest

from chunkreg.config import (
    ConfigError,
    LevelPolicy,
    RunConfig,
    StageSpec,
    SubjectSpec,
    get_profile,
    load_config,
)
from chunkreg.grid import GridSpec, Profile
from chunkreg.planner import plan

HIPCT = GridSpec((1920, 2560, 2560), 0.05)


def cfg(**kw) -> RunConfig:
    base = dict(
        root="/tmp/run",
        subjects=tuple(
            SubjectSpec(f"s{i:02d}", f"subjects/s{i:02d}.zarr") for i in range(1, 11)
        ),
        profile=get_profile("a16"),
        profile_name="a16",
    )
    base.update(kw)
    return RunConfig(**base)


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #
def test_named_profiles_differ_only_in_channels_and_receptive_field():
    a16, a4 = get_profile("a16"), get_profile("a4")
    assert (a16.core, a16.halo) == (a4.core, a4.halo)
    assert a16.channels == 16 and a4.channels == 4
    assert a16.padded == a4.padded


def test_intensity_profile_gets_a_bigger_clamp_for_free():
    """No feature receptive field means the whole remaining halo is clamp."""
    assert get_profile("a16").d_max_vox() == pytest.approx(12.0)
    assert get_profile("i1").d_max_vox() == pytest.approx(36.0)
    assert get_profile("i1").d_max_mm(0.05) == pytest.approx(1.8)


def test_unknown_profile_names_are_rejected():
    with pytest.raises(ConfigError, match="unknown profile"):
        get_profile("enormous")


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def test_stage_accepts_three_spellings():
    assert StageSpec.from_obj("rigid").kind == "rigid"
    s = StageSpec.from_obj({"greedy": {"scales": [2, 1], "iterations": [10, 5]}})
    assert (s.kind, s.scales, s.iterations) == ("greedy", (2, 1), (10, 5))
    assert StageSpec.from_obj({"kind": "affine", "scales": [1], "iterations": [4]}).kind == "affine"


def test_stage_rejects_mismatched_schedules():
    with pytest.raises(ConfigError, match="same length"):
        StageSpec(scales=(4, 2, 1), iterations=(10, 5))
    with pytest.raises(ConfigError, match="coarse to fine"):
        StageSpec(scales=(1, 2), iterations=(5, 10))


def test_moments_stage_needs_no_schedule():
    assert StageSpec(kind="moments", scales=(), iterations=()).s_max == 1


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_a_profile_whose_halo_cannot_pay_for_itself_is_rejected():
    """Four in-chunk scales exhaust a 48 halo before any displacement."""
    bad = get_profile("a16", scales=(4, 2, 1))
    with pytest.raises(ConfigError, match="clamp of"):
        cfg(profile=bad).validate()


def test_the_rejection_says_how_to_fix_it():
    bad = get_profile("a16", scales=(4, 2, 1))
    with pytest.raises(ConfigError) as e:
        cfg(profile=bad).validate()
    msg = str(e.value)
    assert "Raise halo to at least 56" in msg
    assert "receptive field (24)" in msg


def test_a_bigger_halo_buys_back_the_deeper_pyramid():
    ok = get_profile("a16", halo=64, scales=(4, 2, 1))
    assert cfg(profile=ok, gpu_mem_gb=200.0).validate() == []
    assert ok.d_max_vox() == pytest.approx(16.0)


def test_a_seeded_stage_may_not_outrun_the_profile_halo_budget():
    policy = LevelPolicy(
        seeded_stages=(StageSpec(scales=(8, 4, 2, 1), iterations=(4, 4, 4, 4)),)
    )
    with pytest.raises(ConfigError, match="budgets its halo"):
        cfg(levels=policy).validate()


def test_duplicate_subject_ids_are_rejected():
    subs = (SubjectSpec("s01", "a.zarr"), SubjectSpec("s01", "b.zarr"))
    with pytest.raises(ConfigError, match="duplicate subject"):
        cfg(subjects=subs).validate()


def test_no_subjects_is_rejected():
    with pytest.raises(ConfigError, match="at least one subject"):
        cfg(subjects=()).validate()


def test_a_profile_too_large_for_the_device_warns_rather_than_fails():
    warnings = cfg(profile=get_profile("a16"), gpu_mem_gb=40.0).validate()
    assert any("above 80%" in w for w in warnings)


def test_the_small_profiles_fit_their_device_class():
    assert cfg(profile=get_profile("a8"), gpu_mem_gb=40.0).validate() == []
    assert cfg(profile=get_profile("a4"), gpu_mem_gb=24.0).validate() == []


def test_shape_update_step_must_be_a_fraction():
    with pytest.raises(ConfigError, match="shape_update_step"):
        cfg(levels=LevelPolicy(shape_update_step=1.5)).validate()


def test_retry_ladder_steps_are_checked():
    from chunkreg.config import RetryPolicy

    with pytest.raises(ConfigError, match="unknown retry steps"):
        RetryPolicy(ladder=("reboot",))


# --------------------------------------------------------------------------- #
# Paths and identity
# --------------------------------------------------------------------------- #
def test_paths_are_derived_from_the_root():
    c = cfg(root="/scratch/run")
    assert c.template_path(2).as_posix().endswith("levels/L2/template.zarr")
    assert c.field_path(2, "s01").as_posix().endswith("levels/L2/fields/s01.zarr")
    assert c.task_path(4, 1, 7).name == "task_000007.zarr"
    assert c.scratch_dir(4, 1).name == "L4_it1"


def test_unknown_accumulator_is_rejected():
    with pytest.raises(ValueError, match="unknown accumulator"):
        cfg().accumulator_path(0, 0, "average")


def test_fingerprint_tracks_meaningful_changes():
    a = cfg()
    assert a.fingerprint() == cfg().fingerprint()
    assert a.fingerprint() != cfg(profile=get_profile("a4"), profile_name="a4").fingerprint()


# --------------------------------------------------------------------------- #
# YAML loading
# --------------------------------------------------------------------------- #
def test_load_from_a_mapping():
    c = load_config(
        {
            "root": "/scratch/run",
            "profile": "a16",
            "subjects": [{"id": "s01", "path": "a.zarr"}],
            "spacing_mm": 0.05,
            "levels": {"caps": [4, 3, 2], "stop": {"residual_frac": 0.25}},
        }
    )
    assert c.profile.channels == 16
    assert c.levels.caps == (4, 3, 2)
    assert c.levels.residual_frac == 0.25
    assert c.spacing_mm == 0.05


def test_channels_shorthand_selects_a_profile():
    base = {"root": ".", "subjects": [{"id": "s", "path": "a.zarr"}]}
    assert load_config({**base, "channels": "intensity"}).profile.channels == 1
    assert load_config({**base, "channels": "pca:4"}).profile.channels == 4


def test_caps_are_padded_for_deeper_pyramids():
    p = LevelPolicy(caps=(8, 6))
    assert [p.cap(k) for k in range(5)] == [8, 6, 6, 6, 6]


def test_unknown_config_keys_are_rejected():
    with pytest.raises(ConfigError, match="unknown configuration keys"):
        load_config({"root": ".", "subjekts": []})


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #
def test_plan_reproduces_the_worked_example_geometry():
    p = plan(cfg(), HIPCT)
    assert p.depth == 4
    assert [lv.n_chunks for lv in p.levels] == [1, 4, 18, 100, 800]
    assert [round(lv.spacing_mm, 6) for lv in p.levels] == [0.8, 0.4, 0.2, 0.1, 0.05]


def test_plan_clamps_scale_with_level_spacing():
    p = plan(cfg(), HIPCT)
    # Level 0 is one chunk, so the halo bound does not apply; it is bounded by
    # the aperture, a quarter of the volume's shortest extent.
    assert p.levels[0].d_max_mm == pytest.approx(0.25 * 96.0)
    assert p.levels[1].d_max_mm == pytest.approx(4.8)
    assert p.levels[3].d_max_mm == pytest.approx(1.2)
    assert p.levels[4].d_max_mm == pytest.approx(0.6)


def test_plan_reports_the_chunk_field_of_view():
    """L bounds the deformation wavelength a level can resolve."""
    p = plan(cfg(), HIPCT)
    assert p.levels[4].chunk_width_mm == pytest.approx(12.8)
    assert p.levels[1].chunk_width_mm == pytest.approx(102.4)


def test_work_scales_linearly_with_channels_and_subjects():
    a16 = plan(cfg(), HIPCT).work_ch_vox
    a4 = plan(cfg(profile=get_profile("a4"), profile_name="a4"), HIPCT).work_ch_vox
    assert a16 / a4 == pytest.approx(4.0, rel=1e-6)

    few = cfg(subjects=tuple(SubjectSpec(f"s{i}", "x") for i in range(5)))
    assert plan(cfg(), HIPCT).work_ch_vox / plan(few, HIPCT).work_ch_vox == pytest.approx(2.0)


def test_the_native_level_dominates_the_work():
    p = plan(cfg(), HIPCT)
    assert p.dominant_level().level == 4
    assert p.levels[4].work_ch_vox / p.work_ch_vox > 0.5


def test_gpu_hours_follow_from_throughput():
    c = cfg(throughput_ch_vox_per_s=1.5e6)
    p = plan(c, HIPCT)
    assert p.gpu_hours == pytest.approx(p.work_ch_vox / 1.5e6 / 3600)
    assert p.wall_hours(8) == pytest.approx(p.gpu_hours / 8)
    with pytest.raises(ValueError):
        p.wall_hours(0)


def test_a_volume_that_fits_one_chunk_plans_a_single_level():
    p = plan(cfg(), GridSpec((256, 256, 256), 1.0))
    assert p.depth == 0
    assert p.levels[0].n_chunks == 1


def test_plan_formats_a_readable_table():
    text = plan(cfg(), HIPCT).format()
    assert "profile a16" in text
    assert "halo budget: clamp 12 + support 12" in text
    assert "GPU-hours" in text
    assert text.count("\n") > 8


def test_task_count_follows_chunks_per_task():
    from chunkreg.config import Resources, SlurmConfig

    c = cfg(slurm=SlurmConfig(register=Resources(gpus=1, chunks_per_task=16)))
    p = plan(c, HIPCT)
    # 800 chunks x 10 subjects = 8000 pairs, 16 per task.
    assert p.levels[4].n_tasks == 500


# --------------------------------------------------------------------------- #
# Per-level stage parameters
# --------------------------------------------------------------------------- #
def test_level_params_patch_only_the_levels_they_name():
    policy = LevelPolicy(
        seeded_stages=(StageSpec(scales=(2, 1), iterations=(50, 30)),),
        level_params={3: {"iterations": (80, 50), "smooth_warp_sigma": 0.25}},
    )
    assert policy.stages(2)[0].iterations == (50, 30)
    assert policy.stages(2)[0].smooth_warp_sigma == 0.5
    assert policy.stages(3)[0].iterations == (80, 50)
    assert policy.stages(3)[0].smooth_warp_sigma == 0.25
    assert policy.stages(3)[0].scales == (2, 1), "unnamed fields are untouched"


def test_level_params_reach_every_deformable_stage_of_a_level():
    policy = LevelPolicy(
        level_stages={
            1: (
                StageSpec(kind="affine", scales=(2, 1), iterations=(10, 5)),
                StageSpec(kind="greedy", scales=(2, 1), iterations=(10, 5)),
            )
        },
        level_params={1: {"lr": 0.05}},
    )
    assert [s.lr for s in policy.stages(1)] == [0.05, 0.05]


def test_a_moments_stage_ignores_a_schedule_patch():
    """moments carries no schedule, so it must not be handed one."""
    policy = LevelPolicy(
        level_stages={
            0: (StageSpec(kind="moments", scales=(), iterations=()),
                StageSpec(kind="greedy", scales=(2, 1), iterations=(10, 5)))
        },
        level_params={0: {"iterations": (7, 7)}},
    )
    moments, greedy = policy.stages(0)
    assert moments.iterations == ()
    assert greedy.iterations == (7, 7)


def test_level_stages_replace_the_whole_list():
    policy = LevelPolicy(
        level_stages={2: (StageSpec(kind="greedy", scales=(1,), iterations=(4,)),)}
    )
    assert len(policy.stages(2)) == 1
    assert policy.stages(2)[0].iterations == (4,)
    assert policy.stages(1) == LevelPolicy.seeded_stages


def test_a_mistyped_tuning_key_names_the_level_and_the_alternatives():
    policy = LevelPolicy(level_params={2: {"smooth_wrap_sigma": 0.3}})
    with pytest.raises(ConfigError) as e:
        policy.stages(2)
    assert "level 2" in str(e.value)
    assert "smooth_warp_sigma" in str(e.value)


def test_a_tuned_level_is_validated_against_the_halo_budget():
    """The check that guards seeded_stages must guard per-level overrides too."""
    policy = LevelPolicy(
        level_params={3: {"scales": (8, 4, 2, 1), "iterations": (4, 4, 4, 4)}}
    )
    with pytest.raises(ConfigError, match="level 3"):
        cfg(levels=policy).validate()


def test_a_bad_override_is_caught_at_load_not_mid_run():
    with pytest.raises(ConfigError, match="unknown per-level stage parameter"):
        load_config(
            {
                "root": ".",
                "profile": "i1",
                "subjects": [{"id": "s01"}],
                "levels": {"level_params": {"1": {"nonsense": 1}}},
            }
        )


def test_level_keys_may_be_strings_because_json_has_no_integer_keys():
    cfg_json = load_config(
        {
            "root": ".",
            "profile": "i1",
            "subjects": [{"id": "s01"}],
            "levels": {"level_params": {"2": {"lr": 0.2}}},
        }
    )
    assert cfg_json.levels.level_params[2] == {"lr": 0.2}
    assert cfg_json.levels.stages(2)[0].lr == 0.2


def test_overrides_for_levels_the_pyramid_lacks_are_reported():
    from chunkreg.config import unused_level_overrides

    policy = LevelPolicy(level_params={1: {"lr": 0.2}, 7: {"lr": 0.2}})
    assert unused_level_overrides(cfg(levels=policy), n_levels=5) == [7]


def test_tuned_levels_lists_both_kinds_of_override():
    policy = LevelPolicy(
        level_stages={4: (StageSpec(scales=(1,), iterations=(2,)),)},
        level_params={1: {"lr": 0.2}},
    )
    assert policy.tuned_levels() == (1, 4)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def test_a_json_config_file_loads(tmp_path):
    import json as _json

    path = tmp_path / "run.json"
    path.write_text(
        _json.dumps(
            {
                "root": ".",
                "profile": "i1",
                "spacing_mm": 0.05,
                "subjects": [{"id": "s01", "source": "raw/s01.tif"}],
                "levels": {"level_params": {"1": {"lr": 0.3}}},
            }
        ),
        encoding="utf-8",
    )
    loaded = load_config(path)
    assert loaded.subjects[0].source == "raw/s01.tif"
    assert loaded.levels.stages(1)[0].lr == 0.3


def test_the_shipped_json_example_is_valid():
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "configs" / "example.json"
    loaded = load_config(example)
    assert loaded.profile_name == "a16"
    assert loaded.levels.tuned_levels() == (1, 2, 3, 4)
    assert loaded.levels.stages(4)[0].smooth_warp_sigma == 0.30


def test_a_subject_path_defaults_to_the_conventional_location():
    loaded = load_config(
        {"root": "/run", "profile": "i1", "subjects": [{"id": "s01"}]}
    )
    assert loaded.subject_path("s01").as_posix().endswith("subjects/s01.zarr")


def test_a_mistyped_nested_key_is_rejected():
    """Top-level typos were already caught; nested ones were silently ignored."""
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(
            {
                "root": ".",
                "profile": "i1",
                "subjects": [{"id": "s01"}],
                "levels": {"residual_frac": 0.5},
            }
        )


def test_an_ingest_block_is_read():
    loaded = load_config(
        {
            "root": ".",
            "profile": "i1",
            "subjects": [{"id": "s01"}],
            "ingest": {"dtype": "uint8", "median_radius": 1.5},
        }
    )
    assert loaded.ingest.dtype == "uint8"
    assert loaded.ingest.median_radius == 1.5

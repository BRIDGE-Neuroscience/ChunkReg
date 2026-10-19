"""levels.anneal (per-pass smoothing schedule), levels.stop.require_subvoxel and features.anatomix_window."""
import pytest

from chunkreg.config import ConfigError, load_config
from chunkreg.features import anatomix_window, describe


def _raw(tmp_path, **over):
    raw = {
        "root": str(tmp_path),
        "profile": "a16",
        "spacing_mm": 0.5,
        "subjects": [{"id": "a", "source": "a.nii.gz"}, {"id": "b", "source": "b.nii.gz"}],
        "template_subject": "a",
        "grid": {"reference": "a"},
    }
    raw.update(over)
    return raw


def test_anneal_schedule_is_geometric_then_held(tmp_path):
    cfg = load_config(_raw(tmp_path, levels={
        "level_stages": {"0": [{"greedy": {"scales": [2, 1], "iterations": [5, 5]}}]},
        "anneal": {"0": {"smooth_grad_sigma": [4.0, 1.0], "smooth_warp_sigma": [2.0, 0.5], "passes": 3}},
    }))
    # Level 0 is full resolution; the schedule is keyed by that number.
    g = [cfg.levels.stages_for_pass(0, k, coarsest=4)[0] for k in range(5)]
    assert [round(s.smooth_grad_sigma, 6) for s in g] == [4.0, 2.0, 1.0, 1.0, 1.0]
    assert [round(s.smooth_warp_sigma, 6) for s in g] == [2.0, 1.0, 0.5, 0.5, 0.5]
    # a level without a schedule is untouched
    assert cfg.levels.stages_for_pass(1, 7, coarsest=4) == cfg.levels.stages(1, coarsest=4)


def test_plateau_stop_round_trips_and_waits_for_the_anneal(tmp_path):
    from chunkreg.pipelines.groupwise import _plateaued

    cfg = load_config(_raw(tmp_path, levels={
        "anneal": {"0": {"smooth_warp_sigma": [3.0, 1.0], "passes": 4}},
        "stop": {"plateau": {"passes": 2, "rel": 0.1}},
    }))
    assert (cfg.levels.plateau_passes, cfg.levels.plateau_rel, cfg.levels.plateau_after) == (2, 0.1, None)
    again = load_config(cfg.to_config_dict())
    assert again.levels.plateau_passes == 2 and again.levels.plateau_rel == 0.1
    flat = [5.0, 4.0, 3.0, 3.0, 2.95, 2.9, 2.9]
    # the anneal runs 4 passes, so the first pass that may plateau is 4 + 2 = 6
    assert not _plateaued(cfg, 0, 5, flat[:6])
    assert _plateaued(cfg, 0, 6, flat[:7])
    # a level without a schedule can plateau from pass `passes`
    assert _plateaued(cfg, 1, 2, [3.0, 3.0, 3.0])
    assert not _plateaued(cfg, 1, 2, [3.0, 2.0, 1.0])
    # a rising residual is a plateau too
    assert _plateaued(cfg, 1, 2, [3.0, 3.5, 4.0])
    with pytest.raises(ConfigError):
        load_config(_raw(tmp_path, levels={"stop": {"plateau": {"rel": 0.1}}}))


def test_anneal_and_require_subvoxel_round_trip(tmp_path):
    cfg = load_config(_raw(tmp_path, levels={
        "anneal": {"0": {"smooth_warp_sigma": [3.0, 1.0], "passes": 10}},
        "stop": {"require_subvoxel": True},
    }))
    assert cfg.levels.require_subvoxel is True
    again = load_config(cfg.to_config_dict())
    assert again.levels.anneal == cfg.levels.anneal
    assert again.levels.require_subvoxel is True


@pytest.mark.parametrize("bad", [
    {"anneal": {"0": {"smooth_warp_sigma": [3.0], "passes": 4}}},
    {"anneal": {"0": {"smooth_warp_sigma": [3.0, 0.0], "passes": 4}}},
    {"anneal": {"0": {"lr": [1.0, 0.5], "passes": 4}}},
    {"anneal": {"0": {"smooth_warp_sigma": [3.0, 1.0], "passes": 0}}},
])
def test_anneal_rejects_bad_entries(tmp_path, bad):
    with pytest.raises(ConfigError):
        load_config(_raw(tmp_path, levels=bad))


def test_anatomix_window_spec_and_round_trip(tmp_path):
    assert anatomix_window("anatomix:w64") == 64 and anatomix_window("anatomix") is None
    assert describe("anatomix:w64+mindssc@8") == (8, 24)
    cfg = load_config(_raw(tmp_path, features={"mind": False, "sample_channels": None, "anatomix_window": 64}))
    assert cfg.profile.features == "anatomix:w64" and cfg.profile.channels == 16
    again = load_config(cfg.to_config_dict())
    assert again.profile.features == "anatomix:w64"
    off = load_config(_raw(tmp_path, features={"mind": False, "sample_channels": None, "anatomix_window": None}))
    assert off.profile.features == "anatomix"


@pytest.mark.parametrize("win", [32, 80])
def test_anatomix_window_must_be_a_multiple_of_32_from_64(tmp_path, win):
    with pytest.raises(ConfigError):
        load_config(_raw(tmp_path, features={"mind": False, "anatomix_window": win}))


def test_plateau_reads_the_template_step_for_an_estimated_template(tmp_path):
    from chunkreg.grid import GridSpec
    from chunkreg.pipelines.groupwise import PassReport, _should_stop

    raw = _raw(tmp_path, levels={"stop": {"plateau": {"passes": 2, "rel": 0.1}}, "caps": [9]})
    raw.pop("template_subject")
    cfg = load_config(raw)
    grid = GridSpec((16, 16, 16), 1.0)
    rep = lambda it, step: PassReport(level=0, iteration=it, d99_mm=5.0, ubar_mm=step, step_mm=step,
                                      fold_frac=0.0, retries=0, skipped=0, seconds=1.0)
    # still above a voxel, so the template has not converged, but the step stopped shrinking
    stopped, why = _should_stop(cfg, rep(2, 3.0), 0.0, grid, 2, 9, [5, 5, 5], [3.2, 3.0, 3.0])
    assert stopped and "step plateaued" in why
    stopped, _ = _should_stop(cfg, rep(2, 1.5), 0.0, grid, 2, 9, [5, 5, 5], [3.0, 2.0, 1.5])
    assert not stopped

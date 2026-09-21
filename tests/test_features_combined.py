"""MIND-SSC alongside another extractor, and a random channel subset per pass."""

from __future__ import annotations

import numpy as np
import pytest

from chunkreg.config import ConfigError, load_config
from chunkreg.features import describe, get_extractor
from chunkreg.features.combined import ChannelSample, CombinedFeatures, stratified_pick
from chunkreg.features.mindssc import MindSSCFeatures

from .conftest import blob


def patch(seed=0, shape=(20, 22, 18)):
    return blob(shape, seed=seed, smooth=1.5).astype(np.float32)


# --------------------------------------------------------------------------- #
# Specs
# --------------------------------------------------------------------------- #
def test_a_combined_spec_concatenates_its_parts():
    ex = get_extractor("intensity+mindssc")
    assert isinstance(ex, CombinedFeatures)
    assert ex.channels == 13 and ex.r_f == 3 and ex.groups == (1, 12)
    p = patch()
    out = ex(p, 0.05)
    assert out.shape == (13,) + p.shape
    np.testing.assert_allclose(out[0], get_extractor("intensity")(p, 0.05)[0])
    # Inside a combination MIND is raw, as anatomix builds it.
    raw = MindSSCFeatures(normalisation="none")(p, 0.05)
    np.testing.assert_allclose(out[1:], raw, rtol=1e-6)


def test_a_sampled_spec_registers_k_channels():
    ex = get_extractor("intensity+mindssc@5", seed=1)
    assert isinstance(ex, ChannelSample)
    assert ex.channels == 5 and ex.r_f == 3 and ex.name == "intensity+mindssc@5"
    p = patch(1)
    full = get_extractor("intensity+mindssc")(p, 0.05)
    np.testing.assert_allclose(ex(p, 0.05), full[ex.pick], rtol=1e-6)


def test_sampling_every_channel_is_no_sampling():
    assert isinstance(get_extractor("intensity+mindssc@13"), CombinedFeatures)


@pytest.mark.parametrize("bad", ["", "mind", "anatomix@x", "intensity+mindssc@14", "intensity@0"])
def test_bad_specs_are_refused(bad):
    with pytest.raises(ValueError):
        describe(bad)


def test_describe_needs_no_weights():
    assert describe("anatomix") == (16, 24)
    assert describe("anatomix+mindssc") == (28, 24)
    assert describe("anatomix+mindssc@16") == (16, 24)
    assert describe("pca:8+mindssc") == (20, 24)
    assert describe("intensity") == (1, 0)


# --------------------------------------------------------------------------- #
# The draw
# --------------------------------------------------------------------------- #
def test_the_draw_is_a_function_of_the_pass():
    a = stratified_pick((16, 12), 16, seed=0, level=2, iteration=1)
    b = stratified_pick((16, 12), 16, seed=0, level=2, iteration=1)
    np.testing.assert_array_equal(a, b)
    assert len(a) == 16 and len(set(a.tolist())) == 16
    others = {tuple(stratified_pick((16, 12), 16, 0, 2, it)) for it in range(6)}
    assert len(others) > 1, "every pass drew the same channels"
    assert tuple(stratified_pick((16, 12), 16, 7, 2, 1)) != tuple(a)


def test_the_draw_is_stratified_by_part():
    for it in range(20):
        pick = stratified_pick((16, 12), 16, 0, 0, it)
        anat = int((pick < 16).sum())
        assert (anat, 16 - anat) == (9, 7), pick  # 16 * 16/28 and 16 * 12/28


def test_every_part_is_kept_even_when_small():
    for it in range(10):
        pick = stratified_pick((1, 12), 3, 0, 0, it)
        assert 0 in pick, "the one-channel part was dropped"


def test_over_passes_every_channel_takes_part():
    seen = set()
    for it in range(30):
        seen |= set(stratified_pick((16, 12), 8, 0, 1, it).tolist())
    assert seen == set(range(28))


def test_select_replaces_the_previous_draw():
    ex = get_extractor("intensity+mindssc@4", seed=2)
    first = ex.select(0, 0).copy()
    second = ex.select(3, 5).copy()
    again = ex.select(0, 0)
    np.testing.assert_array_equal(first, again)
    assert ex.pick is again
    assert len(second) == 4


def test_without_select_the_default_is_pass_zero():
    a = get_extractor("intensity+mindssc@4", seed=2)
    b = get_extractor("intensity+mindssc@4", seed=2)
    b.select(0, 0)
    np.testing.assert_array_equal(a.pick, b.pick)


# --------------------------------------------------------------------------- #
# The config block
# --------------------------------------------------------------------------- #
BASE = {"root": ".", "subjects": [{"id": "a"}]}


def test_anatomix_profiles_get_mind_within_their_channel_budget():
    p = load_config(BASE).profile
    assert (p.features, p.channels, p.r_f) == ("anatomix+mindssc@16", 16, 24)


def test_mind_can_be_turned_off_and_sampling_widened():
    assert load_config({**BASE, "features": {"mind": False}}).profile.features == "anatomix"
    p = load_config({**BASE, "features": {"sample_channels": None}}).profile
    assert (p.features, p.channels) == ("anatomix+mindssc", 28)
    p = load_config({**BASE, "features": {"sample_channels": 8}}).profile
    assert (p.features, p.channels) == ("anatomix+mindssc@8", 8)


def test_mind_on_intensity_raises_the_receptive_field():
    p = load_config({**BASE, "profile": "i1", "features": {"mind": True}}).profile
    assert (p.features, p.channels, p.r_f) == ("intensity+mindssc", 13, 3)


def test_a_bad_features_block_is_refused():
    with pytest.raises(ConfigError, match="unknown key"):
        load_config({**BASE, "features": {"mindssc": True}})
    with pytest.raises(ConfigError, match="sample_channels"):
        load_config({**BASE, "features": {"sample_channels": 40}})


# --------------------------------------------------------------------------- #
# A register task registers the pass's channels
# --------------------------------------------------------------------------- #
def test_a_register_task_selects_the_channels_of_its_pass(small_profile):
    from dataclasses import replace

    from chunkreg.config import LevelPolicy, RunConfig, StageSpec, SubjectSpec
    from chunkreg.grid import GridSpec, pyramid
    from chunkreg.passes import build_manifest, run_register_task
    from chunkreg.passes.promote import seed_level_zero
    from chunkreg.store import Volume, ingest_array

    profile = replace(small_profile, features="intensity+mindssc@4", channels=4, r_f=3,
                      halo=16)
    for sid in ("a", "b"):
        ingest_array((blob((32, 32, 32), seed=len(sid)) * 4000).astype(np.uint16),
                     f"subjects/{sid}.zarr", 0.05, profile)
    cfg = RunConfig(
        root=".",
        subjects=(SubjectSpec("a", "subjects/a.zarr"), SubjectSpec("b", "subjects/b.zarr")),
        profile=profile, profile_name="test", backend="memory", gpu_mem_gb=1000.0, seed=4,
        levels=LevelPolicy(
            caps=(1,),
            level0_stages=(StageSpec(kind="greedy", scales=(1,), iterations=(2,), cc_kernel=3,
                                     smooth_grad_sigma=0.5),),
            sharpen_laplacian_levels=(),
        ),
    )
    grid = pyramid(GridSpec((32, 32, 32), 0.05), profile)[0]
    vols = {s: Volume.open(f"subjects/{s}.zarr") for s in ("a", "b")}
    seed_level_zero(cfg, grid, vols)

    ex = get_extractor(profile.features, seed=cfg.seed)
    seen = []
    real = ex.__call__

    class Spy:
        name, channels, r_f = ex.name, ex.channels, ex.r_f

        def setup(self, device=None):
            ex.setup(device)

        def select(self, level, iteration, seed=None):
            return ex.select(level, iteration, seed)

        def __call__(self, p, spacing, mask=None):
            seen.append(tuple(ex.pick))
            return real(p, spacing, mask)

    for iteration in (0, 3):
        manifest = build_manifest(cfg, 0, iteration, grid)
        run_register_task(cfg, manifest, 0, extractor=Spy())
        want = tuple(stratified_pick(ex.groups, 4, cfg.seed, 0, iteration))
        assert seen and all(s == want for s in seen), (iteration, seen, want)
        seen.clear()


# --------------------------------------------------------------------------- #
# The device path
# --------------------------------------------------------------------------- #
def _anatomix_mind_module():
    return pytest.importorskip("anatomix.registration.registration_infrastructure.mindssc")


def test_the_device_path_uses_anatomix_mind(monkeypatch):
    torch = pytest.importorskip("torch")
    mod = _anatomix_mind_module()
    calls = []
    real = mod.MINDSSC
    monkeypatch.setattr(mod, "MINDSSC", lambda *a, **k: calls.append(a) or real(*a, **k))
    p = torch.from_numpy(patch(3))
    out = MindSSCFeatures(normalisation="none")(p, 0.05)
    assert calls, "anatomix's MINDSSC was not used"
    assert isinstance(out, torch.Tensor) and out.shape == (12,) + tuple(p.shape)
    np.testing.assert_allclose(out.numpy(), real(p[None, None], 1, 2)[0].numpy(), rtol=1e-6)


def test_the_fallback_port_matches_the_numpy_descriptor():
    torch = pytest.importorskip("torch")
    p = patch(4)
    want = MindSSCFeatures(normalisation="none")(p, 0.05)
    got = MindSSCFeatures(normalisation="none")._port(torch.from_numpy(p))
    np.testing.assert_allclose(got.numpy(), want, rtol=1e-4, atol=1e-5)


def test_sampling_and_combining_stay_on_the_device():
    torch = pytest.importorskip("torch")
    ex = get_extractor("intensity+mindssc@5", seed=0)
    ex.select(1, 2)
    p = torch.from_numpy(patch(5))
    out = ex(p, 0.05)
    assert isinstance(out, torch.Tensor) and out.shape == (5,) + tuple(p.shape)
    full = get_extractor("intensity+mindssc")(p, 0.05)
    assert isinstance(full, torch.Tensor)
    np.testing.assert_allclose(out.numpy(), full[torch.as_tensor(ex.pick)].numpy(), rtol=1e-6)

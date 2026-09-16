"""Grid, profile, pyramid and tiling invariants."""

from __future__ import annotations

import numpy as np
import pytest

from chunkreg.grid import (
    Chunk,
    GridSpec,
    Profile,
    chunks_touching,
    pyramid,
    pyramid_depth,
    shard_grid,
    tile,
    window,
)

HIPCT = GridSpec(shape=(1920, 2560, 2560), spacing_mm=0.05)


# --------------------------------------------------------------------------- #
# Profile: the halo budget
# --------------------------------------------------------------------------- #
def test_default_profile_halo_budget():
    """48 halo = 12 clamp + 12 kernel/smoothing at s_max 2 + 24 receptive field."""
    p = Profile()
    assert p.padded == 352
    assert p.s_max == 2
    assert p.support_vox() == pytest.approx(12.0)
    assert p.d_max_vox() == pytest.approx(12.0)
    assert p.overhead == pytest.approx((352 / 256) ** 3, rel=1e-9)


def test_clamp_scales_with_level_spacing():
    p = Profile()
    assert p.d_max_mm(0.4) == pytest.approx(4.8)
    assert p.d_max_mm(0.1) == pytest.approx(1.2)
    assert p.d_max_mm(0.05) == pytest.approx(0.6)


def test_four_scale_pyramid_would_exhaust_the_halo():
    """scales [4,2,1] leaves nothing for displacement at halo 48."""
    p = Profile(scales=(4, 2, 1))
    assert p.support_vox() == pytest.approx(24.0)
    assert p.d_max_vox() == pytest.approx(0.0)
    # A bigger halo buys it back.
    assert Profile(halo=64, scales=(4, 2, 1)).d_max_vox() == pytest.approx(16.0)


def test_profile_memory_by_channel_count():
    gb = {c: Profile(channels=c).mem_gb() for c in (16, 8, 4, 1)}
    assert gb[16] == pytest.approx(90 * 16 * 352**3 / 1024**3, rel=1e-9)
    assert gb[16] / gb[8] == pytest.approx(2.0)
    assert gb[1] < 5.0


def test_profile_rejects_misaligned_geometry():
    with pytest.raises(ValueError, match="inner_chunk"):
        Profile(core=100, inner_chunk=64)
    with pytest.raises(ValueError, match="coarse to fine"):
        Profile(scales=(1, 2))
    with pytest.raises(ValueError, match="odd"):
        Profile(k=8)


# --------------------------------------------------------------------------- #
# Pyramid
# --------------------------------------------------------------------------- #
def test_pyramid_matches_the_worked_example():
    p = Profile()
    levels = pyramid(HIPCT, p)
    assert pyramid_depth(HIPCT.shape, p.core) == 4
    assert len(levels) == 5
    assert [g.shape for g in levels] == [
        (120, 160, 160),
        (240, 320, 320),
        (480, 640, 640),
        (960, 1280, 1280),
        (1920, 2560, 2560),
    ]
    assert [round(g.spacing_mm, 6) for g in levels] == [0.8, 0.4, 0.2, 0.1, 0.05]


def test_chunk_counts_match_the_worked_example():
    p = Profile()
    counts = [len(tile(g, p)) for g in pyramid(HIPCT, p)]
    assert counts == [1, 4, 18, 100, 800]


def test_coarsest_level_is_always_one_chunk():
    p = Profile()
    for shape in [(1920, 2560, 2560), (256, 256, 256), (8000, 8000, 8000), (3, 517, 41)]:
        g = GridSpec(shape, 0.05)
        assert len(tile(pyramid(g, p)[0], p)) == 1


def test_small_volume_needs_no_pyramid():
    """A volume that already fits one chunk is a single-level run."""
    g = GridSpec((256, 256, 256), 1.0)
    levels = pyramid(g, Profile())
    assert len(levels) == 1
    assert levels[0] == g


def test_finest_level_is_the_native_grid():
    g = GridSpec((1920, 2560, 2560), 0.05, origin_mm=(1.0, -2.0, 3.5))
    assert pyramid(g, Profile())[-1] == g


def test_levels_share_a_world_frame():
    """Mean pooling shifts the origin by half the lost extent; centres agree."""
    g = GridSpec((512, 512, 512), 0.05, origin_mm=(1.0, -2.0, 3.5))
    levels = pyramid(g, Profile())
    centres = [
        np.asarray(lv.world(np.asarray(lv.shape, float) / 2.0 - 0.5)) for lv in levels
    ]
    for c in centres[1:]:
        assert np.allclose(c, centres[0], atol=1e-6)


# --------------------------------------------------------------------------- #
# Tiling
# --------------------------------------------------------------------------- #
def test_cores_tile_exactly_once():
    g = GridSpec((300, 260, 700), 0.1)
    p = Profile()
    seen = np.zeros(g.shape, dtype=np.int32)
    for c in tile(g, p):
        seen[c.core_slices()] += 1
    assert np.all(seen == 1)


def test_pads_stay_inside_the_volume_and_contain_their_core():
    g = GridSpec((300, 260, 700), 0.1)
    for c in tile(g, Profile()):
        for d in range(3):
            assert c.pad_origin[d] >= 0
            assert c.pad_origin[d] + c.pad_shape[d] <= g.shape[d]
            assert c.pad_origin[d] <= c.core_origin[d]
            assert (
                c.pad_origin[d] + c.pad_shape[d]
                >= c.core_origin[d] + c.core_shape[d]
            )


def test_shard_id_is_the_chunk_index_and_matches_the_shard_grid():
    g = GridSpec((480, 640, 640), 0.2)
    p = Profile()
    chunks = tile(g, p)
    assert shard_grid(g, p) == (2, 3, 3)
    assert {c.shard_id for c in chunks} == {
        (i, j, k) for i in range(2) for j in range(3) for k in range(3)
    }


def test_blend_read_set_is_the_owner_plus_at_most_26_neighbours():
    """Halo < core, so a shard is only ever reached by immediate neighbours."""
    g = GridSpec((1024, 1024, 1024), 0.1)
    p = Profile()
    chunks = tile(g, p)
    interior = next(c for c in chunks if c.index == (1, 1, 1))
    touching = chunks_touching(chunks, interior.core_origin, interior.core_shape)
    assert interior in touching
    assert len(touching) == 27
    for c in touching:
        assert max(abs(a - b) for a, b in zip(c.index, interior.index)) <= 1


def test_chunk_intersects_is_exact_on_a_disjoint_box():
    g = GridSpec((1024, 1024, 1024), 0.1)
    chunks = tile(g, Profile())
    far = next(c for c in chunks if c.index == (0, 0, 0))
    assert not far.intersects((768, 768, 768), (256, 256, 256))
    assert far.intersects((0, 0, 0), (8, 8, 8))


# --------------------------------------------------------------------------- #
# Blend windows
# --------------------------------------------------------------------------- #
def test_window_is_one_on_the_core_and_zero_past_the_taper():
    g = GridSpec((1024, 1024, 1024), 0.1)
    p = Profile()
    c = next(x for x in tile(g, p) if x.index == (1, 1, 1))
    w = window(c, g, p)
    assert w.shape == c.pad_shape
    off = c.core_offset_in_pad
    core = tuple(slice(o, o + s) for o, s in zip(off, c.core_shape))
    assert np.allclose(w[core], 1.0)
    assert w.min() >= 0.0
    assert w.max() == pytest.approx(1.0)


def test_window_sum_is_at_least_one_everywhere():
    """Cores tile exactly, so the blend denominator never approaches zero."""
    g = GridSpec((600, 300, 300), 0.1)
    p = Profile()
    den = np.zeros(g.shape, dtype=np.float64)
    for c in tile(g, p):
        den[c.pad_slices()] += window(c, g, p)
    assert den.min() >= 1.0 - 1e-6
    assert np.isfinite(den).all()


def test_window_does_not_taper_at_a_volume_boundary():
    """Nothing lies outside to blend with, so the edge keeps full weight."""
    g = GridSpec((512, 512, 512), 0.1)
    p = Profile()
    c = next(x for x in tile(g, p) if x.index == (0, 0, 0))
    w = window(c, g, p)
    assert w[0, 0, 0] == pytest.approx(1.0)


def test_window_is_smooth_across_an_interior_boundary():
    """The taper rises monotonically from ~0 at the padded edge to 1 at the core.

    Half-sample offsets in the ramp keep the first weight just above zero, so
    no sample is discarded outright and the ramp has no kink at its ends.
    """
    g = GridSpec((1024, 1024, 1024), 0.1)
    p = Profile()
    c = next(x for x in tile(g, p) if x.index == (1, 1, 1))
    w = window(c, g, p)
    line = w[:, w.shape[1] // 2, w.shape[2] // 2]
    rising = line[: c.core_offset_in_pad[0] + 1]
    assert np.all(np.diff(rising) >= -1e-7)
    assert 0.0 < line[0] < 1e-3
    assert rising[-1] == pytest.approx(1.0)

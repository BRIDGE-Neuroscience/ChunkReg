"""The CPU reference engine must actually recover deformations.

If this engine cannot invert a known warp, the end-to-end pipeline tests that
depend on it prove nothing.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from chunkreg import fields
from chunkreg.config import StageSpec
from chunkreg.engines import get_engine
from chunkreg.engines.base import StageNotSupported

from .conftest import blob


def stack(v: np.ndarray) -> np.ndarray:
    return v[None].astype(np.float32)


def smooth_warp(shape, amp_mm, sigma=6.0, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(0, 1, (3,) + tuple(shape)).astype(np.float32)
    u = np.stack([ndimage.gaussian_filter(raw[d], sigma) for d in range(3)])
    u *= amp_mm / max(np.abs(u).max(), 1e-8)
    return u.astype(np.float32)


GREEDY = StageSpec(kind="greedy", scales=(4, 2, 1), iterations=(60, 40, 25))


def test_engine_registry_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown engine"):
        get_engine("magic")


def test_demons_declares_what_it_cannot_do():
    """A reference engine must refuse an affine stage, not silently skip it."""
    e = get_engine("demons")
    v = stack(blob((24, 24, 24)))
    with pytest.raises(StageNotSupported, match="affine"):
        e.register(v, v, 0.05, [StageSpec(kind="affine", scales=(1,), iterations=(2,))])


def test_identical_images_produce_a_near_zero_field():
    e = get_engine("demons")
    v = stack(blob((32, 32, 32), seed=1))
    res = e.register(v, v, 0.05, [GREEDY])
    assert fields.percentile_magnitude(res.disp_mm, 99.9) < 0.02


def test_recovers_a_known_translation():
    e = get_engine("demons")
    spacing, shift = 0.05, 3
    v = blob((40, 40, 40), seed=2)
    moving = np.roll(v, shift, axis=0)
    res = e.register(stack(v), stack(moving), spacing, [GREEDY])
    inner = (slice(None),) + tuple(slice(10, -10) for _ in range(3))
    median = np.median(res.disp_mm[inner].reshape(3, -1), axis=1)
    assert median[0] == pytest.approx(shift * spacing, abs=0.3 * spacing)
    assert abs(median[1]) < 0.3 * spacing
    assert abs(median[2]) < 0.3 * spacing


def test_recovers_a_smooth_deformation():
    """Registering a warped copy recovers the warp's *inverse*.

    The subject is ``moving(y) = v(y + t(y))``. A recovered ``u`` satisfies
    ``moving(x + u(x)) = v(x)``, which expands to ``u(x) + t(x + u(x)) = 0``.
    That expression is exactly ``compose(inner=u, outer=t)``, so composing the
    recovered field with the truth must land on zero. Comparing ``u`` to ``t``
    directly would be wrong: to first order they differ by a sign.
    """
    e = get_engine("demons")
    spacing = 0.05
    shape = (48, 48, 48)
    v = blob(shape, seed=3, smooth=2.5)
    truth = smooth_warp(shape, amp_mm=8 * spacing, seed=4)
    moving = fields.warp(v, truth, spacing)

    res = e.register(stack(v), stack(moving), spacing, [GREEDY])
    leftover = fields.compose(res.disp_mm, truth, spacing)

    inner = (slice(None),) + tuple(slice(12, -12) for _ in range(3))
    before = np.abs(truth[inner]).mean()
    after = np.abs(leftover[inner]).mean()
    assert after < 0.35 * before, f"leftover {after:.4f} vs initial {before:.4f}"


def test_recovered_field_is_the_inverse_not_the_original():
    """Guards the direction convention: u is close to -t, not to +t."""
    e = get_engine("demons")
    spacing = 0.05
    shape = (48, 48, 48)
    v = blob(shape, seed=13, smooth=2.5)
    truth = smooth_warp(shape, amp_mm=8 * spacing, seed=14)
    moving = fields.warp(v, truth, spacing)

    u = e.register(stack(v), stack(moving), spacing, [GREEDY]).disp_mm
    inner = (slice(None),) + tuple(slice(12, -12) for _ in range(3))
    to_negated = np.abs(u[inner] + truth[inner]).mean()
    to_original = np.abs(u[inner] - truth[inner]).mean()
    assert to_negated < 0.5 * to_original


def test_registration_reduces_the_residual_image_difference():
    e = get_engine("demons")
    spacing = 0.05
    shape = (48, 48, 48)
    v = blob(shape, seed=5, smooth=2.5)
    truth = smooth_warp(shape, amp_mm=4 * spacing, seed=6)
    moving = fields.warp(v, truth, spacing)

    res = e.register(stack(v), stack(moving), spacing, [GREEDY])
    warped = fields.warp(moving, res.disp_mm, spacing)
    box = tuple(slice(12, -12) for _ in range(3))
    before = np.abs(v[box] - moving[box]).mean()
    after = np.abs(v[box] - warped[box]).mean()
    assert after < 0.6 * before


def test_moments_stage_removes_a_gross_offset():
    e = get_engine("demons")
    spacing = 0.05
    shape = (40, 40, 40)
    v = np.zeros(shape, np.float32)
    v[12:20, 16:24, 16:24] = 1.0
    moving = np.roll(v, 6, axis=0)
    res = e.register(
        stack(v), stack(moving), spacing, [StageSpec(kind="moments", scales=(), iterations=())]
    )
    assert res.disp_mm[0].mean() == pytest.approx(6 * spacing, abs=spacing)


def test_result_reports_achieved_iterations_per_scale():
    """Cap tuning needs the achieved histogram, so the engine must report it."""
    e = get_engine("demons")
    v = stack(blob((32, 32, 32), seed=7))
    res = e.register(v, v, 0.05, [GREEDY])
    assert len(res.iters_per_scale) == len(GREEDY.scales)
    assert all(1 <= n <= cap for n, cap in zip(res.iters_per_scale, GREEDY.iterations))
    assert res.total_iters == sum(res.iters_per_scale)


def test_identical_images_early_stop_well_inside_the_cap():
    e = get_engine("demons")
    v = stack(blob((32, 32, 32), seed=8))
    res = e.register(v, v, 0.05, [GREEDY])
    assert res.iters_per_scale[-1] < GREEDY.iterations[-1]
    assert res.converged


def test_engine_rejects_mismatched_inputs():
    e = get_engine("demons")
    a = stack(blob((16, 16, 16)))
    b = stack(blob((16, 16, 20)))
    with pytest.raises(ValueError, match="share a grid"):
        e.register(a, b, 0.05, [GREEDY])


def test_multichannel_input_is_accepted():
    """Feature stacks are the normal case; intensity is the C=1 degenerate one."""
    e = get_engine("demons")
    spacing = 0.05
    v = blob((32, 32, 32), seed=9)
    chans = np.stack([v, ndimage.sobel(v, axis=0)]).astype(np.float32)
    moving = np.stack([np.roll(c, 2, axis=0) for c in chans]).astype(np.float32)
    res = e.register(chans, moving, spacing, [GREEDY])
    inner = (slice(None),) + tuple(slice(8, -8) for _ in range(3))
    assert np.median(res.disp_mm[inner].reshape(3, -1), axis=1)[0] > 0.5 * 2 * spacing


def test_output_is_deterministic():
    e = get_engine("demons")
    v, m = stack(blob((32, 32, 32), seed=10)), stack(blob((32, 32, 32), seed=11))
    a = e.register(v, m, 0.05, [GREEDY]).disp_mm
    b = e.register(v, m, 0.05, [GREEDY]).disp_mm
    assert np.array_equal(a, b)

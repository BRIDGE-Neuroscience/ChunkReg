"""Displacement-field conventions: direction, component order, composition.

These are the tests that stop a sign or transposition error from surviving into
a cluster run, where it would cost thousands of GPU-hours before anyone noticed.
"""

from __future__ import annotations

import numpy as np
import pytest

from chunkreg import fields
from chunkreg.grid import GridSpec


def smooth_blob(shape=(48, 40, 44), seed=0) -> np.ndarray:
    """Band-limited noise: enough structure to register, no aliasing."""
    from scipy import ndimage

    rng = np.random.default_rng(seed)
    v = rng.random(shape).astype(np.float32)
    return ndimage.gaussian_filter(v, 2.0).astype(np.float32)


def constant_field(shape, vector_mm) -> np.ndarray:
    u = np.zeros((3,) + tuple(shape), dtype=np.float32)
    for d in range(3):
        u[d] = vector_mm[d]
    return u


# --------------------------------------------------------------------------- #
# Direction and component order
# --------------------------------------------------------------------------- #
def test_identity_field_leaves_an_image_alone():
    img = smooth_blob()
    out = fields.warp(img, np.zeros((3,) + img.shape, np.float32), 0.05)
    assert np.allclose(out, img, atol=1e-6)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_component_d_displaces_along_array_axis_d(axis):
    """Component order is (z, y, x), matching array axes, not grid_sample."""
    img = smooth_blob()
    spacing, shift = 0.05, 4
    u = np.zeros((3,) + img.shape, np.float32)
    u[axis] = shift * spacing
    out = fields.warp(img, u, spacing)
    # phi(x) = x + u, so the output samples the image *ahead* along `axis`:
    # out[i] = img[i + shift], i.e. the content moves toward lower indices.
    expected = np.roll(img, -shift, axis=axis)
    inner = tuple(slice(8, -8) for _ in range(3))
    assert np.allclose(out[inner], expected[inner], atol=1e-5)


def test_warp_samples_zero_outside_the_volume():
    """A subject that does not cover the template contributes nothing."""
    img = np.ones((16, 16, 16), np.float32)
    u = constant_field(img.shape, (100.0, 0.0, 0.0))
    assert np.allclose(fields.warp(img, u, 1.0), 0.0)


def test_warp_recovers_a_subvoxel_shift():
    img = smooth_blob()
    spacing = 0.05
    u = constant_field(img.shape, (0.5 * spacing, 0.0, 0.0))
    out = fields.warp(img, u, spacing)
    inner = tuple(slice(6, -6) for _ in range(3))
    midpoint = 0.5 * (img[6:-6] + np.roll(img, -1, axis=0)[6:-6])
    assert np.allclose(out[inner], midpoint[:, 6:-6, 6:-6], atol=1e-5)


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #
def test_compose_with_zero_is_the_identity_on_both_sides():
    u = np.random.default_rng(1).normal(0, 0.02, (3, 20, 22, 24)).astype(np.float32)
    zero = np.zeros_like(u)
    assert np.allclose(fields.compose(zero, u, 0.05), u, atol=1e-5)
    assert np.allclose(fields.compose(u, zero, 0.05), u, atol=1e-5)


def test_compose_adds_constant_displacements():
    shape = (24, 24, 24)
    a = constant_field(shape, (0.10, -0.05, 0.0))
    b = constant_field(shape, (0.05, 0.05, -0.10))
    out = fields.compose(a, b, 0.05)
    assert np.allclose(out, a + b, atol=1e-5)


def test_compose_matches_sequential_warping():
    """compose(inner, outer) must warp an image exactly as applying both does."""
    img = smooth_blob((40, 40, 40), seed=3)
    spacing = 0.05
    rng = np.random.default_rng(7)
    from scipy import ndimage

    def smooth_field(scale):
        raw = rng.normal(0, scale, (3,) + img.shape).astype(np.float32)
        return np.stack([ndimage.gaussian_filter(raw[d], 3.0) for d in range(3)])

    inner = smooth_field(0.05)
    outer = smooth_field(0.05)
    total = fields.compose(inner, outer, spacing)

    once = fields.warp(fields.warp(img, outer, spacing), inner, spacing)
    at_once = fields.warp(img, total, spacing)
    inner_box = tuple(slice(10, -10) for _ in range(3))
    assert np.allclose(once[inner_box], at_once[inner_box], atol=2e-3)


def test_seeded_composition_recovers_the_true_displacement():
    """The seeding identity: compose(inner=residual, outer=seed) = truth."""
    shape = (32, 32, 32)
    spacing = 0.05
    truth = constant_field(shape, (0.30, 0.0, 0.0))
    seed = constant_field(shape, (0.25, 0.0, 0.0))
    residual = constant_field(shape, (0.05, 0.0, 0.0))
    assert np.allclose(fields.compose(residual, seed, spacing), truth, atol=1e-5)


# --------------------------------------------------------------------------- #
# Clamping
# --------------------------------------------------------------------------- #
def test_clamp_limits_magnitude_and_keeps_direction():
    u = np.zeros((3, 4, 4, 4), np.float32)
    u[:, 0, 0, 0] = (3.0, 4.0, 0.0)  # magnitude 5
    out = fields.clamp(u, 1.0)
    assert fields.magnitude(out)[0, 0, 0] == pytest.approx(1.0, rel=1e-5)
    assert np.allclose(out[:, 0, 0, 0], np.array([0.6, 0.8, 0.0]), atol=1e-5)


def test_clamp_leaves_short_vectors_untouched():
    u = np.zeros((3, 4, 4, 4), np.float32)
    u[:, 1, 1, 1] = (0.1, 0.0, 0.0)
    assert np.allclose(fields.clamp(u, 1.0), u)


def test_clamp_is_disabled_by_none_or_zero():
    u = np.full((3, 2, 2, 2), 9.0, np.float32)
    assert np.allclose(fields.clamp(u, None), u)
    assert np.allclose(fields.clamp(u, 0.0), u)


# --------------------------------------------------------------------------- #
# Resampling between grids and lattices
# --------------------------------------------------------------------------- #
def test_resampling_a_field_preserves_the_vectors():
    """Millimetre storage is what makes promoting a seed exact."""
    src = GridSpec((16, 16, 16), 0.2)
    dst = GridSpec((32, 32, 32), 0.1, origin_mm=(0.05, 0.05, 0.05))
    u = constant_field(src.shape, (0.37, -0.11, 0.02))
    out = fields.resample_field(u, src, dst)
    assert out.shape == (3,) + dst.shape
    assert np.allclose(out, constant_field(dst.shape, (0.37, -0.11, 0.02)), atol=1e-5)


def test_field_survives_a_lattice_round_trip():
    """Store on a half-resolution lattice, read back dense."""
    from scipy import ndimage

    level = GridSpec((32, 32, 32), 0.05)
    lattice = level.coarsened(2)
    rng = np.random.default_rng(11)
    raw = rng.normal(0, 0.05, (3,) + level.shape).astype(np.float32)
    u = np.stack([ndimage.gaussian_filter(raw[d], 4.0) for d in range(3)])
    stored = fields.resample_field(u, level, lattice)
    back = fields.resample_field(stored, lattice, level)
    inner = (slice(None),) + tuple(slice(4, -4) for _ in range(3))
    assert np.abs(back[inner] - u[inner]).max() < 2e-3


def test_resample_scalar_preserves_a_constant():
    src = GridSpec((8, 8, 8), 0.4)
    dst = GridSpec((16, 16, 16), 0.2, origin_mm=(0.1, 0.1, 0.1))
    img = np.full(src.shape, 2.5, np.float32)
    assert np.allclose(fields.resample_scalar(img, src, dst), 2.5, atol=1e-5)


# --------------------------------------------------------------------------- #
# Jacobian and folds
# --------------------------------------------------------------------------- #
def test_identity_has_unit_jacobian_and_no_folds():
    u = np.zeros((3, 16, 16, 16), np.float32)
    det = fields.jacobian_determinant(u, 0.05)
    assert np.allclose(det, 1.0, atol=1e-5)
    assert fields.fold_fraction(u, 0.05) == 0.0


def test_uniform_expansion_has_the_expected_jacobian():
    """u = a*x scales each axis by (1 + a), so det = (1 + a)^3."""
    shape = (16, 16, 16)
    spacing, a = 0.05, 0.1
    ident = fields.identity(shape) * spacing
    u = (a * ident).astype(np.float32)
    det = fields.jacobian_determinant(u, spacing)
    inner = tuple(slice(2, -2) for _ in range(3))
    assert np.allclose(det[inner], (1 + a) ** 3, rtol=1e-4)


def test_a_strong_inward_field_folds():
    shape = (16, 16, 16)
    spacing = 0.05
    ident = fields.identity(shape) * spacing
    u = (-2.0 * ident).astype(np.float32)  # det = (1 - 2)^3 = -1
    assert fields.fold_fraction(u, spacing) > 0.5


def test_percentile_magnitude_reports_millimetres():
    u = np.zeros((3, 10, 10, 10), np.float32)
    u[0] = 0.4
    assert fields.percentile_magnitude(u, 99.9) == pytest.approx(0.4, rel=1e-5)


# --------------------------------------------------------------------------- #
# The engine bridge
# --------------------------------------------------------------------------- #
def identity_sampling_grid(shape):
    """What an engine returns for a null transform: normalised, (x, y, z) last."""
    ident = fields.identity(shape)
    g = np.empty(tuple(shape) + (3,), np.float32)
    for d in range(3):
        n = max(shape[d] - 1, 1)
        g[..., 2 - d] = ident[d] / n * 2.0 - 1.0
    return g[None]


def test_identity_sampling_grid_maps_to_zero_displacement():
    shape = (12, 14, 16)
    u = fields.grid_to_disp_mm(identity_sampling_grid(shape), shape, 0.05)
    assert u.shape == (3,) + shape
    assert np.allclose(u, 0.0, atol=1e-4)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_bridge_undoes_the_reversed_component_order(axis):
    """A grid shifted along its (x,y,z) slot must land on the right array axis."""
    shape = (12, 14, 16)
    spacing, shift = 0.05, 2.0
    g = identity_sampling_grid(shape)
    n = max(shape[axis] - 1, 1)
    g[..., 2 - axis] += shift / n * 2.0  # move `shift` voxels along array axis
    u = fields.grid_to_disp_mm(g, shape, spacing)
    for d in range(3):
        expected = shift * spacing if d == axis else 0.0
        assert np.allclose(u[d], expected, atol=1e-4), f"component {d}"


def test_bridge_round_trips():
    rng = np.random.default_rng(5)
    shape = (10, 12, 14)
    spacing = 0.05
    u = rng.normal(0, 0.02, (3,) + shape).astype(np.float32)
    back = fields.grid_to_disp_mm(fields.disp_mm_to_grid(u, spacing), shape, spacing)
    assert np.allclose(back, u, atol=1e-5)


def test_bridge_accepts_unbatched_grids_and_rejects_bad_shapes():
    shape = (8, 8, 8)
    g = identity_sampling_grid(shape)
    assert fields.grid_to_disp_mm(g[0], shape, 0.05).shape == (3,) + shape
    with pytest.raises(ValueError, match="does not match"):
        fields.grid_to_disp_mm(g, (9, 8, 8), 0.05)

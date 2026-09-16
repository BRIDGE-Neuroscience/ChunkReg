"""The PyTorch path must reproduce the NumPy reference, operation by operation.

Runs on PyTorch's CPU device so it needs no GPU; the kernels are the same ones
a CUDA device runs. Skipped where PyTorch is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

torch = pytest.importorskip("torch")

from chunkreg import fields as F  # noqa: E402
from chunkreg import gpu_ops as G  # noqa: E402
from chunkreg import stats, xp  # noqa: E402
from chunkreg.grid import GridSpec  # noqa: E402

RNG = np.random.default_rng(0)
SHAPE = (17, 12, 14)
SP = 0.05
TOL = 2e-5


def t(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def close(got, want, tol=TOL):
    g = got.numpy() if isinstance(got, torch.Tensor) else np.asarray(got)
    np.testing.assert_allclose(g, want, rtol=0, atol=tol)


@pytest.fixture
def img():
    return RNG.random(SHAPE, dtype=np.float32)


@pytest.fixture
def field():
    # Large enough that samples leave the volume on every face.
    return ((RNG.random((3,) + SHAPE, dtype=np.float32) - 0.5) * 6 * SP).astype(np.float32)


def test_sampling_matches_map_coordinates_inside_and_out():
    a = RNG.random((5, 6, 7), dtype=np.float32)
    coords = RNG.uniform(-1.5, 8.5, size=(3, 9, 9, 9)).astype(np.float32)
    coords[:, 0, 0, 0] = [0, 0, 0]
    coords[:, 0, 0, 1] = [4, 5, 6]          # exactly on the far corner
    coords[:, 0, 0, 2] = [4 + 1e-4, 5, 6]   # just past it
    for mode in ("constant", "nearest"):
        want = ndimage.map_coordinates(a, coords, order=1, mode=mode, cval=0.0, prefilter=False)
        close(G.sample(t(a), t(coords), mode), want)


def test_warps_match(img, field):
    close(F.warp(t(img), t(field), SP), F.warp(img, field, SP))
    stack = np.stack([img, img[::-1].copy(), img ** 2])
    close(F.warp_stack(t(stack), t(field), SP), F.warp_stack(stack, field, SP))
    big = RNG.random(tuple(n + 6 for n in SHAPE), dtype=np.float32)
    close(
        F.warp_offset(t(big), (3, 3, 3), t(field), SP),
        F.warp_offset(big, (3, 3, 3), field, SP),
    )


def test_compose_and_clamp_match(field):
    other = field[:, ::-1, :, :].copy()
    close(F.compose(t(field), t(other), SP), F.compose(field, other, SP))
    close(F.clamp(t(field), 0.05), F.clamp(field, 0.05))
    close(F.clamp(t(field), None), F.clamp(field, None))


def test_resampling_matches(img, field):
    src = GridSpec(SHAPE, 0.1, (0.3, -0.2, 0.05))
    dst = GridSpec((30, 20, 25), 0.05, (0.2, -0.25, 0.0))
    close(F.resample_scalar(t(img), src, dst), F.resample_scalar(img, src, dst))
    close(F.resample_field(t(field), src, dst), F.resample_field(field, src, dst))


def test_jacobian_and_folds_match(field):
    close(F.jacobian_determinant(t(field), SP), F.jacobian_determinant(field, SP), 1e-4)
    assert F.fold_fraction(t(field), SP) == pytest.approx(F.fold_fraction(field, SP))
    thin = field[:, :2].copy()  # two samples along an axis: one-sided only
    close(F.jacobian_determinant(t(thin), SP), F.jacobian_determinant(thin, SP), 1e-4)


def test_magnitude_percentile_margin_match(field):
    close(F.magnitude(t(field)), F.magnitude(field))
    for q in (0, 37.5, 99, 99.9, 100):
        assert F.percentile_magnitude(t(field), q) == pytest.approx(
            F.percentile_magnitude(field, q), abs=1e-6
        )
    assert F.required_margin(t(field), SP) == F.required_margin(field, SP)


def test_histograms_are_identical(field):
    mag = F.magnitude(field)
    assert np.array_equal(stats.histogram(t(mag)), stats.histogram(mag))


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0, 3.7])
def test_gaussian_matches_scipy(img, field, sigma):
    close(F.smooth(t(img), sigma), F.smooth(img, sigma))
    close(F.smooth(t(field), sigma), F.smooth(field, sigma))


def test_anisotropic_constant_gaussian_matches_scipy(img):
    want = ndimage.gaussian_filter(img, (0.0, 1.5, 0.7), mode="constant")
    close(G.gaussian(t(img), (0.0, 1.5, 0.7), mode="constant"), want)


@pytest.mark.parametrize("factor", [1, 2, 3])
def test_pooling_matches(field, factor):
    close(F.pool_field(t(field), factor), F.pool_field(field, factor))
    target = (7, 5, 6)
    close(
        F.pool_field(t(field), factor, target), F.pool_field(field, factor, target)
    )


def test_upsample_box_matches_repeat():
    from chunkreg.passes.blend import _upsample_box

    lat = RNG.random((3, 9, 8, 7), dtype=np.float32)
    box = (slice(3, 15), slice(1, 16), slice(0, 13))
    close(G.upsample_box(t(lat), 2, box), _upsample_box(lat, 2, box))


@pytest.mark.parametrize("size", [2, 3, 4])
def test_median_matches_scipy(size):
    a = (RNG.random((11, 9, 10)) * 1000).astype(np.float32)
    close(G.median_filter(t(a), size, slab=3), ndimage.median_filter(a, size=size))


def test_engine_bridge_round_trips(field):
    grid = F.disp_mm_to_grid(field, SP)
    close(F.disp_mm_to_grid(t(field), SP), grid)
    close(F.grid_to_disp_mm(t(grid), SHAPE, SP), F.grid_to_disp_mm(grid, SHAPE, SP), 1e-4)


def test_channel_normalisation_matches():
    from chunkreg.features.base import normalise_channels

    x = RNG.normal(size=(4,) + SHAPE).astype(np.float32)
    for method in ("l2", "standardized", "none"):
        close(G.normalise_channels(t(x), method), normalise_channels(x, method), 1e-4)


def test_tissue_fraction_matches(img):
    from chunkreg.passes._common import tissue_fraction

    assert tissue_fraction(t(img)) == pytest.approx(tissue_fraction(img))


# --------------------------------------------------------------------------- #
# The device setting
# --------------------------------------------------------------------------- #
def test_put_and_get_follow_the_device():
    a = np.arange(6, dtype=np.uint16).reshape(1, 2, 3)
    with xp.using("cpu"):
        assert isinstance(xp.put(a), np.ndarray)
    with xp.using("torch-cpu"):
        v = xp.put(a)
        assert isinstance(v, torch.Tensor) and v.dtype == torch.float32
        assert np.array_equal(xp.get(v), a.astype(np.float32))
    assert xp.device_name() == "cpu"


def test_cuda_without_a_gpu_refuses_to_fall_back():
    if torch.cuda.is_available():
        pytest.skip("this machine has a GPU")
    # honour_env=False: the suite pins CHUNKREG_DEVICE=cpu, which would
    # otherwise replace the device under test.
    with pytest.raises(RuntimeError, match="Refusing to fall back"):
        xp.resolve("cuda", honour_env=False)
    assert xp.resolve("auto", honour_env=False) == "cpu"


def test_unknown_devices_are_refused():
    with pytest.raises(ValueError, match="unknown device"):
        xp.resolve("gpu", honour_env=False)


def test_the_environment_overrides_the_config(monkeypatch):
    monkeypatch.setenv("CHUNKREG_DEVICE", "torch-cpu")
    assert xp.resolve("cpu") == "torch-cpu"
    assert xp.resolve("cpu", honour_env=False) == "cpu"

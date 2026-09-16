"""The convention selftest.

Every displacement in this package is a signed, scaled, axis-ordered quantity
passing between numpy, an engine's normalised sampling grid and a half
precision store. A sign flip or a transposition in any of those produces a run
that looks healthy, costs thousands of GPU-hours and is wrong.

This test drives a known integer shift through the whole path, both whole-chunk
and tiled, and checks the recovered displacement lands on the right axis with
the right sign and magnitude. It is the first thing to run against a new engine
build.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import fields as _fields
from . import xp
from .config import StageSpec
from .engines import get_engine
from .features import get_extractor

__all__ = ["run_selftest"]

_STAGE = StageSpec(kind="greedy", scales=(4, 2, 1), iterations=(60, 40, 25))


def _blob(shape=(48, 48, 48), seed=0):
    rng = np.random.default_rng(seed)
    if xp.uses_torch():
        v = _fields.smooth(xp.put(rng.random(shape)), 2.0)
        v = v - v.min()
        return v / max(float(v.max()), 1e-8)
    v = ndimage.gaussian_filter(rng.random(shape).astype(np.float32), 2.0)
    v -= v.min()
    return (v / max(v.max(), 1e-8)).astype(np.float32)


def _roll(a, shift: int, axis: int):
    if xp.is_tensor(a):
        import torch

        return torch.roll(a, shifts=shift, dims=axis)
    return np.roll(a, shift, axis=axis)


def _median(a) -> np.ndarray:
    """Per-component median of a ``(3, ...)`` block, as host numbers."""
    if xp.is_tensor(a):
        flat = a.reshape(a.shape[0], -1).double()
        # np.median averages the two middle values of an even count.
        lo = flat.kthvalue((flat.shape[1] + 1) // 2, dim=1).values
        hi = flat.kthvalue(flat.shape[1] // 2 + 1, dim=1).values
        return xp.get((lo + hi) / 2)
    return np.median(np.asarray(a).reshape(a.shape[0], -1), axis=1)


def _check(name, got, want, tol, failures, verbose) -> None:
    ok = abs(got - want) <= tol
    if verbose:
        mark = "ok  " if ok else "FAIL"
        print(f"  {mark} {name}: got {got:+.4f}, want {want:+.4f} (tol {tol:.4f})")
    if not ok:
        failures.append(f"{name}: got {got:+.4f}, want {want:+.4f}")


def run_selftest(
    engine: str = "demons",
    features: str = "intensity",
    spacing_mm: float = 0.05,
    verbose: bool = False,
) -> bool:
    """Drive known shifts through the conventions. True when all pass."""
    eng = get_engine(engine)
    ex = get_extractor(features)
    ex.setup()
    failures: list[str] = []

    if verbose:
        print(f"convention selftest: engine {engine}, features {features}")

    # 1. The engine bridge, with no registration involved.
    shape = (12, 14, 16)
    ident = _fields.disp_mm_to_grid(xp.zeros((3,) + shape), spacing_mm)
    back = _fields.grid_to_disp_mm(ident, shape, spacing_mm)
    _check("bridge identity", float(abs(back).max()), 0.0, 1e-3, failures, verbose)

    for axis in range(3):
        u = xp.zeros((3,) + shape)
        u[axis] = 3 * spacing_mm
        rt = _fields.grid_to_disp_mm(
            _fields.disp_mm_to_grid(u, spacing_mm), shape, spacing_mm
        )
        _check(
            f"bridge round trip axis {axis}",
            float(_median(rt)[axis]),
            3 * spacing_mm,
            1e-4,
            failures,
            verbose,
        )

    # 2. Composition: applying two shifts equals composing them.
    a = xp.zeros((3, 16, 16, 16))
    a[0] = 0.10
    b = xp.zeros_like(a)
    b[1] = -0.05
    comp = _median(_fields.compose(a, b, spacing_mm))
    _check("compose axis 0", float(comp[0]), 0.10, 1e-4, failures, verbose)
    _check("compose axis 1", float(comp[1]), -0.05, 1e-4, failures, verbose)

    # 3. A registration that must recover a known shift on a known axis.
    vol = _blob(seed=1)
    for axis, shift in ((0, 3), (1, -2), (2, 2)):
        moving = _roll(vol, shift, axis)
        res = eng.register(
            ex(vol, spacing_mm), ex(moving, spacing_mm), spacing_mm, [_STAGE]
        )
        inner = (slice(None),) + tuple(slice(12, -12) for _ in range(3))
        median = _median(res.disp_mm[inner])
        for d in range(3):
            want = shift * spacing_mm if d == axis else 0.0
            _check(
                f"recover shift {shift:+d} on axis {axis}, component {d}",
                float(median[d]),
                want,
                0.6 * spacing_mm,
                failures,
                verbose,
            )

    # 4. The same shift solved in tiles and blended must agree with the whole.
    from .grid import GridSpec, Profile, tile, window

    profile = Profile(
        core=24, halo=10, inner_chunk=8, lattice_factor=2, channels=1,
        features="intensity", r_f=0, k=3, sigma_g=0.5, sigma_w=0.5, scales=(2, 1),
    )
    grid = GridSpec(vol.shape, spacing_mm)
    moving = _roll(vol, 2, 0)
    num = xp.zeros((3,) + tuple(vol.shape), np.float64)
    den = xp.zeros(tuple(vol.shape), np.float64)
    for chunk in tile(grid, profile):
        sl = chunk.pad_slices()
        res = eng.register(
            ex(vol[sl], spacing_mm), ex(moving[sl], spacing_mm), spacing_mm, [_STAGE]
        )
        w = xp.put(window(chunk, grid, profile))
        num[(slice(None),) + sl] += res.disp_mm * w
        den[sl] += w
    blended = num / xp.clip(den, 1e-9, None)
    inner = (slice(None),) + tuple(slice(12, -12) for _ in range(3))
    _check(
        "tiled and blended recovers the same shift",
        float(_median(blended[inner])[0]),
        2 * spacing_mm,
        0.6 * spacing_mm,
        failures,
        verbose,
    )
    _check(
        "blend denominator never vanishes",
        float(den.min()),
        1.0,
        0.999,
        failures,
        verbose,
    )

    if verbose:
        print("SELFTEST", "PASSED" if not failures else f"FAILED ({len(failures)})")
        for f in failures:
            print(f"  {f}")
    return not failures

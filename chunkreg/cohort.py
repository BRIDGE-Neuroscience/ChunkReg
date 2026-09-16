"""Putting a cohort of scans onto one run grid.

Every pass reads subject level ``k`` and template level ``k`` with the same
voxel indices, so every subject store has to be on one grid. The scans a cohort
is built from rarely are: they differ in shape, field of view and voxel size,
and some have anisotropic voxels. This module decides the run grid and
resamples each scan onto it at ingest, so everything downstream still sees one
grid.

Resolution
    ``grid.spacing_mm`` is the finest pyramid level: the resolution a run
    registers at when every level runs. ``finest`` (the default) takes the
    finest voxel in the cohort, ``coarsest`` the coarsest, and a number sets it
    outright. A scan finer than the run grid is anti-aliased on the way down;
    a scan coarser than it is interpolated up, which adds voxels but no detail.

Extent
    By default the smallest box that holds every scan. ``grid.shape`` sets it
    outright, and a scan larger than that box is cropped.

Placement
    ``centre`` (the default) centres every scan in the run grid, so level 0's
    moments and rigid stages only have to correct what centring leaves.
    ``corner`` puts every scan's first voxel corner on the grid's corner.
    Either way, a scan whose voxel size divides the run spacing is shifted by
    at most half a run voxel so its voxels line up with the grid's. A scan
    already on the run spacing is then copied rather than interpolated, and a
    scan at half the spacing is pooled exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage

from .grid import GridSpec

__all__ = [
    "Placement",
    "ResampledSource",
    "resolve_run_grid",
    "placement_notes",
    "as_voxel_mm",
]

_REL = 1e-6


def as_voxel_mm(value) -> tuple[float, float, float] | None:
    """A voxel size as ``(z, y, x)`` millimetres, from a number or a triple."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = (float(value),) * 3
    else:
        v = tuple(float(x) for x in value)
        if len(v) != 3:
            raise ValueError(f"a voxel size needs 1 or 3 values, got {list(value)}")
    if not all(x > 0 and math.isfinite(x) for x in v):
        raise ValueError(f"voxel sizes must be positive, got {list(v)}")
    return v


def _is_integer(x: float) -> bool:
    return abs(x - round(x)) <= _REL * max(1.0, abs(x))


# --------------------------------------------------------------------------- #
# Where a scan sits in the run frame
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Placement:
    """A scan's own sampling, placed in the run grid's world frame.

    ``origin_mm`` is the world position of the centre of the scan's voxel
    ``(0, 0, 0)``, so scan voxel ``i`` sits at ``origin_mm + voxel_mm * i``.
    Recorded in each subject store, so a field on the run grid can be taken
    back to the original scan's coordinates.
    """

    shape: tuple[int, int, int]
    voxel_mm: tuple[float, float, float]
    origin_mm: tuple[float, float, float]

    @property
    def extent_mm(self) -> tuple[float, float, float]:
        return tuple(n * v for n, v in zip(self.shape, self.voxel_mm))

    def to_json(self) -> dict:
        return {
            "shape": list(self.shape),
            "voxel_mm": list(self.voxel_mm),
            "origin_mm": list(self.origin_mm),
        }

    @classmethod
    def from_json(cls, d: dict) -> "Placement":
        return cls(
            tuple(int(n) for n in d["shape"]),
            tuple(float(v) for v in d["voxel_mm"]),
            tuple(float(o) for o in d["origin_mm"]),
        )


def resolve_run_grid(
    scans: Mapping[str, tuple[Sequence[int], Sequence[float]]],
    spacing: float | str = "finest",
    shape: Sequence[int] | None = None,
    align: str = "centre",
) -> tuple[GridSpec, dict[str, Placement]]:
    """The run grid for a cohort, and where each scan sits on it.

    ``scans`` maps a subject id to its ``(shape, voxel_mm)``, both ``(z, y,
    x)``. The result is a pure function of its arguments, so the same cohort
    always resolves to the same grid.
    """
    if not scans:
        raise ValueError("a run grid needs at least one scan")
    if align not in ("centre", "corner"):
        raise ValueError(f"grid.align must be 'centre' or 'corner', got {align!r}")
    info = {
        sid: (tuple(int(n) for n in shp), as_voxel_mm(vox))
        for sid, (shp, vox) in scans.items()
    }

    if spacing == "finest":
        s = min(min(v) for _, v in info.values())
    elif spacing == "coarsest":
        s = max(max(v) for _, v in info.values())
    elif isinstance(spacing, str):
        raise ValueError(
            f"grid.spacing_mm must be a number, 'finest' or 'coarsest', got {spacing!r}"
        )
    else:
        s = float(spacing)
        if not s > 0:
            raise ValueError(f"grid.spacing_mm must be positive, got {spacing}")

    if shape is None:
        extent = [max(n[d] * v[d] for n, v in info.values()) for d in range(3)]
        # The tolerance keeps a scan that exactly fills N run voxels from
        # asking for N + 1 through rounding in the product above.
        dims = tuple(max(1, math.ceil(e / s - _REL)) for e in extent)
    else:
        dims = tuple(int(n) for n in shape)
        if len(dims) != 3 or min(dims) < 1:
            raise ValueError(f"grid.shape must be three positive integers, got {shape}")

    # Centred on the world origin, so the grid does not depend on scan order.
    origin = tuple(-s * (n - 1) / 2.0 for n in dims)
    grid = GridSpec(dims, s, origin)

    placements = {}
    for sid, (n, v) in info.items():
        placed = []
        for d in range(3):
            if align == "centre":
                o = -v[d] * (n[d] - 1) / 2.0
            else:
                o = (origin[d] - s / 2.0) + v[d] / 2.0
            factor = s / v[d]
            if _is_integer(factor):
                # Pooling by an integer factor puts pooled voxel 0 at
                # o + v*(f-1)/2. Snapping that onto a run voxel makes the pool
                # exact instead of interpolated, for a shift of at most half a
                # run voxel.
                f = round(factor)
                pooled0 = o + v[d] * (f - 1) / 2.0
                offset = (pooled0 - origin[d]) / s
                o -= (offset - math.floor(offset + 0.5)) * s
            placed.append(o)
        placements[sid] = Placement(n, v, tuple(placed))
    return grid, placements


def placement_notes(grid: GridSpec, placement: Placement) -> tuple[str, list[str]]:
    """How a scan reaches the run grid, and anything worth warning about."""
    s = grid.spacing_mm
    v = placement.voxel_mm
    factors = [s / x for x in v]
    if all(abs(f - 1.0) <= _REL for f in factors):
        how = "copied"
    else:
        parts = []
        for axis, f in zip("zyx", factors):
            if abs(f - 1.0) <= _REL:
                continue
            parts.append(
                f"{axis} down x{f:.3g}" if f > 1 else f"{axis} up x{1 / f:.3g}"
            )
        how = "resampled (" + ", ".join(parts) + ")"
    warnings = []
    coarse = [axis for axis, f in zip("zyx", factors) if f < 1.0 - _REL]
    if coarse:
        warnings.append(
            f"voxels are coarser than the run grid along {', '.join(coarse)}; "
            f"interpolating up adds voxels but no detail"
        )
    lo = [placement.origin_mm[d] - v[d] / 2 for d in range(3)]
    hi = [placement.origin_mm[d] + v[d] * (placement.shape[d] - 0.5) for d in range(3)]
    glo = [grid.origin_mm[d] - s / 2 for d in range(3)]
    ghi = [grid.origin_mm[d] + s * (grid.shape[d] - 0.5) for d in range(3)]
    cut = [
        axis
        for d, axis in enumerate("zyx")
        if lo[d] < glo[d] - s / 2 or hi[d] > ghi[d] + s / 2
    ]
    if cut:
        warnings.append(
            f"extends past the run grid along {', '.join(cut)} and is cropped there"
        )
    return how, warnings


# --------------------------------------------------------------------------- #
# Reading a scan as if it were already on the run grid
# --------------------------------------------------------------------------- #
class ResampledSource:
    """A scan presented on the run grid, resampled one requested box at a time.

    Behaves like a :class:`chunkreg.io_formats.VolumeSource` with the run
    grid's shape, so :func:`chunkreg.store.ingest_source` streams it into a
    store exactly as it would a scan that was already on the grid.

    Downsampling first mean-pools by the whole part of the factor and then
    applies a Gaussian for what is left, sigma ``(r - 1) / 2`` pooled voxels
    for a remaining factor ``r``. Trilinear interpolation places the result.
    Outside the scan is zero. Large boxes are split so a read never holds more
    than ``max_read_voxels`` scan voxels at once.
    """

    def __init__(
        self,
        source,
        placement: Placement,
        grid: GridSpec,
        max_read_voxels: int = 2**28,
    ) -> None:
        if tuple(int(n) for n in source.shape) != placement.shape:
            raise ValueError(
                f"placement is for shape {placement.shape} but the source is "
                f"{tuple(source.shape)}"
            )
        self.source = source
        self.placement = placement
        self.grid = grid
        self.shape = grid.shape
        self.dtype = np.dtype(source.dtype)
        self.spacing_mm = grid.spacing_mm
        self.max_read_voxels = int(max_read_voxels)
        self.description = getattr(source, "description", "scan")

        s = grid.spacing_mm
        v = np.asarray(placement.voxel_mm, dtype=np.float64)
        factor = s / v
        self._pool = np.maximum(1, np.floor(factor + _REL)).astype(np.int64)
        self._sigma = np.maximum(0.0, (factor / self._pool - 1.0) / 2.0)
        self._pooled_mm = v * self._pool
        self._pooled_origin = (
            np.asarray(placement.origin_mm, dtype=np.float64) + v * (self._pool - 1) / 2.0
        )
        # Where run voxel 0 lands, and how far one run voxel moves, in pooled
        # voxels of the scan.
        self._p0 = (np.asarray(grid.origin_mm) - self._pooled_origin) / self._pooled_mm
        self._dp = s / self._pooled_mm
        self._copy = bool(
            np.all(self._pool == 1)
            and np.all(np.abs(factor - 1.0) <= _REL)
            and all(_is_integer(p) for p in self._p0)
        )
        self._margin = np.ceil(4.0 * self._sigma).astype(np.int64) + 1

    @property
    def mode(self) -> str:
        return "copy" if self._copy else "resample"

    def __getitem__(self, key) -> np.ndarray:
        if not isinstance(key, tuple) or len(key) != 3:
            raise IndexError("index a resampled source with three slices")
        lo, hi = [], []
        for sl, n in zip(key, self.shape):
            if not isinstance(sl, slice) or sl.step not in (None, 1):
                raise IndexError("only contiguous slices are supported")
            a, b, _ = sl.indices(n)
            lo.append(a)
            hi.append(max(a, b))
        lo = np.asarray(lo, dtype=np.int64)
        hi = np.asarray(hi, dtype=np.int64)
        if self._copy:
            return self._read_copy(lo, hi)
        return self._read_resampled(lo, hi)

    def read(self) -> np.ndarray:
        return self[(slice(None),) * 3]

    # -- copy: same spacing, whole-voxel offset ------------------------------ #
    def _read_copy(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        from .store import read_padded

        offset = np.rint(self._p0).astype(np.int64)
        return read_padded(
            self.source, lo + offset, hi - lo, mode="constant", dtype=self.dtype
        )

    # -- resample ----------------------------------------------------------- #
    def _read_resampled(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        size = hi - lo
        out = np.zeros(tuple(int(n) for n in size), dtype=np.float32)
        if np.any(size == 0):
            return out
        step = size.copy()
        while True:
            span = step * self._dp + 2 * self._margin + 2
            if np.prod(span * self._pool, dtype=np.float64) <= self.max_read_voxels:
                break
            if np.all(step == 1):
                break
            d = int(np.argmax(step * self._dp * self._pool))
            step[d] = max(1, step[d] // 2)
        for z in range(int(lo[0]), int(hi[0]), int(step[0])):
            for y in range(int(lo[1]), int(hi[1]), int(step[1])):
                for x in range(int(lo[2]), int(hi[2]), int(step[2])):
                    a = np.array([z, y, x], dtype=np.int64)
                    b = np.minimum(a + step, hi)
                    out[tuple(slice(int(p - q), int(r - q)) for p, q, r in zip(a, lo, b))] = (
                        self._resample_box(a, b)
                    )
        return out

    def _resample_box(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        coords = [
            self._p0[d] + self._dp[d] * np.arange(lo[d], hi[d], dtype=np.float64)
            for d in range(3)
        ]
        plo = np.array(
            [math.floor(c[0]) - self._margin[d] for d, c in enumerate(coords)],
            dtype=np.int64,
        )
        phi = np.array(
            [math.floor(c[-1]) + 2 + self._margin[d] for d, c in enumerate(coords)],
            dtype=np.int64,
        )
        pooled = self._pooled_block(plo, phi - plo)
        if np.any(self._sigma > 0):
            pooled = ndimage.gaussian_filter(
                pooled, sigma=tuple(float(x) for x in self._sigma), mode="constant"
            )
        a = pooled
        for d in range(3):
            a = _lerp_axis(a, coords[d] - plo[d], d)
        return a

    def _pooled_block(self, origin: np.ndarray, shape: np.ndarray) -> np.ndarray:
        """Mean-pooled scan voxels over a pooled-grid box, zero outside the scan.

        The mean is over the scan voxels a window actually contains, so a
        window straddling the scan's edge is not darkened by the air past it.
        """
        from .store import read_padded

        m = self._pool
        raw = read_padded(self.source, origin * m, shape * m, mode="constant")
        if np.all(m == 1):
            return raw
        sums = raw.reshape(
            int(shape[0]), int(m[0]), int(shape[1]), int(m[1]), int(shape[2]), int(m[2])
        ).sum(axis=(1, 3, 5), dtype=np.float64)
        counts = [
            np.clip(
                np.minimum((np.arange(shape[d]) + origin[d] + 1) * m[d], self.placement.shape[d])
                - np.maximum((np.arange(shape[d]) + origin[d]) * m[d], 0),
                0,
                None,
            ).astype(np.float64)
            for d in range(3)
        ]
        count = counts[0][:, None, None] * counts[1][None, :, None] * counts[2][None, None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(count > 0, sums / np.maximum(count, 1), 0.0)
        return mean.astype(np.float32)


def _lerp_axis(a: np.ndarray, coords: np.ndarray, axis: int) -> np.ndarray:
    """Linear interpolation of ``a`` along one axis at fractional positions.

    Trilinear interpolation on an axis-aligned grid is separable, so three of
    these replace ``map_coordinates`` without building a coordinate array the
    size of the output. Positions outside ``[0, n - 1]`` blend toward zero.
    """
    n = a.shape[axis]
    i0 = np.floor(coords).astype(np.int64)
    w = (coords - i0).astype(np.float32)
    i1 = i0 + 1
    shape = [1, 1, 1]
    shape[axis] = len(coords)

    def take(idx):
        valid = ((idx >= 0) & (idx < n)).astype(np.float32).reshape(shape)
        return np.take(a, np.clip(idx, 0, n - 1), axis=axis) * valid

    return take(i0) * (1.0 - w).reshape(shape) + take(i1) * w.reshape(shape)

"""Grids, chunk profiles, pyramid construction, tiling and blend windows.

Conventions fixed here and relied on everywhere else:

* Arrays are ordered ``(Z, Y, X)``, matching zarr, TIFF stacks, FireANTs
  ``.array`` and MONAI.
* A grid is isotropic and axis-aligned: ``world = origin + spacing * voxel``.
* The chunk geometry is a *profile* and does not vary between pyramid levels.
  The coarsest level is by construction the level at which the whole volume
  fits inside a single chunk.

The profile is the load-bearing object. Because the halo is fixed, the halo
bound of the design is read backwards: it defines the displacement clamp
rather than constraining it (see :meth:`Profile.d_max_vox`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from typing import Iterable, Iterator, Literal, Sequence

import numpy as np

__all__ = [
    "GridSpec",
    "Profile",
    "Chunk",
    "pyramid",
    "pyramid_depth",
    "tile",
    "window",
    "chunks_touching",
    "shard_grid",
    "axis_windows",
    "window_on",
]


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GridSpec:
    """An isotropic, axis-aligned sampling grid.

    Parameters
    ----------
    shape:
        Voxel counts as ``(Z, Y, X)``.
    spacing_mm:
        Isotropic voxel size in millimetres.
    origin_mm:
        World coordinate of the centre of voxel ``(0, 0, 0)``, as ``(z, y, x)``
        in millimetres.
    """

    shape: tuple[int, int, int]
    spacing_mm: float
    origin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if len(self.shape) != 3:
            raise ValueError(f"shape must be 3D, got {self.shape!r}")
        if any(int(s) <= 0 for s in self.shape):
            raise ValueError(f"shape must be positive, got {self.shape!r}")
        if not self.spacing_mm > 0:
            raise ValueError(f"spacing_mm must be positive, got {self.spacing_mm!r}")
        object.__setattr__(self, "shape", tuple(int(s) for s in self.shape))
        object.__setattr__(self, "spacing_mm", float(self.spacing_mm))
        object.__setattr__(self, "origin_mm", tuple(float(o) for o in self.origin_mm))

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def extent_mm(self) -> tuple[float, float, float]:
        """Physical size of the sampled box, centre-to-centre plus one voxel."""
        return tuple(s * self.spacing_mm for s in self.shape)

    def world(self, voxel: np.ndarray) -> np.ndarray:
        """Map voxel coordinates (..., 3) to world millimetres."""
        v = np.asarray(voxel, dtype=np.float64)
        return v * self.spacing_mm + np.asarray(self.origin_mm, dtype=np.float64)

    def voxel(self, world: np.ndarray) -> np.ndarray:
        """Map world millimetres (..., 3) to (fractional) voxel coordinates."""
        w = np.asarray(world, dtype=np.float64)
        return (w - np.asarray(self.origin_mm, dtype=np.float64)) / self.spacing_mm

    def coarsened(self, factor: int) -> "GridSpec":
        """The grid obtained by ``factor``-fold mean pooling of this one.

        Pooled voxel 0 covers source voxels ``0 .. factor-1``, so its centre
        sits ``spacing * (factor - 1) / 2`` past the source origin. Carrying
        that shift is what keeps every pyramid level in the same world frame.
        """
        if factor < 1:
            raise ValueError(f"factor must be >= 1, got {factor}")
        if factor == 1:
            return self
        shape = tuple(int(math.ceil(s / factor)) for s in self.shape)
        shift = self.spacing_mm * (factor - 1) / 2.0
        origin = tuple(o + shift for o in self.origin_mm)
        return GridSpec(shape, self.spacing_mm * factor, origin)


# --------------------------------------------------------------------------- #
# Chunk profile
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Profile:
    """Fixed chunk geometry and similarity parameters.

    One profile applies at every pyramid level, which is what makes peak GPU
    memory, the padded chunk shape and the scheduler resource request constant
    for a whole run.

    The halo is spent three ways: on the displacement the chunk is allowed to
    solve for, on the support of the similarity kernel and its regularisation
    at the coarsest in-chunk pyramid scale, and on the feature extractor's
    receptive field. Whatever is left over is the clamp.
    """

    core: int = 256
    halo: int = 48
    inner_chunk: int = 64
    lattice_factor: int = 2
    channels: int = 16
    backbone: str = "anatomix"
    features: str = "anatomix"
    r_f: int = 24
    """Feature-extractor receptive-field radius in voxels. Measured by
    ``chunkreg setup``; the default is a conservative placeholder."""
    k: int = 7
    """Local similarity (LNCC) kernel width, in voxels of the scale it runs at."""
    sigma_g: float = 1.0
    sigma_w: float = 0.5
    scales: tuple[int, ...] = (2, 1)
    """In-chunk pyramid, coarse to fine. ``max(scales)`` enters the halo
    budget, because the kernel and smoothing are specified in the voxels of
    the downsampled grid they run at."""
    taper: int | None = None
    """Hann taper width in voxels; defaults to the halo."""

    def __post_init__(self) -> None:
        if self.core <= 0 or self.halo < 0:
            raise ValueError("core must be positive and halo non-negative")
        if self.core % self.inner_chunk:
            raise ValueError(
                f"core {self.core} must be a multiple of inner_chunk {self.inner_chunk}"
            )
        if self.core % self.lattice_factor:
            raise ValueError(
                f"core {self.core} must be a multiple of lattice_factor "
                f"{self.lattice_factor}"
            )
        if self.halo % self.lattice_factor:
            raise ValueError(
                f"halo {self.halo} must be a multiple of lattice_factor "
                f"{self.lattice_factor}"
            )
        if not self.scales:
            raise ValueError("scales must be non-empty")
        if list(self.scales) != sorted(self.scales, reverse=True):
            raise ValueError(f"scales must be coarse to fine, got {self.scales!r}")
        if self.k % 2 == 0:
            raise ValueError(f"similarity kernel width k must be odd, got {self.k}")
        object.__setattr__(self, "scales", tuple(int(s) for s in self.scales))

    # -- derived geometry --------------------------------------------------- #
    @property
    def s_max(self) -> int:
        return int(max(self.scales))

    @property
    def taper_vox(self) -> int:
        return int(self.halo if self.taper is None else self.taper)

    @property
    def padded(self) -> int:
        """Side length of the padded chunk actually read and registered."""
        return self.core + 2 * self.halo

    @property
    def overhead(self) -> float:
        """Compute multiplier from registering padded chunks rather than cores."""
        return (self.padded / self.core) ** 3

    def support_vox(self, s_max: int | None = None) -> float:
        """Halo consumed by the similarity kernel and its regularisation.

        Both are specified in the voxels of the pyramid scale they run at, so
        an LNCC window of 7 at scale 8 spans 56 native voxels. Pass ``s_max``
        to evaluate the cost of a deeper in-chunk pyramid than the profile's.
        """
        s = self.s_max if s_max is None else int(s_max)
        return s * ((self.k - 1) / 2.0 + 3.0 * max(self.sigma_g, self.sigma_w))

    def d_max_vox(self) -> float:
        """Displacement clamp implied by the halo, in native voxels of a level.

        This is the halo bound solved for the clamp::

            h >= D/spacing + s_max*[(k-1)/2 + 3*max(sigma)] + r_f

        A negative result means the profile cannot register anything: the
        kernel support and feature receptive field already exceed the halo.
        """
        return self.halo - self.support_vox() - self.r_f

    def d_max_mm(self, spacing_mm: float) -> float:
        """The clamp in millimetres at a level of the given spacing."""
        return self.d_max_vox() * spacing_mm

    def mem_bytes(self, bytes_per_voxel_channel: float = 90.0) -> float:
        """Peak device bytes for one padded chunk at this channel count."""
        return bytes_per_voxel_channel * self.channels * self.padded**3

    def mem_gb(self, bytes_per_voxel_channel: float = 90.0) -> float:
        return self.mem_bytes(bytes_per_voxel_channel) / 1024**3

    def lattice_shape(self, shape: Sequence[int]) -> tuple[int, int, int]:
        """Shape of the displacement storage lattice for an image shape."""
        f = self.lattice_factor
        return tuple(int(math.ceil(s / f)) for s in shape)


# --------------------------------------------------------------------------- #
# Chunks
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Chunk:
    """One tile of a level: a core that it owns, plus the halo it may read."""

    id: int
    index: tuple[int, int, int]
    """Position in the chunk lattice, which is also the output shard id."""
    core_origin: tuple[int, int, int]
    core_shape: tuple[int, int, int]
    pad_origin: tuple[int, int, int]
    pad_shape: tuple[int, int, int]

    @property
    def shard_id(self) -> tuple[int, int, int]:
        """Shards are chosen equal to cores, so this is the chunk index."""
        return self.index

    @property
    def core_offset_in_pad(self) -> tuple[int, int, int]:
        return tuple(c - p for c, p in zip(self.core_origin, self.pad_origin))

    def core_slices(self) -> tuple[slice, slice, slice]:
        return tuple(
            slice(o, o + s) for o, s in zip(self.core_origin, self.core_shape)
        )

    def pad_slices(self) -> tuple[slice, slice, slice]:
        return tuple(slice(o, o + s) for o, s in zip(self.pad_origin, self.pad_shape))

    def intersects(self, origin: Sequence[int], shape: Sequence[int]) -> bool:
        """Does this chunk's padded box overlap the given box?"""
        lo = np.asarray(origin, dtype=np.int64)
        hi = lo + np.asarray(shape, dtype=np.int64)
        plo = np.asarray(self.pad_origin, dtype=np.int64)
        phi = plo + np.asarray(self.pad_shape, dtype=np.int64)
        return bool(np.all(np.minimum(hi, phi) > np.maximum(lo, plo)))


def pyramid_depth(shape: Sequence[int], core: int) -> int:
    """Number of *extra* levels above native so the top fits in one chunk.

    Returns ``K`` such that ``ceil(max(shape) / 2**K) <= core``.
    """
    longest = int(max(shape))
    if longest <= core:
        return 0
    return int(math.ceil(math.log2(longest / core)))


def pyramid(native: GridSpec, profile: Profile) -> list[GridSpec]:
    """Level grids from coarsest (index 0, one chunk) to native (index K).

    Level ``k`` is the native grid mean-pooled by ``2**(K-k)``.
    """
    K = pyramid_depth(native.shape, profile.core)
    return [native.coarsened(2 ** (K - k)) for k in range(K + 1)]


def shard_grid(grid: GridSpec, profile: Profile) -> tuple[int, int, int]:
    """Number of shards per axis at this level. Shards equal chunk cores."""
    return tuple(int(math.ceil(s / profile.core)) for s in grid.shape)


def tile(grid: GridSpec, profile: Profile) -> list[Chunk]:
    """Tile a level into cores of ``profile.core``, padded by ``profile.halo``.

    Cores tile the grid exactly and without overlap, so every voxel is owned by
    exactly one chunk and the blend denominator is never zero. Padded boxes are
    clipped at the volume boundary.
    """
    shape = np.asarray(grid.shape, dtype=np.int64)
    core = int(profile.core)
    halo = int(profile.halo)
    starts = [np.arange(0, int(n), core, dtype=np.int64) for n in shape]
    chunks: list[Chunk] = []
    for cid, idx in enumerate(
        product(*(range(len(s)) for s in starts))
    ):
        co = np.array([starts[d][idx[d]] for d in range(3)], dtype=np.int64)
        cs = np.minimum(core, shape - co)
        po = np.maximum(co - halo, 0)
        ph = np.minimum(co + cs + halo, shape) - po
        chunks.append(
            Chunk(
                id=cid,
                index=tuple(int(i) for i in idx),
                core_origin=tuple(int(v) for v in co),
                core_shape=tuple(int(v) for v in cs),
                pad_origin=tuple(int(v) for v in po),
                pad_shape=tuple(int(v) for v in ph),
            )
        )
    return chunks


def chunks_touching(
    chunks: Iterable[Chunk], origin: Sequence[int], shape: Sequence[int]
) -> list[Chunk]:
    """Chunks whose padded box overlaps the given box.

    This is the read set of a blend task: because shards equal cores and the
    halo is smaller than the core, it is the chunk owning the shard plus at
    most its 26 face, edge and corner neighbours.
    """
    return [c for c in chunks if c.intersects(origin, shape)]


# --------------------------------------------------------------------------- #
# Partition-of-unity windows
# --------------------------------------------------------------------------- #
def _axis_window(
    pad_len: int,
    core_lo: int,
    core_len: int,
    taper: int,
    at_low: bool,
    at_high: bool,
    kind: Literal["hann", "trapezoid"],
) -> np.ndarray:
    """1D window: 1 across the core, tapering to 0 within the halo."""
    w = np.ones(pad_len, dtype=np.float64)
    core_hi = core_lo + core_len

    def ramp(n: int) -> np.ndarray:
        t = (np.arange(n, dtype=np.float64) + 0.5) / n
        if kind == "hann":
            return 0.5 - 0.5 * np.cos(np.pi * t)
        return t

    if not at_low:
        n = min(taper, core_lo)
        if n > 0:
            w[core_lo - n : core_lo] = ramp(n)
        w[: core_lo - n] = 0.0
    if not at_high:
        tail = pad_len - core_hi
        n = min(taper, tail)
        if n > 0:
            w[core_hi : core_hi + n] = ramp(n)[::-1]
        w[core_hi + n :] = 0.0
    return w


@lru_cache(maxsize=4096)
def axis_windows(
    chunk: Chunk,
    grid: GridSpec,
    profile: Profile,
    kind: Literal["hann", "trapezoid"] = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three 1D factors whose outer product is the chunk's blend window.

    The window is separable, so the factors are what is worth computing and
    keeping: three arrays of a few hundred floats instead of the padded cube,
    which at the production profile is 174 MB and was being rebuilt once per
    neighbour per subject.
    """
    taper = profile.taper_vox
    off = chunk.core_offset_in_pad
    axes = []
    for d in range(3):
        at_low = chunk.pad_origin[d] == 0 and chunk.core_origin[d] == 0
        at_high = (
            chunk.pad_origin[d] + chunk.pad_shape[d] >= grid.shape[d]
            and chunk.core_origin[d] + chunk.core_shape[d] >= grid.shape[d]
        )
        w = _axis_window(
            chunk.pad_shape[d],
            off[d],
            chunk.core_shape[d],
            taper,
            at_low,
            at_high,
            kind,
        )
        w.flags.writeable = False  # shared across callers via the cache
        axes.append(w)
    return tuple(axes)


def window_on(
    chunk: Chunk,
    grid: GridSpec,
    profile: Profile,
    box: tuple[slice, slice, slice],
    kind: Literal["hann", "trapezoid"] = "hann",
) -> np.ndarray:
    """The blend window over a sub-box of a chunk's padded box.

    A blend task only ever weights the overlap between a neighbour's padded box
    and its own core, so only that part of the window has to exist.
    """
    wz, wy, wx = axis_windows(chunk, grid, profile, kind)
    return (
        wz[box[0]][:, None, None]
        * wy[box[1]][None, :, None]
        * wx[box[2]][None, None, :]
    ).astype(np.float32)


def window(
    chunk: Chunk,
    grid: GridSpec,
    profile: Profile,
    kind: Literal["hann", "trapezoid"] = "hann",
) -> np.ndarray:
    """Blend weight over a chunk's padded box, shape ``chunk.pad_shape``.

    The window is 1 on the core and tapers into the halo, so a chunk's own core
    is never outvoted by a neighbour's extrapolation. Windows are not required
    to sum to one: the blend divides by their analytic sum, which is at least 1
    everywhere because cores tile the grid exactly.
    """
    return window_on(
        chunk, grid, profile, (slice(None), slice(None), slice(None)), kind
    )

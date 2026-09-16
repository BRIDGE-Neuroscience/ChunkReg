"""PyTorch implementations of the array operations, for GPU runs.

Each function here is the device twin of a NumPy/SciPy function elsewhere in
the package and reproduces it, including edge handling, to floating-point
rounding. ``tests/test_gpu_ops.py`` holds every pair to that.

Two SciPy conventions are worth stating, because PyTorch's defaults differ:

* ``map_coordinates(order=1, mode="constant")`` returns ``cval`` for any sample
  outside ``[0, n - 1]`` on any axis and interpolates normally inside. PyTorch's
  ``grid_sample`` with zero padding would instead blend toward zero over the
  last half voxel, so samples are taken with border padding and the outside is
  masked explicitly.
* ``gaussian_filter`` truncates its kernel at ``int(4 * sigma + 0.5)`` voxels,
  and ``mode="nearest"`` is edge replication.

Nothing here moves data off the device except scalars the caller asked for.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .grid import GridSpec

__all__ = [
    "identity",
    "sample",
    "magnitude",
    "percentile",
    "warp",
    "warp_offset",
    "warp_stack",
    "required_margin",
    "compose",
    "clamp",
    "resample",
    "resample_field",
    "jacobian_determinant",
    "fold_fraction",
    "gaussian",
    "pool_field",
    "pool_scalar",
    "upsample_box",
    "outer3",
    "histogram",
    "tissue_fraction",
    "median_filter",
    "grid_to_disp_mm",
    "disp_mm_to_grid",
    "normalise_channels",
]


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def _axes(shape: Sequence[int], device, dtype=torch.float32) -> list[torch.Tensor]:
    return [torch.arange(int(n), device=device, dtype=dtype) for n in shape]


def identity(shape: Sequence[int], device) -> torch.Tensor:
    """Voxel coordinates of every sample, ``(3, Z, Y, X)`` float32."""
    return torch.stack(torch.meshgrid(*_axes(shape, device), indexing="ij"), dim=0)


def sample(
    vol: torch.Tensor,
    coords: torch.Tensor,
    mode: str = "constant",
    cval: float = 0.0,
) -> torch.Tensor:
    """Trilinear sampling at voxel coordinates, as ``map_coordinates(order=1)``.

    ``vol`` is ``(Z, Y, X)`` or ``(C, Z, Y, X)``; ``coords`` is ``(3, ...)`` in
    voxel units of ``vol``. ``mode`` is ``"constant"`` or ``"nearest"``.
    """
    squeeze = vol.ndim == 3
    inp = (vol[None] if squeeze else vol)[None].to(torch.float32)
    sizes = vol.shape[-3:]
    norm = []
    for d in range(3):
        n = int(sizes[d])
        if n > 1:
            norm.append(coords[d] * (2.0 / (n - 1)) - 1.0)
        else:
            norm.append(torch.zeros_like(coords[d]))
    # grid_sample reads the last axis as (x, y, z): the reverse of array order.
    grid = torch.stack(norm[::-1], dim=-1)[None]
    out = F.grid_sample(
        inp, grid, mode="bilinear", padding_mode="border", align_corners=True
    )[0]
    if squeeze:
        out = out[0]
    if mode == "constant":
        inside = torch.ones(coords.shape[1:], dtype=torch.bool, device=coords.device)
        for d in range(3):
            inside &= (coords[d] >= 0) & (coords[d] <= int(sizes[d]) - 1)
        out = torch.where(inside, out, torch.as_tensor(cval, dtype=out.dtype, device=out.device))
    elif mode != "nearest":
        raise ValueError(f"unsupported sampling mode {mode!r}")
    return out


def magnitude(u: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum(u.to(torch.float32) ** 2, dim=0))


def percentile(x: torch.Tensor, q: float) -> float:
    """``np.percentile`` with linear interpolation, exact for any size.

    ``torch.quantile`` refuses inputs above 16 million elements, which a single
    padded chunk exceeds, so the two neighbouring order statistics are found
    with ``kthvalue`` instead.
    """
    flat = x.reshape(-1).to(torch.float64)
    n = flat.numel()
    if n == 0:
        raise ValueError("percentile of an empty array")
    rank = float(q) / 100.0 * (n - 1)
    lo = int(math.floor(rank))
    hi = min(lo + 1, n - 1)
    v_lo = torch.kthvalue(flat, lo + 1).values
    if hi == lo or rank == lo:
        return float(v_lo)
    v_hi = torch.kthvalue(flat, hi + 1).values
    return float(v_lo + (v_hi - v_lo) * (rank - lo))


def warp(image, disp_mm, spacing_mm, order=1, cval=0.0):
    _linear_only(order)
    img = image.to(torch.float32)
    u = disp_mm.to(torch.float32)
    if u.shape[0] != 3 or tuple(u.shape[1:]) != tuple(img.shape):
        raise ValueError(
            f"field {tuple(u.shape)} does not match image {tuple(img.shape)} "
            f"as (3, *image.shape)"
        )
    coords = identity(img.shape, img.device) + u / float(spacing_mm)
    return sample(img, coords, "constant", cval)


def warp_offset(source, offset, disp_mm, spacing_mm, cval=0.0, order=1):
    _linear_only(order)
    u = disp_mm.to(torch.float32)
    off = torch.as_tensor(
        [float(o) for o in offset], dtype=torch.float32, device=u.device
    ).reshape(3, 1, 1, 1)
    coords = identity(u.shape[1:], u.device) + off + u / float(spacing_mm)
    return sample(source.to(torch.float32), coords, "constant", cval)


def warp_stack(stack, disp_mm, spacing_mm, order=1, cval=0.0):
    _linear_only(order)
    a = stack.to(torch.float32)
    u = disp_mm.to(torch.float32)
    if a.ndim != 4:
        raise ValueError(f"expected a (C, Z, Y, X) stack, got {tuple(a.shape)}")
    if u.shape[0] != 3 or tuple(u.shape[1:]) != tuple(a.shape[1:]):
        raise ValueError(
            f"field {tuple(u.shape)} does not match stack {tuple(a.shape)}"
        )
    coords = identity(a.shape[1:], a.device) + u / float(spacing_mm)
    return sample(a, coords, "constant", cval)


def required_margin(disp_mm, spacing_mm, extra=2) -> int:
    if disp_mm.numel() == 0:
        return int(extra)
    peak = float(disp_mm.abs().max())
    return int(math.ceil(peak / float(spacing_mm))) + int(extra)


def compose(inner, outer, spacing_mm):
    ui = inner.to(torch.float32)
    uo = outer.to(torch.float32)
    if ui.shape != uo.shape:
        raise ValueError(
            f"fields must share a grid, got {tuple(ui.shape)} and {tuple(uo.shape)}"
        )
    coords = identity(ui.shape[1:], ui.device) + ui / float(spacing_mm)
    return ui + sample(uo, coords, "nearest")


def clamp(disp_mm, max_mm):
    u = disp_mm.to(torch.float32)
    if max_mm is None or max_mm <= 0:
        return u
    mag = magnitude(u)
    over = mag > max_mm
    if not bool(over.any()):
        return u
    scale = torch.where(over, float(max_mm) / mag.clamp_min(1e-30), torch.ones_like(mag))
    return u * scale[None]


def _grid_coords(src: GridSpec, dst: GridSpec, device) -> torch.Tensor:
    axes = []
    for d in range(3):
        i = torch.arange(int(dst.shape[d]), device=device, dtype=torch.float32)
        world = i * float(dst.spacing_mm) + float(dst.origin_mm[d])
        axes.append((world - float(src.origin_mm[d])) / float(src.spacing_mm))
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)


def resample(image, src_grid: GridSpec, dst_grid: GridSpec, cval=0.0):
    """``resample_scalar``: edge-replicating, like its NumPy twin."""
    coords = _grid_coords(src_grid, dst_grid, image.device)
    return sample(image.to(torch.float32), coords, "nearest", cval)


def resample_field(disp_mm, src_grid: GridSpec, dst_grid: GridSpec):
    coords = _grid_coords(src_grid, dst_grid, disp_mm.device)
    return sample(disp_mm.to(torch.float32), coords, "nearest")


def _linear_only(order: int) -> None:
    if int(order) != 1:
        raise NotImplementedError("the GPU path interpolates linearly only (order=1)")


# --------------------------------------------------------------------------- #
# Derivatives
# --------------------------------------------------------------------------- #
def _gradient(f: torch.Tensor, axis: int, h: float) -> torch.Tensor:
    """``np.gradient(f, h, axis=axis, edge_order=1)``."""
    n = f.shape[axis]
    if n < 2:
        raise ValueError(
            "Shape of array too small to calculate a numerical gradient, "
            "at least (edge_order + 1) elements are required."
        )
    out = torch.empty_like(f)

    def sl(a, b):
        idx = [slice(None)] * f.ndim
        idx[axis] = slice(a, b)
        return tuple(idx)

    if n > 2:
        out[sl(1, -1)] = (f[sl(2, None)] - f[sl(None, -2)]) / (2.0 * h)
    out[sl(0, 1)] = (f[sl(1, 2)] - f[sl(0, 1)]) / h
    out[sl(n - 1, n)] = (f[sl(n - 1, n)] - f[sl(n - 2, n - 1)]) / h
    return out


def jacobian_determinant(disp_mm, spacing_mm):
    u = disp_mm.to(torch.float32)
    s = float(spacing_mm)
    j = [[_gradient(u[d], e, s) for e in range(3)] for d in range(3)]
    for d in range(3):
        j[d][d] = j[d][d] + 1.0
    return (
        j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
        - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
        + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0])
    )


def fold_fraction(disp_mm, spacing_mm) -> float:
    det = jacobian_determinant(disp_mm, spacing_mm)
    return float((det <= 0.0).to(torch.float32).mean())


# --------------------------------------------------------------------------- #
# Filters and lattices
# --------------------------------------------------------------------------- #
def _gauss_kernel(sigma: float, truncate: float, device) -> torch.Tensor:
    radius = int(truncate * float(sigma) + 0.5)
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float64)
    phi = torch.exp(-0.5 / (float(sigma) ** 2) * x * x)
    return (phi / phi.sum()).to(torch.float32)


def gaussian(
    vol: torch.Tensor,
    sigma,
    mode: str = "nearest",
    truncate: float = 4.0,
) -> torch.Tensor:
    """``ndimage.gaussian_filter`` over the last three axes.

    ``vol`` is ``(Z, Y, X)`` or ``(C, Z, Y, X)``; each leading channel is
    filtered on its own, as filtering the components of a field separately
    would be.
    """
    sig = (float(sigma),) * 3 if np.isscalar(sigma) else tuple(float(s) for s in sigma)
    squeeze = vol.ndim == 3
    x = (vol[None] if squeeze else vol)[:, None].to(torch.float32)  # (C, 1, Z, Y, X)
    pad_mode = {"nearest": "replicate", "constant": "constant"}[mode]
    for axis, s in enumerate(sig):
        if s <= 0:
            continue
        k = _gauss_kernel(s, truncate, x.device)
        r = (k.numel() - 1) // 2
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = k.numel()
        pad = [0, 0, 0, 0, 0, 0]  # x_lo, x_hi, y_lo, y_hi, z_lo, z_hi
        pad[2 * (2 - axis)] = r
        pad[2 * (2 - axis) + 1] = r
        x = F.conv3d(F.pad(x, pad, mode=pad_mode), k.reshape(shape))
    out = x[:, 0]
    return out[0] if squeeze else out


def _pad_edge_to(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Edge-pad the last three axes up to a multiple of ``factor``."""
    extra = [(-int(n)) % factor for n in x.shape[-3:]]
    if not any(extra):
        return x
    return F.pad(x[None], [0, extra[2], 0, extra[1], 0, extra[0]], mode="replicate")[0]


def pool_field(u: torch.Tensor, factor: int, target=None) -> torch.Tensor:
    """Mean-pool a ``(C, Z, Y, X)`` block by ``factor``, edge-padding a ragged end.

    ``target`` crops or pads the pooled result to a lattice extent, which is
    what a padded chunk stored on its lattice box needs.
    """
    a = u.to(torch.float32)
    f = int(factor)
    if target is not None:
        need = [int(n) * f for n in target]
        grow = [max(0, n - s) for n, s in zip(need, a.shape[-3:])]
        if any(grow):
            a = F.pad(a[None], [0, grow[2], 0, grow[1], 0, grow[0]], mode="replicate")[0]
        a = a[..., : need[0], : need[1], : need[2]]
    if f == 1:
        return a
    a = _pad_edge_to(a, f)
    return F.avg_pool3d(a[None], f)[0]


def pool_scalar(v: torch.Tensor, factor: int) -> torch.Tensor:
    return pool_field(v[None], factor)[0]


def upsample_box(lat: torch.Tensor, factor: int, box) -> torch.Tensor:
    """Nearest-neighbour upsample of a lattice over one sub-box, as ``np.repeat``."""
    a = lat.to(torch.float32)
    f = int(factor)
    if f == 1:
        return a[(slice(None),) + tuple(box)].contiguous()
    lo = [b.start // f for b in box]
    hi = [-(-b.stop // f) for b in box]
    sub = a[:, lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    up = sub.repeat_interleave(f, dim=1).repeat_interleave(f, dim=2).repeat_interleave(f, dim=3)
    inner = (slice(None),) + tuple(
        slice(b.start - lo[d] * f, b.start - lo[d] * f + (b.stop - b.start))
        for d, b in enumerate(box)
    )
    return up[inner].contiguous()


def outer3(wz, wy, wx) -> torch.Tensor:
    return wz[:, None, None] * wy[None, :, None] * wx[None, None, :]


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
_EDGES: dict[str, torch.Tensor] = {}


def histogram(values: torch.Tensor, edges: np.ndarray) -> np.ndarray:
    """``chunkreg.stats.histogram`` on the device; only the counts come back."""
    v = values.reshape(-1).to(torch.float64)
    n_bins = len(edges)
    if v.numel() == 0:
        return np.zeros(n_bins, dtype=np.int64)
    key = str(v.device)
    e = _EDGES.get(key)
    if e is None:
        e = _EDGES[key] = torch.as_tensor(edges, dtype=torch.float64, device=v.device)
    idx = torch.bucketize(v, e, right=True) - 1
    idx = idx.clamp_(0, n_bins - 1)
    return torch.bincount(idx, minlength=n_bins).to("cpu").numpy().astype(np.int64)


def tissue_fraction(patch: torch.Tensor, threshold: float = 0.0) -> float:
    a = patch.to(torch.float32)
    if a.numel() == 0:
        return 0.0
    thr = threshold if threshold > 0 else max(0.02, 0.05 * float(a.max()))
    return float((a > thr).to(torch.float32).mean())


def _symmetric_index(n: int, lo: int, hi: int, device) -> torch.Tensor:
    """Indices for SciPy's ``reflect`` padding (edge sample repeated)."""
    i = torch.arange(-lo, n + hi, device=device)
    period = 2 * n
    i = torch.remainder(i, period)
    return torch.where(i >= n, period - 1 - i, i)


def median_filter(vol: torch.Tensor, size: int, slab: int = 8) -> torch.Tensor:
    """``ndimage.median_filter(vol, size=size)`` with SciPy's defaults.

    The window covers ``[i - size // 2, i + (size - 1) // 2]`` on each axis, the
    rank taken is ``size**3 // 2`` (the upper median for an even window), and
    the edges reflect with the edge sample repeated. Worked in slabs so the
    unfolded windows stay small.
    """
    k = int(size)
    if k <= 1:
        return vol
    x = vol.to(torch.float32)
    lo, hi = k // 2, (k - 1) // 2
    for axis in range(3):
        idx = _symmetric_index(int(x.shape[axis]), lo, hi, x.device)
        x = x.index_select(axis, idx)
    rank = (k ** 3) // 2
    zout = int(vol.shape[0])
    out = torch.empty(tuple(vol.shape), dtype=torch.float32, device=vol.device)
    for z0 in range(0, zout, slab):
        z1 = min(z0 + slab, zout)
        part = x[z0 : z1 + k - 1].unfold(0, k, 1).unfold(1, k, 1).unfold(2, k, 1)
        flat = part.reshape(part.shape[0], part.shape[1], part.shape[2], -1)
        out[z0:z1] = torch.kthvalue(flat, rank + 1, dim=-1).values
    return out


# --------------------------------------------------------------------------- #
# Engine bridge and feature normalisation
# --------------------------------------------------------------------------- #
def grid_to_disp_mm(grid_norm, shape, spacing_mm):
    g = grid_norm.to(torch.float32)
    if g.ndim == 5:
        if g.shape[0] != 1:
            raise ValueError(f"expected a batch of 1, got {g.shape[0]}")
        g = g[0]
    shape = tuple(int(n) for n in shape)
    if g.ndim != 4 or g.shape[-1] != 3 or tuple(g.shape[:3]) != shape:
        raise ValueError(f"grid {tuple(g.shape)} does not match shape {shape}")
    vox = torch.stack(
        [(g[..., 2 - d] + 1.0) * 0.5 * max(shape[d] - 1, 1) for d in range(3)], dim=0
    )
    return (vox - identity(shape, g.device)) * float(spacing_mm)


def disp_mm_to_grid(disp_mm, spacing_mm):
    u = disp_mm.to(torch.float32)
    shape = tuple(u.shape[1:])
    vox = identity(shape, u.device) + u / float(spacing_mm)
    comps = [vox[d] / max(shape[d] - 1, 1) * 2.0 - 1.0 for d in range(3)]
    return torch.stack(comps[::-1], dim=-1)[None]


def normalise_channels(x, method: str = "l2"):
    a = x.to(torch.float32)
    if method == "none":
        return a
    if method == "l2":
        n = torch.sqrt(torch.sum(a * a, dim=0, keepdim=True))
        return a / n.clamp_min(1e-6)
    if method == "standardized":
        mean = a.mean(dim=0, keepdim=True)
        std = torch.sqrt(((a - mean) ** 2).mean(dim=0, keepdim=True))  # np.std, ddof=0
        return (a - mean) / std.clamp_min(1e-6)
    raise ValueError(
        f"unknown feature normalisation {method!r}; use 'l2', 'standardized' or 'none'"
    )

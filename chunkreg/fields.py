"""Displacement-field conventions and operations.

Every transform in ``chunkreg`` is a dense displacement field in millimetres,
stored as ``(3, Z, Y, X)``. The map is::

    phi(x) = x + u(x)

and it sends **template (fixed) coordinates to subject (moving) coordinates**,
so the warped subject is ``M(x + u(x))``. This is the ANTs and anatomix
direction.

Component order
---------------
Components are stored in **array-axis order** ``(z, y, x)``: component ``d`` of
the field displaces along array axis ``d``. This deviates from the design note,
which proposed ``(x, y, z)`` to match ``grid_sample``'s last axis. Axis order
is used by every function in this module and by every blockwise read in the
store, whereas ``grid_sample`` is touched in exactly one place, so the flip is
confined to :func:`disp_mm_to_grid` and :func:`grid_to_disp_mm`. Keeping the
storage order aligned with the array order removes a whole class of silent
transposition bugs.

Interpolation is trilinear (``order=1``) throughout. Images sample with zero
padding beyond the volume; fields sample with edge replication, so composing
near a boundary extends the last valid vector rather than pulling in zeros.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from scipy import ndimage

from . import xp as _xp
from .grid import GridSpec


def _gpu():
    from . import gpu_ops

    return gpu_ops

__all__ = [
    "identity",
    "warp",
    "warp_stack",
    "warp_offset",
    "required_margin",
    "compose",
    "clamp",
    "resample_scalar",
    "resample_field",
    "change_lattice",
    "jacobian_determinant",
    "fold_fraction",
    "magnitude",
    "percentile_magnitude",
    "grid_to_disp_mm",
    "disp_mm_to_grid",
    "pool_field",
    "smooth",
    "DISP_DTYPE",
]

DISP_DTYPE = np.float16
"""On-disk dtype for displacement fields. Half precision is ~5e-4 mm relative
resolution at a 10 mm displacement, far below any achievable registration
accuracy, and halves the largest non-image object in the pipeline."""

_IMAGE_MODE = "constant"
_FIELD_MODE = "nearest"


def _as_f32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)


def identity(shape: Sequence[int]) -> np.ndarray:
    """Voxel coordinates of every sample, as ``(3, Z, Y, X)`` float32."""
    axes = [np.arange(int(n), dtype=np.float32) for n in shape]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=0)


def magnitude(disp_mm: np.ndarray) -> np.ndarray:
    """Per-voxel displacement magnitude in millimetres, shape ``(Z, Y, X)``."""
    if _xp.is_tensor(disp_mm):
        return _gpu().magnitude(disp_mm)
    d = np.asarray(disp_mm, dtype=np.float32)
    return np.sqrt(np.sum(d * d, axis=0, dtype=np.float32))


def percentile_magnitude(disp_mm: np.ndarray, q: float = 99.9) -> float:
    """A percentile of displacement magnitude, the ``d99`` of the stopping rule."""
    if _xp.is_tensor(disp_mm):
        return _gpu().percentile(_gpu().magnitude(disp_mm), q)
    return float(np.percentile(magnitude(disp_mm), q))


def warp(
    image: np.ndarray,
    disp_mm: np.ndarray,
    spacing_mm: float,
    order: int = 1,
    cval: float = 0.0,
) -> np.ndarray:
    """Sample ``image`` through ``phi``: return ``image(x + u(x))``.

    ``image`` and ``disp_mm`` share a grid. Samples outside the volume take
    ``cval``, so a subject that does not cover the template's field of view
    contributes zero rather than a wrapped or clamped intensity.
    """
    if _xp.is_tensor(image):
        return _gpu().warp(image, disp_mm, spacing_mm, order, cval)
    img = _as_f32(image)
    u = _as_f32(disp_mm)
    if u.shape[0] != 3 or u.shape[1:] != img.shape:
        raise ValueError(
            f"field {u.shape} does not match image {img.shape} as (3, *image.shape)"
        )
    coords = identity(img.shape) + u / float(spacing_mm)
    return ndimage.map_coordinates(
        img, coords, order=order, mode=_IMAGE_MODE, cval=cval, prefilter=False
    ).astype(np.float32, copy=False)


def warp_stack(
    stack: np.ndarray,
    disp_mm: np.ndarray,
    spacing_mm: float,
    order: int = 1,
    cval: float = 0.0,
) -> np.ndarray:
    """Warp every channel of a ``(C, Z, Y, X)`` stack through one field.

    The sampling coordinates depend only on the field, so they are built once
    and reused across channels rather than rebuilt per channel.
    """
    if _xp.is_tensor(stack):
        return _gpu().warp_stack(stack, disp_mm, spacing_mm, order, cval)
    a = _as_f32(stack)
    u = _as_f32(disp_mm)
    if a.ndim != 4:
        raise ValueError(f"expected a (C, Z, Y, X) stack, got {a.shape}")
    if u.shape[0] != 3 or u.shape[1:] != a.shape[1:]:
        raise ValueError(
            f"field {u.shape} does not match stack {a.shape} as (3, *stack.shape[1:])"
        )
    coords = identity(a.shape[1:]) + u / float(spacing_mm)
    return np.stack(
        [
            ndimage.map_coordinates(
                a[c], coords, order=order, mode=_IMAGE_MODE, cval=cval, prefilter=False
            )
            for c in range(a.shape[0])
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def warp_offset(
    source: np.ndarray,
    offset: Sequence[int],
    disp_mm: np.ndarray,
    spacing_mm: float,
    cval: float = 0.0,
    order: int = 1,
) -> np.ndarray:
    """Warp out of a larger source block into a smaller target box.

    ``source`` covers the target box expanded by a margin; ``offset`` is the
    target's origin within ``source``. The result has the field's shape.

    This exists because a seeded chunk cannot be warped in place. The seed
    carried down from a coarser level can be many times the halo, so sampling
    ``x + u_seed(x)`` reaches outside the padded chunk. Reading the subject
    with a margin sized from the seed actually present, then warping into the
    chunk, is what keeps the pre-warp exact instead of fading to zero at the
    chunk edge.
    """
    if _xp.is_tensor(source):
        return _gpu().warp_offset(source, offset, disp_mm, spacing_mm, cval, order)
    src = _as_f32(source)
    u = _as_f32(disp_mm)
    shape = tuple(u.shape[1:])
    off = np.asarray(offset, dtype=np.float32).reshape(3, 1, 1, 1)
    coords = identity(shape) + off + u / np.float32(spacing_mm)
    return ndimage.map_coordinates(
        src, coords, order=order, mode=_IMAGE_MODE, cval=cval, prefilter=False
    ).astype(np.float32, copy=False)


def required_margin(disp_mm: np.ndarray, spacing_mm: float, extra: int = 2) -> int:
    """Voxels of source margin needed to warp through this field without
    sampling outside the block that was read."""
    if _xp.is_tensor(disp_mm):
        return _gpu().required_margin(disp_mm, spacing_mm, extra)
    if disp_mm.size == 0:
        return extra
    peak = float(np.max(np.abs(np.asarray(disp_mm, dtype=np.float32))))
    return int(np.ceil(peak / float(spacing_mm))) + int(extra)


def compose(inner: np.ndarray, outer: np.ndarray, spacing_mm: float) -> np.ndarray:
    """Displacement of the map that applies ``inner`` and then ``outer``.

    ::

        result(x) = inner(x) + outer(x + inner(x))

    Coordinates flow ``x -> x + inner -> x + inner + outer(...)``, so this is
    ``phi_outer . phi_inner`` as function composition.

    For seeded registration the subject is pre-warped by the seed and the
    engine solves a residual in that frame, giving
    ``compose(inner=residual, outer=seed)``.
    """
    if _xp.is_tensor(inner):
        return _gpu().compose(inner, _xp.put(outer), spacing_mm)
    ui = _as_f32(inner)
    uo = _as_f32(outer)
    if ui.shape != uo.shape:
        raise ValueError(f"fields must share a grid, got {ui.shape} and {uo.shape}")
    coords = identity(ui.shape[1:]) + ui / float(spacing_mm)
    outer_at = np.stack(
        [
            ndimage.map_coordinates(
                uo[d], coords, order=1, mode=_FIELD_MODE, prefilter=False
            )
            for d in range(3)
        ],
        axis=0,
    )
    return (ui + outer_at).astype(np.float32, copy=False)


def clamp(disp_mm: np.ndarray, max_mm: float | None) -> np.ndarray:
    """Limit displacement magnitude to ``max_mm``, preserving direction.

    This is what turns the halo bound from a hope into a guarantee: a chunk
    physically cannot move a feature further than its halo allows. ``None`` or
    a non-positive limit disables the clamp.
    """
    if _xp.is_tensor(disp_mm):
        return _gpu().clamp(disp_mm, max_mm)
    u = _as_f32(disp_mm)
    if max_mm is None or max_mm <= 0:
        return u
    mag = magnitude(u)
    scale = np.ones_like(mag)
    over = mag > max_mm
    if np.any(over):
        scale[over] = max_mm / mag[over]
        u = u * scale[None]
    return u.astype(np.float32, copy=False)


def _grid_coords(src_grid: GridSpec, dst_grid: GridSpec) -> np.ndarray:
    """Where each sample of ``dst_grid`` sits in ``src_grid``'s voxel frame."""
    ident = identity(dst_grid.shape)
    org_s = np.asarray(src_grid.origin_mm, dtype=np.float32).reshape(3, 1, 1, 1)
    org_d = np.asarray(dst_grid.origin_mm, dtype=np.float32).reshape(3, 1, 1, 1)
    world = ident * np.float32(dst_grid.spacing_mm) + org_d
    return (world - org_s) / np.float32(src_grid.spacing_mm)


def _sample_on(
    src: np.ndarray,
    src_grid: GridSpec,
    dst_grid: GridSpec,
    mode: str,
    order: int,
    cval: float,
) -> np.ndarray:
    """Resample a single ``(Z, Y, X)`` volume from one grid onto another."""
    return ndimage.map_coordinates(
        _as_f32(src),
        _grid_coords(src_grid, dst_grid),
        order=order,
        mode=mode,
        cval=cval,
        prefilter=False,
    ).astype(np.float32, copy=False)


def resample_scalar(
    image: np.ndarray,
    src_grid: GridSpec,
    dst_grid: GridSpec,
    order: int = 1,
    cval: float = 0.0,
) -> np.ndarray:
    """Resample a scalar volume between grids that share a world frame."""
    if _xp.is_tensor(image):
        return _gpu().resample(image, src_grid, dst_grid, cval)
    return _sample_on(image, src_grid, dst_grid, _FIELD_MODE, order, cval)


def resample_field(
    disp_mm: np.ndarray,
    src_grid: GridSpec,
    dst_grid: GridSpec,
) -> np.ndarray:
    """Resample a displacement field between grids sharing a world frame.

    Because the field is stored in millimetres, this changes only the lattice
    the vectors are sampled on; the vectors themselves are untouched. That is
    exactly what makes promoting a seed from one pyramid level to the next
    exact rather than approximate.
    """
    if _xp.is_tensor(disp_mm):
        return _gpu().resample_field(disp_mm, src_grid, dst_grid)
    u = _as_f32(disp_mm)
    # One coordinate array for all three components: the lattice change is the
    # same for each, and this is the hottest resample in the pipeline.
    coords = _grid_coords(src_grid, dst_grid)
    return np.stack(
        [
            ndimage.map_coordinates(
                u[d], coords, order=1, mode=_FIELD_MODE, prefilter=False
            )
            for d in range(3)
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def change_lattice(
    disp_mm: np.ndarray,
    field_grid: GridSpec,
    out_grid: GridSpec,
) -> np.ndarray:
    """Alias of :func:`resample_field`, named for the storage-lattice use.

    Fields are stored on a lattice coarser than their level by
    ``profile.lattice_factor``; this moves between the two.
    """
    return resample_field(disp_mm, field_grid, out_grid)


def jacobian_determinant(disp_mm: np.ndarray, spacing_mm: float) -> np.ndarray:
    """Determinant of the Jacobian of ``phi``, shape ``(Z, Y, X)``.

    Values at or below zero mark folds, where the map has stopped being a
    diffeomorphism. Central differences are used inside and one-sided at the
    boundary, so the result is defined everywhere.
    """
    if _xp.is_tensor(disp_mm):
        return _gpu().jacobian_determinant(disp_mm, spacing_mm)
    u = _as_f32(disp_mm)
    s = float(spacing_mm)
    # grad[d][e] = d u_d / d x_e, with x in millimetres.
    jac = np.empty((3, 3) + u.shape[1:], dtype=np.float32)
    for d in range(3):
        grads = np.gradient(u[d], s, axis=(0, 1, 2), edge_order=1)
        for e in range(3):
            jac[d, e] = grads[e]
    for d in range(3):
        jac[d, d] += 1.0
    det = (
        jac[0, 0] * (jac[1, 1] * jac[2, 2] - jac[1, 2] * jac[2, 1])
        - jac[0, 1] * (jac[1, 0] * jac[2, 2] - jac[1, 2] * jac[2, 0])
        + jac[0, 2] * (jac[1, 0] * jac[2, 1] - jac[1, 1] * jac[2, 0])
    )
    return det.astype(np.float32, copy=False)


def fold_fraction(disp_mm: np.ndarray, spacing_mm: float) -> float:
    """Fraction of voxels where the map folds. A QC gate and a retry trigger."""
    if _xp.is_tensor(disp_mm):
        return _gpu().fold_fraction(disp_mm, spacing_mm)
    det = jacobian_determinant(disp_mm, spacing_mm)
    return float(np.mean(det <= 0.0))


# --------------------------------------------------------------------------- #
# The engine bridge: the only place coordinate order and normalisation change
# --------------------------------------------------------------------------- #
def grid_to_disp_mm(
    grid_norm: np.ndarray,
    shape: Sequence[int],
    spacing_mm: float,
) -> np.ndarray:
    """Convert an engine sampling grid to a displacement field in millimetres.

    ``grid_norm`` is what FireANTs and anatomix call ``warped_coordinates``:
    shape ``(1, Z, Y, X, 3)`` or ``(Z, Y, X, 3)``, normalised to ``[-1, 1]``
    with ``align_corners=True``, last axis ordered ``(x, y, z)``.

    Both the ``[-1, 1]`` normalisation and the reversed component order are
    undone here, so everything downstream sees millimetres in axis order.
    """
    if _xp.is_tensor(grid_norm):
        return _gpu().grid_to_disp_mm(grid_norm, shape, spacing_mm)
    g = np.asarray(grid_norm, dtype=np.float32)
    if g.ndim == 5:
        if g.shape[0] != 1:
            raise ValueError(f"expected a batch of 1, got {g.shape[0]}")
        g = g[0]
    if g.ndim != 4 or g.shape[-1] != 3:
        raise ValueError(f"expected (Z, Y, X, 3) sampling grid, got {g.shape}")
    shape = tuple(int(n) for n in shape)
    if g.shape[:3] != shape:
        raise ValueError(f"grid {g.shape[:3]} does not match shape {shape}")
    vox = np.empty((3,) + shape, dtype=np.float32)
    for d in range(3):
        n = max(shape[d] - 1, 1)
        vox[d] = (g[..., 2 - d] + 1.0) * 0.5 * n
    return ((vox - identity(shape)) * np.float32(spacing_mm)).astype(
        np.float32, copy=False
    )


def disp_mm_to_grid(disp_mm: np.ndarray, spacing_mm: float) -> np.ndarray:
    """Inverse of :func:`grid_to_disp_mm`, returning ``(1, Z, Y, X, 3)``.

    Used to hand a seed back to an engine that wants an initial sampling grid.
    """
    if _xp.is_tensor(disp_mm):
        return _gpu().disp_mm_to_grid(disp_mm, spacing_mm)
    u = _as_f32(disp_mm)
    shape = u.shape[1:]
    vox = identity(shape) + u / np.float32(spacing_mm)
    g = np.empty(shape + (3,), dtype=np.float32)
    for d in range(3):
        n = max(shape[d] - 1, 1)
        g[..., 2 - d] = vox[d] / n * 2.0 - 1.0
    return g[None]


def pool_field(u, factor: int, target: Sequence[int] | None = None):
    """Mean-pool a ``(C, Z, Y, X)`` block onto a lattice ``factor`` times coarser.

    A ragged end is edge-padded to a whole lattice voxel. ``target`` fixes the
    lattice extent, cropping or edge-padding the block to fit it first, which is
    what a padded chunk stored on its lattice box needs.
    """
    if _xp.is_tensor(u):
        return _gpu().pool_field(u, factor, target)
    a = np.asarray(u, dtype=np.float32)
    f = int(factor)
    if target is not None:
        need = tuple(int(n) * f for n in target)
        grow = [(0, 0)] * (a.ndim - 3) + [
            (0, max(0, n - s)) for n, s in zip(need, a.shape[-3:])
        ]
        if any(g[1] for g in grow):
            a = np.pad(a, grow, mode="edge")
        a = a[..., : need[0], : need[1], : need[2]]
    if f == 1:
        return a
    extra = [(-int(n)) % f for n in a.shape[-3:]]
    if any(extra):
        a = np.pad(a, [(0, 0)] * (a.ndim - 3) + [(0, e) for e in extra], mode="edge")
    n = [int(x) // f for x in a.shape[-3:]]
    lead = a.shape[:-3]
    return a.reshape(lead + (n[0], f, n[1], f, n[2], f)).mean(axis=(-5, -3, -1))


def smooth(a, sigma: float, mode: str = "nearest"):
    """Gaussian smoothing over the spatial axes, channel by channel.

    ``a`` is ``(Z, Y, X)`` or ``(C, Z, Y, X)``. Each component of a field is
    smoothed on its own, never across components.
    """
    if _xp.is_tensor(a):
        return _gpu().gaussian(a, sigma, mode)
    x = np.asarray(a, dtype=np.float32)
    if x.ndim == 3:
        return ndimage.gaussian_filter(x, sigma, mode=mode).astype(np.float32, copy=False)
    return np.stack(
        [ndimage.gaussian_filter(x[c], sigma, mode=mode) for c in range(x.shape[0])]
    ).astype(np.float32, copy=False)

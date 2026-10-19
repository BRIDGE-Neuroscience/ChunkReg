"""A subject read through an affine from a scan left in its own frame.

Registering BigBrain onto S-21-46 at 20 um would otherwise begin by writing
BigBrain resampled onto S-21-46's lattice: 7375 x 7465 x 7465 voxels, most of
them empty, hours of a single process. Instead the run grid's voxel positions
are mapped through the affine every time a block is read, and the scan is
sampled where they land. The store records the map; every pass sees an array
of the run grid's shape at every level, so nothing downstream knows.

The map is fixed when the store is made and kept in its metadata::

    run-level voxel --(level grid)--> run world mm
                    --(``world_mm_to_source_um``)--> source physical um
                    --(dataset scale, translation)--> source dataset voxel

``world_mm_to_source_um`` is one 4x4 matrix: the fixed scan's physical frame
composed with the inverse of the pair affine. A block at run level ``k`` is
sampled from the source dataset whose voxel is the largest not coarser than the
level's, trilinearly, zero outside the scan, with the source's background
(BigBrain's white, 255) set to zero first so that outside and background agree.

Conventions: arrays are ``(z, y, x)``; a 4x4 matrix acts on homogeneous
``(z, y, x, 1)`` columns; a dataset's voxel ``i`` sits at physical
``scale * i + translation`` micrometres (OME-NGFF).
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import xp as _xp
from .grid import GridSpec

__all__ = ["AffineArray", "build_spec", "ome_datasets", "world_to_source_um"]

_UNIT_UM = {
    "nanometer": 1e-3, "nm": 1e-3,
    "micrometer": 1.0, "micron": 1.0, "um": 1.0, "µm": 1.0,
    "millimeter": 1e3, "mm": 1e3,
    "centimeter": 1e4, "cm": 1e4,
    "meter": 1e6, "m": 1e6,
}


# --------------------------------------------------------------------------- #
# Reading an OME-Zarr pyramid's geometry (NGFF 0.4, 0.5 and 0.6)
# --------------------------------------------------------------------------- #
def _flatten(transforms: Sequence[dict]) -> list[dict]:
    out: list[dict] = []
    for tr in transforms:
        if tr.get("type") == "sequence":
            out += _flatten(tr.get("transformations") or [])
        else:
            out.append(tr)
    return out


def _axis_units(multiscale: dict) -> list[float]:
    """Per-axis factor to micrometres, for the spatial axes in stored order."""
    axes = multiscale.get("axes")
    if not axes:
        systems = multiscale.get("coordinateSystems") or []
        axes = systems[0]["axes"] if systems else []
    out = []
    for ax in axes:
        unit = ax.get("unit") if isinstance(ax, dict) else None
        if unit is None:
            out.append(1.0)
            continue
        if str(unit).lower() not in _UNIT_UM:
            raise ValueError(f"unrecognised OME unit {unit!r}")
        out.append(_UNIT_UM[str(unit).lower()])
    return out


def ome_datasets(root) -> list[dict]:
    """Each dataset of an OME-Zarr pyramid with its scale and translation in um.

    Returns ``[{"path", "shape", "scale_um", "translation_um"}, ...]`` in the
    file's order (finest first). Only transforms into the datasets' own
    physical system are read; a multiscale-level transform into another system
    (an MNI affine, say) is metadata and is left alone.
    """
    import zarr

    root = Path(root)
    doc = json.loads((root / "zarr.json").read_text(encoding="utf-8"))
    attrs = doc.get("attributes") or {}
    ome = attrs.get("ome", attrs)
    m = ome["multiscales"][0]
    units = _axis_units(m)
    out = []
    for ds in m["datasets"]:
        arr = zarr.open_array(str(root / ds["path"]), mode="r")
        nd = arr.ndim
        scale = np.ones(nd)
        trans = np.zeros(nd)
        for tr in _flatten(ds.get("coordinateTransformations") or []):
            kind = tr.get("type")
            if kind == "scale":
                s = np.asarray(tr["scale"], dtype=np.float64)
                scale, trans = scale * s, trans * s
            elif kind == "translation":
                trans = trans + np.asarray(tr["translation"], dtype=np.float64)
            elif kind == "identity":
                continue
            else:
                raise ValueError(
                    f"dataset {ds['path']!r} of {root} has a {kind!r} transform; "
                    f"only scale, translation and identity are supported"
                )
        u = np.asarray(units[-nd:] if len(units) >= nd else [1.0] * nd)
        out.append({
            "path": str(ds["path"]),
            "shape": [int(n) for n in arr.shape[-3:]],
            "scale_um": [float(v) for v in (scale * u)[-3:]],
            "translation_um": [float(v) for v in (trans * u)[-3:]],
            "dtype": str(arr.dtype),
        })
    return out


def world_to_source_um(
    fixed_grid: GridSpec,
    fixed_scale_um: Sequence[float],
    fixed_translation_um: Sequence[float],
    source_to_fixed_um: np.ndarray,
) -> np.ndarray:
    """The 4x4 map from run world mm to the source scan's physical um.

    ``fixed_grid`` is the run grid (the fixed scan's own sampling, as the run
    places it); ``fixed_scale_um`` and ``fixed_translation_um`` are the fixed
    scan's own OME transform at that sampling, so the fixed voxel ``i`` sits at
    ``scale * i + translation`` um. ``source_to_fixed_um`` maps source physical
    um to fixed physical um (the pairwise affine as it is usually estimated);
    it is inverted here.
    """
    # run world mm -> fixed voxel index
    s = float(fixed_grid.spacing_mm)
    o = np.asarray(fixed_grid.origin_mm, dtype=np.float64)
    W = np.eye(4)
    W[:3, :3] = np.eye(3) / s
    W[:3, 3] = -o / s
    # fixed voxel index -> fixed physical um
    P = np.eye(4)
    P[:3, :3] = np.diag(np.asarray(fixed_scale_um, dtype=np.float64))
    P[:3, 3] = np.asarray(fixed_translation_um, dtype=np.float64)
    T = np.asarray(source_to_fixed_um, dtype=np.float64)
    if T.shape == (3, 4):
        T = np.vstack([T, [0.0, 0.0, 0.0, 1.0]])
    if T.shape != (4, 4):
        raise ValueError(f"the pair affine must be 4x4 (or 3x4), got {T.shape}")
    return np.linalg.inv(T) @ P @ W


def build_spec(
    source_root,
    world_mm_to_source_um: np.ndarray,
    background: tuple[float, float] | None = (250.0, 0.0),
    datasets: Sequence[str] | None = None,
) -> dict:
    """The metadata an affine-read volume stores.

    ``background`` is ``(threshold, value)``: source voxels at or above the
    threshold become ``value`` before interpolation, which is how BigBrain's
    white background and its fill value both become the zero the pipeline
    treats as "nothing here". ``datasets`` restricts which pyramid datasets of
    the source may be read.
    """
    found = ome_datasets(source_root)
    if datasets is not None:
        keep = set(str(d) for d in datasets)
        found = [d for d in found if d["path"] in keep]
    if not found:
        raise ValueError(f"no datasets to read from {source_root}")
    spacings = [max(d["scale_um"]) for d in found]
    if any(np.ptp(d["scale_um"]) > 1e-6 * max(d["scale_um"]) for d in found):
        raise ValueError("anisotropic source datasets are not supported")
    M = np.asarray(world_mm_to_source_um, dtype=np.float64)
    if M.shape != (4, 4):
        raise ValueError(f"world_mm_to_source_um must be 4x4, got {M.shape}")
    return {
        "source": str(Path(source_root).resolve()),
        "datasets": found,
        "world_mm_to_source_um": M.tolist(),
        "background": None if background is None else [float(background[0]), float(background[1])],
        "dtype": found[int(np.argmin(spacings))]["dtype"],
        "convention": (
            "run-level voxel -> level grid world mm -> world_mm_to_source_um -> "
            "source physical um -> (um - translation_um) / scale_um = dataset voxel; "
            "trilinear, zero outside; background applied to the source voxels first"
        ),
    }


# --------------------------------------------------------------------------- #
# The array a level hands out
# --------------------------------------------------------------------------- #
def _pick_dataset(datasets: Sequence[dict], level_um: float) -> dict:
    """The dataset whose voxel is the largest not coarser than the level's.

    Sampling a level from a finer dataset than needed only costs reads; from a
    coarser one it blurs. If every dataset is coarser than the level (the run
    goes finer than the scan was stored), the finest is used.
    """
    fits = [d for d in datasets if max(d["scale_um"]) <= level_um * (1 + 1e-6)]
    if fits:
        return max(fits, key=lambda d: max(d["scale_um"]))
    return min(datasets, key=lambda d: max(d["scale_um"]))


def _slices(key, shape) -> tuple[list[slice], list[int]]:
    """Normalise an index into three slices, remembering integer axes."""
    if not isinstance(key, tuple):
        key = (key,)
    if any(k is Ellipsis for k in key):
        raise IndexError("Ellipsis indexing is not supported on an affine-read level")
    key = key + (slice(None),) * (3 - len(key))
    if len(key) != 3:
        raise IndexError(f"expected at most 3 indices, got {len(key)}")
    out, squeeze = [], []
    for d, k in enumerate(key):
        n = int(shape[d])
        if isinstance(k, slice):
            start, stop, step = k.indices(n)
            if step != 1:
                raise IndexError("strided reads are not supported on an affine-read level")
            out.append(slice(start, max(stop, start)))
        else:
            i = int(k)
            if i < 0:
                i += n
            if not 0 <= i < n:
                raise IndexError(f"index {k} out of range for axis {d} of length {n}")
            out.append(slice(i, i + 1))
            squeeze.append(d)
    return out, squeeze


class AffineArray:
    """One level of a volume, sampled through an affine from a source pyramid.

    Duck-types the slice of the zarr array interface the pipeline reads:
    ``shape``, ``dtype``, ``ndim``, ``chunks`` and ``__getitem__`` with slices.
    Read-only.
    """

    ndim = 3
    slab = 48
    """Output planes sampled per step, bounding the device memory a read takes."""

    def __init__(self, spec: dict, level_grid: GridSpec) -> None:
        self.spec = spec
        self.shape = tuple(int(n) for n in level_grid.shape)
        self.dtype = np.dtype(spec.get("dtype", "uint8"))
        self.chunks = ()
        self.shards = None
        self._grid = level_grid
        self.dataset = _pick_dataset(spec["datasets"], level_grid.spacing_mm * 1000.0)
        self._array = None
        self._src_shape = tuple(int(n) for n in self.dataset["shape"])
        self._matrix = self._voxel_matrix()

    # -- geometry ----------------------------------------------------------- #
    def _voxel_matrix(self) -> np.ndarray:
        """4x4: run-level voxel -> source dataset voxel."""
        g = self._grid
        W = np.eye(4)
        W[:3, :3] = np.eye(3) * g.spacing_mm
        W[:3, 3] = np.asarray(g.origin_mm, dtype=np.float64)
        P = np.asarray(self.spec["world_mm_to_source_um"], dtype=np.float64)
        sc = np.asarray(self.dataset["scale_um"], dtype=np.float64)
        tr = np.asarray(self.dataset["translation_um"], dtype=np.float64)
        V = np.eye(4)
        V[:3, :3] = np.diag(1.0 / sc)
        V[:3, 3] = -tr / sc
        return V @ P @ W

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix.copy()

    def source_box(self, origin, shape) -> tuple[np.ndarray, np.ndarray]:
        """Source voxel bounds ``(lo, hi)`` covering an output box, clipped."""
        o = np.asarray(origin, dtype=np.float64)
        s = np.asarray(shape, dtype=np.float64)
        corners = np.array(
            [[o[d] + (s[d] - 1) * bit[d] for d in range(3)] + [1.0] for bit in product((0, 1), repeat=3)]
        )
        src = (self._matrix @ corners.T)[:3]
        lo = np.floor(src.min(axis=1)).astype(np.int64) - 1
        hi = np.ceil(src.max(axis=1)).astype(np.int64) + 2
        full = np.asarray(self._src_shape, dtype=np.int64)
        return np.clip(lo, 0, full), np.clip(hi, 0, full)

    # -- reading ------------------------------------------------------------ #
    @property
    def array(self):
        if self._array is None:
            import zarr

            self._array = zarr.open_array(
                str(Path(self.spec["source"]) / self.dataset["path"]), mode="r"
            )
        return self._array

    def _source_block(self, lo, hi):
        """The source box as float32 with the background taken out."""
        key = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        raw = np.asarray(self.array[key])
        if raw.ndim > 3:  # leading singleton axes (a c or t axis of length one)
            raw = raw.reshape(raw.shape[-3:])
        block = _xp.put(raw)
        bg = self.spec.get("background")
        if bg is not None:
            thr, val = float(bg[0]), float(bg[1])
            if _xp.is_tensor(block):
                block = block.masked_fill(block >= thr, val)
            else:
                block = np.where(block >= thr, np.float32(val), block)
        return block

    def __getitem__(self, key) -> np.ndarray:
        slices, squeeze = _slices(key, self.shape)
        origin = [sl.start for sl in slices]
        shape = [sl.stop - sl.start for sl in slices]
        out = np.zeros(shape, dtype=np.float32)
        if all(n > 0 for n in shape):
            lo, hi = self.source_box(origin, shape)
            if np.all(hi > lo):
                block = self._source_block(lo, hi)
                self._sample_into(out, block, lo, origin, shape)
        if np.issubdtype(self.dtype, np.integer):
            info = np.iinfo(self.dtype)
            out = np.clip(np.rint(out), info.min, info.max)
        out = out.astype(self.dtype, copy=False)
        if squeeze:
            out = np.squeeze(out, axis=tuple(squeeze))
        return out

    def _sample_into(self, out, block, lo, origin, shape) -> None:
        M = self._matrix
        step = int(self.slab)
        if _xp.is_tensor(block):
            import torch

            from . import gpu_ops

            dev = block.device
            m = torch.as_tensor(M, dtype=torch.float32, device=dev)
            off = torch.as_tensor(np.asarray(lo, dtype=np.float32), device=dev)
            j = torch.arange(origin[1], origin[1] + shape[1], dtype=torch.float32, device=dev)
            k = torch.arange(origin[2], origin[2] + shape[2], dtype=torch.float32, device=dev)
            for z0 in range(0, shape[0], step):
                z1 = min(z0 + step, shape[0])
                i = torch.arange(origin[0] + z0, origin[0] + z1, dtype=torch.float32, device=dev)
                coords = (
                    m[:3, 0, None, None, None] * i[None, :, None, None]
                    + m[:3, 1, None, None, None] * j[None, None, :, None]
                    + m[:3, 2, None, None, None] * k[None, None, None, :]
                    + (m[:3, 3] - off)[:, None, None, None]
                )
                sampled = gpu_ops.sample(block, coords, mode="constant", cval=0.0)
                out[z0:z1] = sampled.detach().cpu().numpy()
            return

        from scipy import ndimage

        src = np.asarray(block, dtype=np.float32)
        j = np.arange(origin[1], origin[1] + shape[1], dtype=np.float64)
        k = np.arange(origin[2], origin[2] + shape[2], dtype=np.float64)
        for z0 in range(0, shape[0], step):
            z1 = min(z0 + step, shape[0])
            i = np.arange(origin[0] + z0, origin[0] + z1, dtype=np.float64)
            coords = (
                M[:3, 0, None, None, None] * i[None, :, None, None]
                + M[:3, 1, None, None, None] * j[None, None, :, None]
                + M[:3, 2, None, None, None] * k[None, None, None, :]
                + (M[:3, 3] - np.asarray(lo, dtype=np.float64))[:, None, None, None]
            )
            out[z0:z1] = ndimage.map_coordinates(
                src, coords.reshape(3, -1), order=1, mode="constant", cval=0.0,
                prefilter=False,
            ).reshape(z1 - z0, shape[1], shape[2])

    def __array__(self, dtype=None, copy=None):
        a = self[(slice(None),) * 3]
        return a if dtype is None else a.astype(dtype)

    def __repr__(self) -> str:
        return (
            f"AffineArray(shape={self.shape}, dtype={self.dtype}, "
            f"source={self.spec['source']!r}, dataset={self.dataset['path']!r})"
        )

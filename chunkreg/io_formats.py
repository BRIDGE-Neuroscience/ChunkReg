"""Reading and writing foreign formats.

Confined to ingest and export. Everything between them is a chunked array, so
nothing in the pipeline has to know that NIfTI puts the slowest axis last, that
an OME-Zarr carries time and channel axes, or that a TIFF stack might be a
directory of files.

Sources are opened lazily where the format allows it. A zarr source, plain or
OME-Zarr, is never read whole: ingest pulls one chunk core at a time, so a
subject larger than memory ingests in bounded memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .grid import GridSpec

__all__ = ["VolumeSource", "open_source", "read_volume", "write_volume", "is_zarr"]

_TIFF = {".tif", ".tiff", ".btf"}
_NIFTI = {".nii", ".mgz", ".mgh"}
# Bare ``.gz`` is deliberately absent: it would route ``stack.tif.gz`` to
# nibabel. Gzipped NIfTI is matched on the full ``.nii.gz`` suffix.

_UNIT_TO_MM = {
    "nanometer": 1e-6,
    "nm": 1e-6,
    "micrometer": 1e-3,
    "micron": 1e-3,
    "um": 1e-3,
    "µm": 1e-3,
    "millimeter": 1.0,
    "mm": 1.0,
    "centimeter": 10.0,
    "cm": 10.0,
    "meter": 1000.0,
    "m": 1000.0,
}

_SPATIAL = ("z", "y", "x")


# --------------------------------------------------------------------------- #
# A lazily read 3D volume
# --------------------------------------------------------------------------- #
@dataclass
class VolumeSource:
    """A 3D volume, indexed in ``(Z, Y, X)`` order, read on demand.

    ``spacing_mm`` is what the file itself says the voxel size is, or ``None``
    when the format does not say or says it without units. It is a single
    number for isotropic voxels and a ``(z, y, x)`` triple otherwise;
    :attr:`voxel_mm` always gives the triple.
    """

    shape: tuple[int, int, int]
    dtype: np.dtype
    spacing_mm: float | tuple[float, float, float] | None
    description: str
    _read: Callable[[tuple], Any]

    @property
    def voxel_mm(self) -> tuple[float, float, float] | None:
        """The file's voxel size as ``(z, y, x)`` millimetres, if it states one."""
        if self.spacing_mm is None:
            return None
        if isinstance(self.spacing_mm, (int, float)):
            return (float(self.spacing_mm),) * 3
        return tuple(float(v) for v in self.spacing_mm)

    def __getitem__(self, key) -> np.ndarray:
        return np.asarray(self._read(key))

    def read(self) -> np.ndarray:
        return self[(slice(None),) * 3]

    @classmethod
    def from_array(cls, a, spacing_mm: float | None = None, description: str = "array"):
        if len(a.shape) != 3:
            raise ValueError(f"expected a 3D volume, got shape {tuple(a.shape)}")
        return cls(
            tuple(int(n) for n in a.shape),
            np.dtype(a.dtype),
            spacing_mm,
            description,
            lambda key: a[key],
        )


def _spatial_reader(arr, names: Sequence[str]) -> tuple[tuple[int, int, int], Callable]:
    """Map ``(z, y, x)`` slicing onto an array with arbitrary named axes.

    Non-spatial axes (time, channel) must have length one and are indexed
    away; spatial axes are reordered to ``(z, y, x)`` whatever order the file
    stores them in.
    """
    lowered = [str(n).lower() for n in names]
    missing = [a for a in _SPATIAL if a not in lowered]
    if missing:
        raise ValueError(
            f"axes {list(names)} have no {missing} axis; a subject has to be a "
            f"3D volume with z, y and x axes"
        )
    spatial = [lowered.index(a) for a in _SPATIAL]
    for i, n in enumerate(arr.shape):
        if i not in spatial and n != 1:
            raise ValueError(
                f"axis {names[i]!r} has length {n}. A subject must be a single "
                f"3D volume; select one time point or channel and save that "
                f"as its own zarr before ingest"
            )
    in_order = sorted(spatial)
    perm = [in_order.index(i) for i in spatial]
    shape = tuple(int(arr.shape[i]) for i in spatial)

    def read(key):
        full: list[Any] = [0] * arr.ndim
        for d, i in enumerate(spatial):
            full[i] = key[d]
        return np.transpose(np.asarray(arr[tuple(full)]), perm)

    return shape, read


# --------------------------------------------------------------------------- #
# Zarr, plain or OME
# --------------------------------------------------------------------------- #
def is_zarr(path) -> bool:
    """A directory that is a zarr array or group, v2 or v3."""
    p = Path(path)
    return p.is_dir() and any(
        (p / n).exists() for n in ("zarr.json", ".zarray", ".zgroup")
    )


def _ome_spacing(
    multiscale: dict, dataset: dict, axes: list
) -> float | tuple[float, float, float] | None:
    """Voxel size in mm from OME-NGFF scale transforms, ordered ``(z, y, x)``."""
    scale = [1.0] * len(axes)
    transforms = list(dataset.get("coordinateTransformations") or [])
    transforms += list(multiscale.get("coordinateTransformations") or [])
    found = False
    for tr in transforms:
        if tr.get("type") == "scale" and "scale" in tr:
            scale = [a * float(b) for a, b in zip(scale, tr["scale"])]
            found = True
    if not found:
        return None
    sizes = {}
    for i, ax in enumerate(axes):
        name = str(ax.get("name", "")).lower() if isinstance(ax, dict) else ""
        if name not in _SPATIAL:
            continue
        unit = ax.get("unit")
        if unit is None:
            return None  # a number with no unit is not a spacing
        factor = _UNIT_TO_MM.get(str(unit).lower())
        if factor is None:
            raise ValueError(f"unrecognised OME unit {unit!r} on axis {ax['name']!r}")
        sizes[name] = scale[i] * factor
    if len(sizes) != 3:
        return None
    return _spacing(tuple(sizes[a] for a in _SPATIAL))


def _spacing(zyx: Sequence[float]) -> float | tuple[float, float, float]:
    """One number for isotropic voxels, the ``(z, y, x)`` triple otherwise.

    Anisotropic scans are accepted: ingest resamples every scan onto the run
    grid, which is isotropic, so the triple is what it needs to do that.
    """
    # Seven significant digits is what a float32 header holds; beyond that is
    # noise that would otherwise leak into the run grid's spacing.
    v = tuple(float(f"{float(x):.7g}") for x in zyx)
    if max(v) <= min(v) * (1 + 1e-3):
        return float(sum(v) / 3.0)
    return v


def _open_zarr(p: Path) -> VolumeSource:
    import zarr

    node = zarr.open(str(p), mode="r")
    if isinstance(node, zarr.Array):
        names = list(_SPATIAL) if node.ndim == 3 else (
            [f"dim{i}" for i in range(node.ndim - 3)] + list(_SPATIAL)
        )
        shape, read = _spatial_reader(node, names)
        return VolumeSource(shape, np.dtype(node.dtype), None, f"zarr array {p}", read)

    attrs = dict(node.attrs)
    ome = attrs.get("ome", attrs)
    multiscales = ome.get("multiscales") if isinstance(ome, dict) else None
    if multiscales:
        m = multiscales[0]
        dataset = m["datasets"][0]  # the first dataset is the full resolution
        arr = node[dataset["path"]]
        axes = list(m.get("axes") or [])
        if axes:
            names = [a["name"] if isinstance(a, dict) else str(a) for a in axes]
            spacing = _ome_spacing(m, dataset, axes)
        else:  # NGFF 0.3 and earlier carry no axes: assume trailing z, y, x
            names = [f"dim{i}" for i in range(arr.ndim - 3)] + list(_SPATIAL)
            spacing = None
        shape, read = _spatial_reader(arr, names)
        return VolumeSource(
            shape,
            np.dtype(arr.dtype),
            spacing,
            f"OME-Zarr {p} (dataset {dataset['path']!r})",
            read,
        )

    keys = sorted(node.array_keys())
    if len(keys) == 1:
        return _open_zarr(p / keys[0])
    raise ValueError(
        f"{p} is a zarr group with {len(keys)} arrays ({keys[:6]}) and no OME "
        f"multiscales metadata, so it is not clear which one is the subject. "
        f"Point 'source' at the array itself, e.g. {p / keys[0] if keys else p}"
    )


# --------------------------------------------------------------------------- #
# Opening any supported source
# --------------------------------------------------------------------------- #
def open_source(path) -> VolumeSource:
    """Open a subject volume for ingest, lazily where the format allows.

    Accepts a zarr array, a zarr group with one array, an OME-Zarr (the full
    resolution dataset is used), an existing chunkreg store, a multi-page
    TIFF, a directory of single-slice TIFFs, or a NIfTI/MGH file.
    """
    p = Path(path)
    if p.is_dir() and (p / "meta.json").exists():
        from .store import Volume

        v = Volume.open(p)
        k = v.n_levels - 1
        arr = v.array(k)
        return VolumeSource(
            v.native_grid.shape,
            np.dtype(arr.dtype),
            v.native_grid.spacing_mm,
            f"chunkreg store {p}",
            lambda key: arr[key],
        )
    if is_zarr(p):
        return _open_zarr(p)
    if p.is_dir():
        return VolumeSource.from_array(_read_tiff_dir(p), description=f"TIFF slices {p}")
    if not p.exists():
        raise FileNotFoundError(f"no such source: {p}")

    name = p.name.lower()
    if p.suffix.lower() in _TIFF or name.endswith((".tif.gz", ".tiff.gz")):
        import tifffile

        return VolumeSource.from_array(
            np.asarray(tifffile.imread(str(p))), description=f"TIFF {p}"
        )
    if p.suffix.lower() in _NIFTI or name.endswith(".nii.gz"):
        return _open_nifti(p)
    raise ValueError(
        f"cannot read {p}: expected a zarr or OME-Zarr, a TIFF (.tif, .tiff, "
        f".btf), a directory of TIFFs, a NIfTI or MGH file (.nii, .nii.gz, "
        f".mgz), or a chunkreg store"
    )


def _open_nifti(p: Path) -> VolumeSource:
    import nibabel as nib

    img = nib.load(str(p))
    proxy = img.dataobj
    if len(proxy.shape) != 3:
        raise ValueError(f"{p} has shape {proxy.shape}; a subject must be 3D")
    # NIfTI is (X, Y, Z); the pipeline is (Z, Y, X) throughout.
    shape = tuple(int(n) for n in proxy.shape[::-1])

    def read(key):
        return np.asarray(proxy[key[2], key[1], key[0]]).transpose(2, 1, 0)

    spacing = None
    try:
        zooms = [float(z) for z in img.header.get_zooms()[:3]]
        unit = img.header.get_xyzt_units()[0] or "mm"
        factor = _UNIT_TO_MM.get(unit, 1.0) if unit != "unknown" else 1.0
        if len(zooms) == 3 and min(zooms) > 0:
            # NIfTI zooms are (x, y, z); the pipeline is (z, y, x).
            spacing = _spacing([z * factor for z in zooms[::-1]])
    except Exception:  # noqa: BLE001 - spacing is advisory; the config can set it
        spacing = None
    return VolumeSource(shape, np.dtype(proxy.dtype), spacing, f"NIfTI {p}", read)


def read_volume(path) -> np.ndarray:
    """Read a whole 3D volume as ``(Z, Y, X)``. For small volumes only."""
    return open_source(path).read()


def _read_tiff_dir(d: Path) -> np.ndarray:
    import tifffile

    files = sorted(f for f in d.iterdir() if f.suffix.lower() in _TIFF)
    if not files:
        raise ValueError(f"no TIFF slices in {d}")
    first = tifffile.imread(str(files[0]))
    out = np.empty((len(files),) + first.shape, dtype=first.dtype)
    out[0] = first
    for i, f in enumerate(files[1:], start=1):
        out[i] = tifffile.imread(str(f))
    return out


def write_volume(path, data: np.ndarray, grid: GridSpec | None = None) -> None:
    """Write a ``(Z, Y, X)`` volume as TIFF or NIfTI, chosen by extension."""
    p = Path(path)
    a = np.asarray(data)
    suffix = p.suffix.lower()
    p.parent.mkdir(parents=True, exist_ok=True)
    if suffix in _TIFF:
        import tifffile

        tifffile.imwrite(str(p), a, bigtiff=a.nbytes > 2**31)
        return
    if suffix in _NIFTI or p.name.lower().endswith(".nii.gz"):
        import nibabel as nib

        affine = np.eye(4)
        if grid is not None:
            # Back to (X, Y, Z) for NIfTI, with the spacing on the diagonal.
            affine[0, 0] = affine[1, 1] = affine[2, 2] = grid.spacing_mm
            affine[:3, 3] = np.asarray(grid.origin_mm)[::-1]
        nib.save(nib.Nifti1Image(a.transpose(2, 1, 0), affine), str(p))
        return
    raise ValueError(
        f"cannot write {p}: use a TIFF (.tif, .tiff, .btf) or NIfTI "
        f"(.nii, .nii.gz) extension"
    )


def spacing_agrees(a, b, rel: float = 1e-3) -> bool:
    """Do two voxel sizes agree? Each is a number or a ``(z, y, x)`` triple."""
    ta = (float(a),) * 3 if isinstance(a, (int, float)) else tuple(map(float, a))
    tb = (float(b),) * 3 if isinstance(b, (int, float)) else tuple(map(float, b))
    return all(math.isclose(x, y, rel_tol=rel) for x, y in zip(ta, tb))

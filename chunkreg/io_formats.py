"""Reading and writing foreign formats.

Confined to ingest and export. Everything between them is a chunked array, so
nothing in the pipeline has to know that NIfTI puts the slowest axis last or
that a TIFF stack might be a directory of files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .grid import GridSpec

__all__ = ["read_volume", "write_volume"]

_TIFF = {".tif", ".tiff", ".btf"}
_NIFTI = {".nii", ".gz", ".mgz"}


def read_volume(path) -> np.ndarray:
    """Read a 3D volume as ``(Z, Y, X)``.

    Accepts a multi-page TIFF, a directory of single-slice TIFFs, a NIfTI file,
    or an existing chunkreg store.
    """
    p = Path(path)
    if p.is_dir():
        if (p / "meta.json").exists():
            from .store import Volume

            v = Volume.open(p)
            return v.read_padded(v.n_levels - 1, (0, 0, 0), v.native_grid.shape)
        return _read_tiff_dir(p)
    suffix = p.suffix.lower()
    if suffix in _TIFF:
        import tifffile

        return np.asarray(tifffile.imread(str(p)))
    if suffix in _NIFTI or p.name.endswith(".nii.gz"):
        import nibabel as nib

        # NIfTI is (X, Y, Z); the pipeline is (Z, Y, X) throughout.
        return np.asarray(nib.load(str(p)).dataobj).transpose(2, 1, 0)
    raise ValueError(
        f"cannot read {p}: expected a TIFF, a directory of TIFFs, a NIfTI "
        f"file, or a chunkreg store"
    )


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
    if suffix in _NIFTI or p.name.endswith(".nii.gz"):
        import nibabel as nib

        affine = np.eye(4)
        if grid is not None:
            # Back to (X, Y, Z) for NIfTI, with the spacing on the diagonal.
            affine[0, 0] = affine[1, 1] = affine[2, 2] = grid.spacing_mm
            affine[:3, 3] = np.asarray(grid.origin_mm)[::-1]
        nib.save(nib.Nifti1Image(a.transpose(2, 1, 0), affine), str(p))
        return
    raise ValueError(f"cannot write {p}: use a .tif or .nii.gz extension")

#!/usr/bin/env python3
"""
median3d.py -- isotropic 3D median filter for very large TIFF stacks.

Default radius is 1.5 pixels, which gives a 19-voxel spherical footprint
(a 3x3x3 cube with the eight corners removed, since sqrt(3) ~ 1.732 > 1.5).
Because the footprint extends only +/-1 voxel per axis, the volume can be
streamed in Z-slabs with a 1-voxel halo; peak RAM is a few slabs rather than
the whole stack.

Input  : a multi-page (Big)TIFF, or a directory of single-slice TIFFs.
Output : a BigTIFF, or a directory of single-slice TIFFs (chosen by extension).

Examples
--------
  # multi-page in, multi-page out, 8 CPU workers
  python median3d.py raw.tif filtered.tif --workers 8

  # directory of slices in, directory of slices out, GPU
  python median3d.py /data/slices/ /data/slices_med/ --gpu

  # gentler 7-voxel cross (matches ImageJ Median 3D at radius 1)
  python median3d.py raw.tif out.tif --radius 1

  # inspect the footprint without touching the data
  python median3d.py raw.tif out.tif --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import tifffile

__version__ = "1.1"

TIFF_SUFFIXES = {".tif", ".tiff", ".btf"}


def is_dir_target(path: Path) -> bool:
    """Output paths without a TIFF suffix are written as a directory of slices."""
    return Path(path).suffix.lower() not in TIFF_SUFFIXES


# --------------------------------------------------------------------------- #
# footprint
# --------------------------------------------------------------------------- #
def spherical_footprint(radius: float) -> np.ndarray:
    """Boolean (z, y, x) footprint of all voxels within `radius` pixels of the centre.

    Uses the same rule as ImageJ's Median 3D: an offset is included when
    dz² + dy² + dx² <= radius². Radius 1.5 gives the 19-voxel neighbourhood
    (3x3x3 minus the eight corners); radius 1.0 gives the 7-voxel cross.
    """
    radius = float(radius)
    if radius < 1:
        raise ValueError("radius must be at least 1 pixel")
    ext = int(math.floor(radius + 1e-9))
    ax = np.arange(-ext, ext + 1)
    zz, yy, xx = np.meshgrid(ax, ax, ax, indexing="ij")
    fp = (zz**2 + yy**2 + xx**2) <= radius**2 + 1e-9
    return fp


def describe_footprint(fp: np.ndarray) -> str:
    lines = [f"footprint shape {fp.shape}, {int(fp.sum())} voxels"]
    for k in range(fp.shape[0]):
        plane = "\n".join("    " + "".join("#" if v else "." for v in row) for row in fp[k])
        lines.append(f"  z{k - fp.shape[0] // 2:+d}:\n{plane}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


class StackReader:
    """Random Z-range access to a multi-page TIFF or a directory of slices."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.is_dir():
            self.files = sorted(
                (p for p in self.path.iterdir() if p.suffix.lower() in TIFF_SUFFIXES),
                key=_natural_key,
            )
            if not self.files:
                raise FileNotFoundError(f"no TIFF files in {self.path}")
            probe = tifffile.imread(self.files[0])
            if probe.ndim != 2:
                raise ValueError(f"expected 2D slices, got {probe.shape}")
            self.shape = (len(self.files), *probe.shape)
            self.dtype = probe.dtype
            self._tif = None
        else:
            self.files = None
            self._tif = tifffile.TiffFile(self.path)
            series = self._tif.series[0]
            shape = tuple(series.shape)
            if len(shape) != 3:
                raise ValueError(
                    f"expected a 3D stack, got shape {shape}. "
                    "Multi-channel / multi-sample TIFFs are not handled."
                )
            self.shape = shape
            self.dtype = series.dtype

    def read(self, z0: int, z1: int) -> np.ndarray:
        z0, z1 = max(0, z0), min(self.shape[0], z1)
        if self.files is not None:
            return np.stack([tifffile.imread(f) for f in self.files[z0:z1]])
        arr = self._tif.asarray(key=range(z0, z1))
        return arr.reshape((z1 - z0, *self.shape[1:]))

    def close(self):
        if self._tif is not None:
            self._tif.close()


class StackWriter:
    """Appends planes to a BigTIFF, or writes them as individual slice files."""

    def __init__(self, path: Path, n_planes: int, compression=None, bigtiff=True, names=None):
        self.path = Path(path)
        self.compression = compression
        self.as_dir = is_dir_target(self.path)
        self._index = 0
        self._pad = max(5, len(str(max(n_planes - 1, 0))))
        self._names = names  # reuse source filenames when slices came from a dir
        if self.as_dir:
            self.path.mkdir(parents=True, exist_ok=True)
            self._tw = None
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._tw = tifffile.TiffWriter(self.path, bigtiff=bigtiff)

    def write(self, block: np.ndarray):
        for plane in block:
            if self.as_dir:
                if self._names is not None:
                    out = self.path / self._names[self._index]
                else:
                    out = self.path / f"slice_{self._index:0{self._pad}d}.tif"
                tifffile.imwrite(out, plane, compression=self.compression)
            else:
                self._tw.write(
                    plane,
                    photometric="minisblack",
                    compression=self.compression,
                    contiguous=self.compression is None,
                )
            self._index += 1

    def close(self):
        if self._tw is not None:
            self._tw.close()


# --------------------------------------------------------------------------- #
# filtering
# --------------------------------------------------------------------------- #
_W = {}  # per-process worker state


def _init_worker(path, radius, mode, use_gpu):
    _W["reader"] = StackReader(Path(path))
    _W["fp"] = spherical_footprint(radius)
    _W["mode"] = mode
    _W["gpu"] = use_gpu
    if use_gpu:
        import cupy  # noqa: F401
        from cupyx.scipy.ndimage import median_filter as gmf

        _W["fn"] = gmf
        _W["cp"] = cupy
    else:
        from scipy.ndimage import median_filter as cmf

        _W["fn"] = cmf


def _filter_slab(bounds):
    """Read [z0-pad, z1+pad), filter, return the core [z0, z1) block."""
    z0, z1 = bounds
    reader, fp, mode = _W["reader"], _W["fp"], _W["mode"]
    padz = fp.shape[0] // 2
    nz = reader.shape[0]

    lo, hi = max(0, z0 - padz), min(nz, z1 + padz)
    slab = reader.read(lo, hi)
    lead, trail = z0 - lo, hi - z1  # halo actually available from real data

    # At the true volume ends, extend by edge replication so every output
    # voxel sees a full neighbourhood.
    pad_lo, pad_hi = padz - lead, padz - trail
    if pad_lo or pad_hi:
        slab = np.pad(slab, ((pad_lo, pad_hi), (0, 0), (0, 0)), mode="edge")

    if _W["gpu"]:
        cp = _W["cp"]
        out = _W["fn"](cp.asarray(slab), footprint=cp.asarray(fp), mode=mode)
        out = cp.asnumpy(out)
        cp.get_default_memory_pool().free_all_blocks()
    else:
        out = _W["fn"](slab, footprint=fp, mode=mode)

    return out[padz : padz + (z1 - z0)]


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def check_output_safe(inp: Path, out: Path, overwrite: bool) -> None:
    """Refuse anything that could damage the source dataset.

    The filter is strictly out-of-place: the input is opened read-only and the
    output must be a distinct, not-yet-populated destination.
    """
    inp, out = Path(inp).resolve(), Path(out).resolve()

    if inp == out:
        raise SystemExit("refusing to run: output path is the input path (no in-place filtering)")

    if inp.is_dir() and out.is_relative_to(inp):
        raise SystemExit(
            f"refusing to run: output {out} sits inside the input directory {inp}; "
            "written slices would be re-read as input on a later run"
        )
    if out.is_dir() and inp.is_relative_to(out):
        raise SystemExit(f"refusing to run: input {inp} sits inside the output directory {out}")

    if not out.exists():
        return

    if is_dir_target(out):
        existing = [p for p in out.iterdir() if p.suffix.lower() in TIFF_SUFFIXES]
        if existing and not overwrite:
            raise SystemExit(
                f"refusing to run: {out} already holds {len(existing)} TIFF files. "
                "Pass --overwrite, or choose a new destination."
            )
    elif not overwrite:
        raise SystemExit(f"refusing to run: {out} already exists. Pass --overwrite to replace it.")


def write_provenance(out: Path, record: dict) -> Path:
    """Drop a JSON sidecar recording exactly how the dataset was produced."""
    out = Path(out)
    target = out / "median3d_provenance.json" if is_dir_target(out) else out.with_suffix(
        out.suffix + ".median3d.json"
    )
    target.write_text(json.dumps(record, indent=2, sort_keys=True))
    return target


def auto_slab(shape, itemsize, budget_gb, workers, fp_z) -> int:
    """Z-slab depth such that each in-flight worker stays inside the budget."""
    plane = shape[1] * shape[2] * itemsize
    # input slab + output slab + scipy's internal copy, roughly 3x
    per_worker = budget_gb * 1e9 / max(workers, 1)
    depth = int(per_worker / (3.0 * plane)) - (fp_z - 1)
    return int(np.clip(depth, 1, shape[0]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Isotropic 3D median filter for very large TIFF stacks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input", type=Path, help="multi-page TIFF or directory of slices")
    ap.add_argument("output", type=Path, help=".tif for BigTIFF, otherwise a directory")
    ap.add_argument(
        "--radius", type=float, default=1.5, metavar="PX",
        help="filter radius in pixels (1.5 -> 19 voxels, 1.0 -> 7, 2.0 -> 33)",
    )
    ap.add_argument("--slab", type=int, default=0, help="Z-slab depth (0 = auto from --memory)")
    ap.add_argument("--memory", type=float, default=8.0, help="total working-memory budget, GB")
    ap.add_argument("--workers", type=int, default=1, help="CPU worker processes")
    ap.add_argument("--gpu", action="store_true", help="use CuPy (forces workers=1)")
    ap.add_argument(
        "--mode", default="nearest",
        choices=["reflect", "constant", "nearest", "mirror", "wrap"],
        help="XY boundary handling",
    )
    ap.add_argument("--compression", default=None, help="e.g. zlib, lzw, zstd (default: none)")
    ap.add_argument("--no-bigtiff", action="store_true")
    ap.add_argument(
        "--overwrite", action="store_true",
        help="permit writing into an existing file or a non-empty output directory",
    )
    ap.add_argument("--dry-run", action="store_true", help="print plan and exit")
    ap.add_argument("--version", action="version", version=f"median3d {__version__}")
    args = ap.parse_args(argv)

    check_output_safe(args.input, args.output, args.overwrite)

    fp = spherical_footprint(args.radius)
    reader = StackReader(args.input)
    shape, dtype = reader.shape, reader.dtype
    src_names = [f.name for f in reader.files] if reader.files is not None else None
    reader.close()

    workers = 1 if args.gpu else max(1, args.workers)
    slab = args.slab or auto_slab(shape, np.dtype(dtype).itemsize, args.memory, workers, fp.shape[0])

    print(describe_footprint(fp))
    print(f"volume {shape} {dtype}  ({np.prod(shape) * np.dtype(dtype).itemsize / 1e9:.1f} GB)")
    print(f"slab depth {slab}, halo {fp.shape[0] // 2}, workers {workers}, gpu {args.gpu}")
    if args.dry_run:
        return 0

    bounds = [(z, min(z + slab, shape[0])) for z in range(0, shape[0], slab)]
    writer = StackWriter(
        args.output,
        shape[0],
        compression=args.compression,
        bigtiff=not args.no_bigtiff,
        names=src_names,
    )
    t0 = time.time()
    done = 0

    try:
        if workers == 1:
            _init_worker(args.input, args.radius, args.mode, args.gpu)
            for b in bounds:
                writer.write(_filter_slab(b))
                done += b[1] - b[0]
                _progress(done, shape[0], t0)
        else:
            # bounded look-ahead so completed slabs don't pile up in RAM
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(args.input, args.radius, args.mode, False),
            ) as ex:
                pending, it = deque(), iter(bounds)
                for _ in range(workers + 1):
                    nxt = next(it, None)
                    if nxt is None:
                        break
                    pending.append((nxt, ex.submit(_filter_slab, nxt)))
                while pending:
                    b, fut = pending.popleft()
                    writer.write(fut.result())
                    done += b[1] - b[0]
                    _progress(done, shape[0], t0)
                    nxt = next(it, None)
                    if nxt is not None:
                        pending.append((nxt, ex.submit(_filter_slab, nxt)))
    finally:
        writer.close()

    elapsed = time.time() - t0
    sidecar = write_provenance(
        args.output,
        {
            "tool": "median3d",
            "version": __version__,
            "command": " ".join(sys.argv),
            "input": str(Path(args.input).resolve()),
            "output": str(Path(args.output).resolve()),
            "radius_px": args.radius,
            "footprint_shape": list(fp.shape),
            "footprint_voxels": int(fp.sum()),
            "boundary_mode": args.mode,
            "volume_shape_zyx": list(shape),
            "dtype": str(dtype),
            "slab_depth": slab,
            "workers": workers,
            "gpu": bool(args.gpu),
            "compression": args.compression,
            "elapsed_s": round(elapsed, 2),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "versions": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "tifffile": tifffile.__version__,
            },
        },
    )
    print(f"\ndone in {elapsed:.1f} s -> {args.output}")
    print(f"provenance -> {sidecar}")
    return 0


def _progress(done, total, t0, _state={}):
    el = time.time() - t0
    eta = el * (total - done) / max(done, 1)
    msg = f"{done}/{total} slices  {el:6.1f}s elapsed  {eta:6.1f}s left"
    if sys.stderr.isatty():
        sys.stderr.write("\r" + msg)
    else:
        # Batch logs: no carriage returns, and throttled so a long job does not
        # write one line per slab into the SGE log.
        last = _state.get("t", 0.0)
        if done < total and el - last < 60.0:
            return
        _state["t"] = el
        sys.stderr.write(msg + "\n")
    sys.stderr.flush()


if __name__ == "__main__":
    raise SystemExit(main())

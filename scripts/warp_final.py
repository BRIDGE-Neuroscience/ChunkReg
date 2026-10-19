"""Warp every subject of a finished run into template space, as the template saw it.

``chunkreg apply`` warps raw intensities, which is right for using a field but
not for comparing ways of averaging: the template is the mean of each subject
*normalised* through its stored ingest window and clipped to [0, 1]. This
writes exactly that, one NIfTI per subject, so any other combination of the
same inputs can be set beside the run's own template like for like.

Per subject, two files in ``--out``:

    <id>.nii.gz        normalised intensity in [0, 1], stored as uint16 x 65535
    <id>_cov.nii.gz    uint8, 1 where the warped sample came from inside the scan

The coverage mask is what separates "this subject's scan ends here" from "this
subject has dark tissue here": both read as zero in the intensity, and
averaging the first in as a zero drags every edge of partial coverage down.

The run's template at the same level is written beside them as
``template.nii.gz`` on the same uint16 scale.

``--window LO,HI`` replaces every subject's stored window for this export
only (nothing is written back to the stores). ``--window 0,256`` writes a
uint8 scan unclipped, as ``raw / 256``, so a different window can be tried
after the warp instead of re-warping for each one.

    python scripts/warp_final.py configs/atlas_v3_n10.json            # finest level run
    python scripts/warp_final.py configs/atlas_v3_n10.json --level 3 --out /path/warped_L3
    python scripts/warp_final.py configs/atlas_v3_n10.json --window 0,256 --out /path/warped_L3_raw
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from chunkreg import fields as _fields
from chunkreg import xp as _xp
from chunkreg.apply import matching_level
from chunkreg.config import load_config
from chunkreg.io_formats import write_volume_blocks
from chunkreg.store import Field, Volume

SCALE = 65535.0


class _NormalisedWarp:
    """One subject through its field, normalised, a slab of planes at a time.

    Records the coverage of each slab it is asked for into ``self.coverage``
    as a side effect, so the intensity and the mask cost one read of the field.
    """

    dtype = np.dtype(np.uint16)
    max_source_voxels = 150_000_000  # ~0.6 GB float32, several copies on a GPU

    def __init__(self, volume: Volume, field: Field) -> None:
        self._volume = volume
        self._field = field
        self._grid = field.level_grid
        self._level = matching_level(volume, self._grid)
        self._vol_shape = tuple(int(n) for n in volume.grid(self._level).shape)
        self.shape = tuple(int(n) for n in self._grid.shape)
        self.coverage = np.zeros(self.shape, dtype=np.uint8)

    def __getitem__(self, key) -> np.ndarray:
        origin = tuple(int(k.start) for k in key)
        size = tuple(int(k.stop) - int(k.start) for k in key)
        img, cov = self._warp(origin, size)
        z0, z1 = key[0].start, key[0].stop
        self.coverage[z0:z1] = cov
        return np.rint(np.clip(img, 0.0, 1.0) * SCALE).astype(np.uint16)

    def _warp(self, origin, size):
        """Warp one box, splitting it across y then x while its source is big."""
        spacing = self._grid.spacing_mm
        u = self._field.read_dense(origin, size)

        # The source box each axis can sample, from the field's own range in
        # this slab rather than its peak magnitude: a final field carries the
        # whole affine, tens of mm, and a margin that size on every side of a
        # slab of full planes is a read many times the volume. Clipped to the
        # volume, outside which every sample is zero anyway.
        lo, hi = [], []
        for d in range(3):
            umin = float(u[d].min()) / spacing
            umax = float(u[d].max()) / spacing
            a = int(np.floor(origin[d] + umin)) - 2
            b = int(np.ceil(origin[d] + size[d] - 1 + umax)) + 3
            a = min(max(a, -2), self._vol_shape[d] + 1)
            lo.append(a)
            hi.append(max(min(b, self._vol_shape[d] + 2), a + 1))
        shape = tuple(b - a for a, b in zip(lo, hi))
        offset = tuple(o - a for o, a in zip(origin, lo))

        # A field that rotates a plane spreads one slab's samples over much of
        # the volume. Halve the box, y first, until its source fits.
        if int(np.prod(shape, dtype=np.int64)) > self.max_source_voxels:
            ax = 1 if size[1] >= size[2] else 2
            if size[ax] > 16:
                del u
                h = size[ax] // 2
                o2 = list(origin); s1 = list(size); s2 = list(size)
                s1[ax] = h; s2[ax] = size[ax] - h; o2[ax] += h
                a_img, a_cov = self._warp(origin, tuple(s1))
                b_img, b_cov = self._warp(tuple(o2), tuple(s2))
                return (np.concatenate([a_img, b_img], axis=ax),
                        np.concatenate([a_cov, b_cov], axis=ax))

        block = self._volume.read_padded(self._level, lo, shape, normalise=True)
        img = _fields.warp_offset(block, offset, u, spacing)

        axes = [
            (np.arange(a, b) >= 0) & (np.arange(a, b) < lim)
            for a, b, lim in zip(lo, hi, self._vol_shape)
        ]
        inside = (
            axes[0][:, None, None] & axes[1][None, :, None] & axes[2][None, None, :]
        )
        cov = _fields.warp_offset(_xp.put(inside), offset, u, spacing)
        return _xp.get(img), _xp.get(cov) > 0.5


class _Scaled:
    """A store's level, through its intensity window, on the uint16 scale.

    A copied template holds [0, 1] values with a (0, 1) window; a template
    linked to the fixed subject's own level holds raw intensities with that
    subject's window. Reading through the window gives the same result either
    way.
    """

    dtype = np.dtype(np.uint16)

    def __init__(self, array, window=(0.0, 1.0)) -> None:
        self._a = array
        self._lo, self._hi = (float(v) for v in (window or (0.0, 1.0)))
        self.shape = tuple(int(n) for n in array.shape)

    def __getitem__(self, key) -> np.ndarray:
        a = (np.asarray(self._a[key], dtype=np.float32) - self._lo) / (self._hi - self._lo)
        return np.rint(np.clip(a, 0.0, 1.0) * SCALE).astype(np.uint16)


def finest_level(cfg) -> int:
    levels = sorted(
        int(p.name[1:])
        for p in (cfg.root_path / "levels").glob("L*")
        if p.name[1:].isdigit() and (p / "template.zarr").exists()
    )
    if not levels:
        raise SystemExit(f"no levels/L*/template.zarr under {cfg.root_path}")
    return levels[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config")
    ap.add_argument("--level", type=int, default=None,
                    help="level to warp at (default: the finest with a template)")
    ap.add_argument("--fields", choices=["final", "level"], default="final",
                    help="final: <root>/fields, with the last template motion "
                         "folded in. level: levels/L<k>/fields")
    ap.add_argument("--out", default=None, help="default: <root>/warped_L<k>")
    ap.add_argument("--subjects", default=None, help="comma-separated, default all")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--planes", type=int, default=32, help="planes per slab")
    ap.add_argument("--window", default=None,
                    help="LO,HI in stored units, replacing every subject's "
                         "stored window for this export (e.g. 0,256 for raw uint8)")
    args = ap.parse_args()
    window = None
    if args.window:
        window = tuple(float(v) for v in args.window.split(","))
        if len(window) != 2 or not window[1] > window[0]:
            raise SystemExit(f"--window needs LO,HI with HI > LO, got {args.window}")

    cfg = load_config(args.config)
    print(f"device {_xp.configure(args.device)}")
    level = finest_level(cfg) if args.level is None else args.level
    out = Path(args.out) if args.out else cfg.root_path / f"warped_L{level}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"level L{level} -> {out}")

    tmpl = Volume.open(cfg.template_path(level))
    tpath = out / "template.nii.gz"
    if not tpath.exists():
        k = tmpl.n_levels - 1
        write_volume_blocks(tpath, _Scaled(tmpl.array(k), tmpl.normalisation), grid=tmpl.grid(k),
                            plane_group=args.planes)
        print(f"  template {tmpl.grid(k).shape} written")

    ids = [s.id for s in cfg.subjects]
    if args.subjects:
        ids = [i for i in ids if i in set(args.subjects.split(","))]
    for sid in ids:
        dst = out / f"{sid}.nii.gz"
        cov = out / f"{sid}_cov.nii.gz"
        if dst.exists() and cov.exists():
            print(f"  {sid}: already written")
            continue
        fpath = (cfg.final_field_path(sid) if args.fields == "final"
                 else cfg.field_path(level, sid))
        field = Field.open(fpath)
        if tuple(field.level_grid.shape) != tuple(tmpl.grid(tmpl.n_levels - 1).shape):
            raise SystemExit(
                f"{fpath} is on {field.level_grid.shape}, the L{level} template "
                f"on {tmpl.grid(tmpl.n_levels - 1).shape}; pick --fields level "
                f"or the level the final fields belong to"
            )
        t0 = time.time()
        vol = Volume.open(cfg.subject_path(sid))
        if window is not None:
            # In memory only: set_normalisation would write it to the store.
            vol.meta["norm_lo"], vol.meta["norm_hi"] = window
        src = _NormalisedWarp(vol, field)
        write_volume_blocks(dst, src, grid=field.level_grid, plane_group=args.planes)
        write_volume_blocks(cov, src.coverage, grid=field.level_grid,
                            plane_group=args.planes)
        print(f"  {sid}: {src.shape}, coverage {src.coverage.mean():.3f}, "
              f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

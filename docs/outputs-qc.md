# Outputs and QC

## The run root

Everything a run writes is under `root`. `ℓ` is the level number (0 is full
resolution), `n` a pass number.

```
<root>/
  grid.json                     the run grid and where each scan sits on it
  calibration.json              measured by `chunkreg setup`
  subjects/<id>.zarr            one store per subject; dataset ℓ is level ℓ
  levels/
    level_order.json            records the level numbering of this root
    L<ℓ>/
      template.zarr             the template at level ℓ
      fields/<id>.zarr          each subject's field at level ℓ
      manifest_it<n>.json       the pass's task list
      pass_it<n>.json           record that pass n finished, with its numbers
      seeded.json               written once the level has its starting template
      qc/it<n>.png              QC sheet for pass n
      qc/residual_<id>.zarr     residual map, overwritten each pass
  fields/<id>.zarr              final fields, at the finest level that ran
  scratch/L<ℓ>_it<n>/           task arrays and accumulators; deleted as they are used
  pairs/<fixed>__<moving>/      the root of a `chunkreg pair` run
  logs/, slurm/                 SLURM runner only
```

`scratch` moves elsewhere if the config sets `scratch`. With a held template,
`fields/<id>.zarr` is a symbolic link to the level's field, or a copy on a
filesystem without links.

Every store is OME-Zarr 0.5 (zarr v3), so napari, neuroglancer and other
OME-Zarr viewers open subjects, templates and fields directly. Volumes have one
dataset per level. A field has axes `c, z, y, x`: three channels, the `z`, `y` and
`x` displacements in millimetres. ChunkReg's own metadata (intensity window,
provenance, where a linked scan lives) is in the group's `chunkreg` attribute.

**Fields map template (fixed) coordinates to subject (moving) coordinates.** The
warped subject is `M(x + u(x))`. For a pair, the fixed volume is the template and
the moving volume's field is `fields/moving.zarr`.

## Reading a failing level

A pass reports `d99`: one voxel in a hundred is worse than that many millimetres. It
cannot say whether those voxels are in cortex or at the edge of the scan. Two things
a run writes on its own answer that.

**A sheet per pass**, `levels/L<ℓ>/qc/it<n>.png`: template mid-slices (axial, coronal,
sagittal) above the residual. About 250 kB, so it reads over SSH:

```bash
scp host:<root>/levels/L2/qc/it3.png .
```

The residual is the part of the correspondence a pass could not explain, which is
what the stopping rule watches. The panel is the maximum over subjects, so a
region any one subject failed in shows. The colour scale is pinned to the level's
clamp, so a colour means the same millimetres from pass to pass and level to level.

- **Black** is a converged chunk, and also a chunk with no tissue: a skipped chunk
  writes zero, which is true.
- **Magenta** is a chunk that exhausted the fold retry ladder. It writes NaN, not
  zero. Magenta over anatomy means the regularisation is being asked for a
  correspondence that does not exist.

**A residual map per subject**, `levels/L<ℓ>/qc/residual_<id>.zarr`, on the
displacement lattice, overwritten each pass. It exports like any store:

```bash
chunkreg export levels/L2/qc/residual_s01.zarr residual_s01.nii.gz
```

At fine levels the map is large (about 200 GB at 20 µm). `retention.residual_map_max_voxels`
drops the maps above a size; the sheet then says no map was written. A sheet or
map that cannot be written is reported and never stops the run.

## Applying and exporting

```bash
chunkreg apply subjects/s01.zarr fields/s01.zarr s01_in_template.zarr
chunkreg export levels/L0/template.zarr template.nii.gz
```

`apply` warps a volume through a field, streaming, into a new store. `export` writes a
store's level (default 0, full resolution) to NIfTI or TIFF. `--level N` pulls a coarser
level of a subject store.

Results sit on the run grid, which is chosen for the cohort. `--on STORE` puts them
back on the lattice a scan arrived on, anisotropic voxels included:

```bash
chunkreg apply subjects/moving.zarr fields/moving.zarr moving_in_fixed.zarr --on subjects/fixed.zarr
chunkreg export moving_in_fixed.zarr moving_in_fixed.nii.gz --on subjects/fixed.zarr
```

For `apply` the warp and the resample are one streaming pass. A warm-start subject's
store reads through its affine, so `apply` gives the moving scan on the fixed lattice
from the native data in one interpolation. At 20 µm that is about 411 G voxels through
one GPU: hours.

`export` reads a group of planes at a time. A TIFF is written straight out. A NIfTI is
assembled through a memory map beside the output first, so it needs the space on disk
twice for the duration. NIfTI stores each dimension in 16 bits, so an axis over 32,767
voxels is refused rather than truncated: write a TIFF or keep the OME-Zarr.

In Python, `PairResult.warp_moving(out)` and `export_moving(out)` do the same for a
registered pair ([API](api.md)).

## `scripts/warp_final.py`

`apply` warps raw intensities. To compare ways of averaging, the subjects have to go
into template space as the template saw them: normalised through their stored window
and clipped to [0, 1]. `warp_final.py` writes exactly that, one file per subject:

```bash
python scripts/warp_final.py run.json                      # finest level that has a template
python scripts/warp_final.py run.json --level 3 --out warped_L3
python scripts/warp_final.py run.json --window 0,256 --out warped_raw
```

| Output (in `--out`, default `<root>/warped_L<ℓ>`) | Contents |
|---|---|
| `<id>.nii.gz` | normalised intensity in [0, 1], stored as uint16 x 65535 |
| `<id>_cov.nii.gz` | uint8, 1 where the warped sample came from inside the scan |
| `template.nii.gz` | the run's template at that level, on the same scale |

The coverage mask separates "this scan ends here" from "this subject has dark tissue
here"; both read as zero in the intensity.

| Option | Meaning |
|---|---|
| `--level N` | Level to warp at. Default: the finest with a template. |
| `--fields MODE` | `final` (default): `<root>/fields`, with the last template motion folded in. `level`: `levels/L<ℓ>/fields`. |
| `--subjects a,b` | Only these subjects. Subjects already written are skipped. |
| `--window LO,HI` | Replace every subject's stored window for this export only. `0,256` gives a uint8 scan unclipped. |
| `--device`, `--planes N` | Compute device (default `auto`); planes per slab (default 32). |

`sge/warp_final.qsub` runs it on Grid Engine, on CPUs only.

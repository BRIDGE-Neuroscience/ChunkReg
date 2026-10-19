# Changelog

## 0.2.0 — 2026-10-20

Levels are numbered as OME-Zarr numbers its datasets: **level 0 is full
resolution**, the coarsest level is K. Config keys (`caps`, `level_stages`,
`level_params`, `anneal`, `lattice_factor`), `--stop-at`/`--from-level`, and
`levels/L<l>` on disk all use that number; `level0_stages` and
`level0_max_disp_frac` became `coarsest_stages` and `coarsest_max_disp_frac`
(the old names are refused with an explanation). Run roots made by 0.1 are
refused on resume.

- A moving scan can stay in its own frame: `subjects[].warm_start.affine_um`
  (moving physical µm → fixed physical µm) makes its store an affine view of
  the scan's own pyramid, sampled on demand (`chunkreg.affine_view`); nothing
  is resampled on disk. `warm_start.field` seeds the coarsest level.
- An OME-Zarr scan whose pyramid matches the run's is linked level by level
  after each level is checked to be a 2×2×2 mean of the one below.
- A template held fixed is linked to the subject's level above 2³⁰ voxels
  instead of being copied as float32; the blend writes no template
  accumulators for it; its settled field is a link to the level's field.
- anatomix sliding windows are anchored to the run grid, so a voxel's features
  are the same in every chunk that holds it.
- `levels.stop.plateau` ends a level once its convergence signal stops
  falling; `levels.lattice_factor` sets the displacement lattice per level.
- Task arrays are deleted as soon as their last blend has read them;
  `scratch` moves them to node-local disk; `retention.residual_map_max_voxels`
  drops the residual QC map at huge levels.
- NGFF 0.6 stores (coordinateSystems, sequence transforms) are read by ingest.
- `scripts/qc_levels.py`: one QC figure per finished level. Docs site under
  `docs/` (MkDocs).
- A chunk's fold check is judged over its core, not its padded box: a fold
  in the halo, where the blend's weight is zero, no longer makes a sound chunk
  emit its seed. A pass has at least as many register tasks as the runner has
  workers, so small levels use every GPU.

## 0.1.0

First release: cohort template and pairwise registration over fixed-size
chunks, FireANTs on anatomix features, multi-GPU and SLURM runners.

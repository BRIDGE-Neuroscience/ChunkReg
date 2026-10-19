# Concepts

## The run grid

Every subject is stored on one isotropic, axis-aligned grid: the **run grid**.
It is the finest level of the pyramid. Subjects may arrive with different
shapes, fields of view and voxel sizes; `chunkreg setup` resamples each one onto
the run grid, one block at a time. By default the grid is the smallest box that
holds every scan, at the finest voxel in the cohort.

To register onto one particular volume, name it in `grid.reference`. Its own
shape and voxel size become the run grid, so the result lands on the lattice you
expect and a finer moving scan does not drag the whole run to its resolution.
Arrays are ordered `(z, y, x)` throughout.

## The pyramid and level numbering

Levels are numbered as OME-Zarr numbers its datasets.

- **Level 0 is full resolution** (the run grid). Level `ℓ` is dataset `ℓ` of every
  subject store.
- The spacing of level `ℓ` is the native spacing times `2^ℓ`.
- **Level `K` is the coarsest**: the first level where the whole volume fits in
  one chunk, `K = ceil(log2(max(shape) / core))`.
- The loop runs coarse to fine: `K, K-1, ..., 0`.

A worked example: a 20 µm volume of 7,500³ voxels (150 mm), core 256.

| Level `ℓ` | Spacing | Shape | Chunks | Clamp | Role |
|---|---|---|---|---|---|
| 5 (`K`) | 640 µm | 235³ | 1 | 37.6 mm | one chunk; moments, rigid, affine, then deformable |
| 4 | 320 µm | 469³ | 8 | 3.84 mm | seeded from level 5 |
| 3 | 160 µm | 938³ | 64 | 1.92 mm | |
| 2 | 80 µm | 1875³ | 512 | 0.96 mm | |
| 1 | 40 µm | 3750³ | 3,375 | 0.48 mm | |
| 0 | 20 µm | 7500³ | 27,000 | 0.24 mm | final fields |

The clamp is 12 voxels of the level for the default profile (see
[Chunks, halos and the clamp](#chunks-halos-and-the-clamp)). Level 5 is one
chunk with no halo to bound it, so it is bounded by `coarsest_max_disp_frac`
(0.25) of the field of view instead. `chunkreg setup` prints this table for your
run as the resolved pyramid.

The number means the same thing everywhere:

| Where | How `ℓ` appears |
|---|---|
| Config | keys of `level_params`, `level_stages`, `anneal`; entries of `run` and `sharpen_laplacian_levels`; integer `stop_at` |
| `caps` | a list from full resolution outward: `caps[0]` is level 0, the last value repeats for deeper pyramids |
| CLI | `--stop-at 2` stops at level 2 (four times native spacing); `--from-level 2` redoes levels 2, 1 and 0 |
| Run root | `levels/L<ℓ>/`, scratch `L<ℓ>_it<n>/`, QC titles, logs |

A spacing such as `"200um"` always works in place of a number. A run root made by
a ChunkReg that numbered levels from the coarsest is refused, not resumed, and
so is a config that still uses `level0_stages` or `level0_max_disp_frac` (now
`coarsest_stages` and `coarsest_max_disp_frac`). Their level numbers would be
read against the wrong resolutions.

## Chunks, halos and the clamp

One chunk geometry is used at every level. That is what gives one memory
footprint and one resource request from the coarsest level to the finest.

- The **core** is the block a chunk owns (256³ by default). One core is one shard,
  which is one file, so parallel workers never write the same file.
- The **halo** is context read around the core (48 voxels by default). The
  **padded chunk** is core plus halo on every side: 352³. The registration of one
  (subject, chunk) pair runs on the padded chunk, as one GPU task.
- The **clamp** is the largest displacement a chunk may solve for. It is whatever
  the halo has left after paying for the similarity kernel and the feature
  extractor.

The halo budget, in voxels of the level:

```
halo  >=  D_max  +  s_max * [ (k - 1)/2 + 3 * max(sigma_g, sigma_w) ]  +  r_f
```

`k` is the similarity kernel width, `sigma_g` and `sigma_w` the smoothing
sigmas, `s_max` the coarsest in-chunk scale, `r_f` the feature receptive field.
The halo is fixed by the profile, so the formula is read backwards to give the
clamp:

```
D_max = halo - s_max * [ (k - 1)/2 + 3 * max(sigma_g, sigma_w) ] - r_f
```

For profile `a16` (halo 48, `k` 7, sigmas 1.0 and 0.5, scales `[2, 1]`, `r_f` 24):
`48 - 2 * (3 + 3) - 24 = 12` voxels. So the clamp is 12 voxels at every level, and
a level's job is to leave a residual that fits under the next level's clamp.

The config is rejected if the profile leaves a clamp under 8 voxels. A seeded
stage that uses a deeper in-chunk pyramid than the profile's `scales` is
rejected. One that needs more halo through a wider kernel or more smoothing is
rejected when it leaves no clamp at all and warned about otherwise. Per-level
tuning cannot overspend the halo unnoticed.

| Profile | Channels | Features | Peak memory per padded chunk |
|---|---|---|---|
| `a16` | 16 | anatomix | about 58 GB; an 80 GB card |
| `a8` | 8 | anatomix, PCA | about 29 GB |
| `a4` | 4 | anatomix, PCA | about 15 GB |
| `i1` | 1 | raw intensity | about 4 GB; clamp 36 voxels, no receptive field to pay for |

Chunks are blended back together with a Hann window that is 1 over the core and
tapers to 0 across the halo, so where two chunks overlap the field is a convex
combination of both.

## Displacement fields

A field is a `(3, Z, Y, X)` array of displacements in **millimetres**, with the
components in array-axis order `(z, y, x)`. It maps **template (fixed)
coordinates to subject (moving) coordinates**: the warped subject is
`M(x + u(x))`. Because the vectors are physical, resampling a field to another
level changes only its lattice, never its values. Fields are stored as float16 on
a lattice coarser than the level by the profile's `lattice_factor` (default 2).

## The loop

Each level repeats a pass until its stopping rule fires:

| Step | What it does | Parallel over |
|---|---|---|
| **register** | For each (subject, chunk): read the template and the subject warped by its current field, extract features, run FireANTs, clamp, fold-check, write a task array. Chunks with less foreground than `min_tissue_fraction` write their seed unchanged. | (subject, chunk) |
| **blend** | Combine the task arrays that reach into each core with Hann weights; smooth; write the field shard. | cores |
| **update** | Average the warped subjects into a new template; recentre it by the mean displacement; sharpen at the coarsest levels. Skipped when the template is held. | cores |
| **promote** | After the level ends, upsample its template and fields to seed level `ℓ-1`. | cores |

On a multi-GPU run, register and blend are dispatched together: a core's blend
starts as soon as the chunks reaching into it are registered, rather than waiting
for the whole register pass.

Level `K` starts from the voxelwise mean of the subjects (or the held subject).
Its stages (`coarsest_stages`) are moments, rigid, affine, then a deep greedy
pyramid. The linear part is folded into the displacement field, so everything
downstream sees only vector zarr. Levels `K-1` to 0 run `seeded_stages`:
`greedy` at scales `[2, 1]`.

Why the run starts at `K`: a deformation wider than a chunk looks like a
translation inside it. Only a level whose chunk covers the deformation can
resolve it, and chunk width doubles with each level up.

## Stopping rules

Each level has a pass cap (`caps`) and a floor of one pass. Before the cap, the
rule depends on whether the template is estimated or held. `d99` is the 99th
percentile (`stop.percentile`) of the residual in millimetres: the part of the
correspondence the pass could not already explain.

| Template | Level `ℓ` stops when | Config |
|---|---|---|
| **Estimated** (cohort) | `d99` < `residual_frac` x the next finer level's clamp, **and** the recentring step is under `ubar_vox` voxels. At the last level that runs there is no next level, so the recentring step alone. | `stop.residual_frac`, `stop.ubar_vox` |
| **Held** (`template_subject`, pairs) | the residual is sub-voxel: `d99` < `residual_vox` x spacing | `stop.residual_vox` |
| | or it fits the next finer level's halo (off if `require_subvoxel`) | `stop.require_subvoxel` |
| | or it has plateaued: `d99` fell by less than `rel` over `passes` passes, counted only after the level's `anneal` schedule has run | `stop.plateau` |
| | or every tissue chunk gave up on folds | |
| Either | the cap is reached | `caps` |

A held template never moves, so there is no recentring step, and the test that
would use it passes vacuously. The residual is what converges instead.

Use `require_subvoxel` and `plateau` for weak or cross-modal similarity, where
`d99` stays large long after the alignment has stopped improving: the first stops
a level fitting the next halo too early, the second stops it running to its cap.

## Template held fixed or estimated

| | Estimated (cohort) | Held (`template_subject`) |
|---|---|---|
| Template | mean of the warped subjects, recentred each pass | the named subject's own data at every level |
| Fields | one per subject | one per subject except the held one |
| Update pass | runs | skipped; nothing to unbias against a particular anatomy |
| Converges on | template stops moving, and the residual fits the next halo | the residual itself |
| Used for | group template | registration onto a volume; `chunkreg pair`; `chunkreg.register` |

The recentring step subtracts a fraction (`shape_update_step`, 0.25) of the mean
displacement from the template and every seed, so group shape bias decays as
0.75 per pass rather than accumulating.

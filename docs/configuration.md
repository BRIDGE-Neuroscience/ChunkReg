# Configuration

A config is JSON or YAML (JSON by `.json` extension, YAML otherwise). Unknown
keys are an error, as is a misspelt tuning key inside a block. Defaults below are
those of `chunkreg/config.py`. Level numbers: 0 is full resolution, the highest is
the coarsest (see [Concepts](concepts.md#the-pyramid-and-level-numbering)).

A minimal cohort config:

```json
{
  "root": "/scratch/cohort",
  "subjects": [{"id": "s01", "source": "raw/s01.ome.zarr"},
               {"id": "s02", "source": "raw/s02.ome.zarr"}],
  "runner": "local",
  "profile": "a16"
}
```

## Top level

| Key | Default | Meaning |
|---|---|---|
| `root` | `.` | Where everything is written. |
| `subjects` | required | List of subjects, below. |
| `spacing_mm` | none | Voxel size of sources that do not state one. A disagreeing file stops `setup`. |
| `template_subject` | none | Hold this subject fixed as the template. It gets no field. |
| `seed` | 0 | Seeds the per-pass channel draw. |
| `backend` | `zarr` | `zarr`, or `memory` (tests). |
| `channels`, `backbone` | | Older spellings. `backbone` picks `anatomix`, `anatomix-dev` or `anatomix-dev-vit`; prefer `profile`. |

## `subjects[]`

| Key | Default | Meaning |
|---|---|---|
| `id` | required | Subject name; appears in every path. |
| `source` | none | OME-Zarr, plain zarr, TIFF stack or folder, or NIfTI to ingest. |
| `path` | `subjects/<id>.zarr` | Where the store goes, relative to `root`. |
| `spacing_mm` | file's own | Number or `[z, y, x]`. Overrides the file. |
| `warm_start` | none | Read this scan in its own frame through a known affine. See [Large scans](large-scans.md#warm_start-the-moving-scan-in-its-own-frame). |
| `warm_start.affine_um` | | 4x4 or 3x4 matrix, or a JSON file holding one: moving physical µm to fixed physical µm. Needs `source`, and a fixed subject named by `grid.reference` or `template_subject`. |
| `warm_start.background` | none | `[threshold, value]`: voxels at or above the threshold become `value` (BigBrain: `[250, 0]`). |
| `warm_start.field` | none | A ChunkReg field store, in the run grid's frame, that seeds the coarsest level. |

## `grid`

| Key | Default | Meaning |
|---|---|---|
| `spacing_mm` | `finest` | Finest level's voxel: a number in mm, `finest` (smallest voxel in the cohort) or `coarsest`. |
| `shape` | none | `[z, y, x]`. Default: the smallest box holding every scan; bigger scans are cropped. |
| `align` | `centre` | `centre` or `corner`: how each scan is placed on the grid. |
| `reference` | none | Subject whose own shape and voxel size are the grid. Not with `spacing_mm` or `shape`. |

## Profile and features

| Key | Default | Meaning |
|---|---|---|
| `profile` | `a16` | Preset: `a16` (16 anatomix channels), `a8`, `a4` (PCA-reduced), `i1` (raw intensity). All core 256, halo 48. |
| `profile_overrides` | none | Any profile field: `core`, `halo`, `inner_chunk`, `lattice_factor`, `r_f`, `k`, `sigma_g`, `sigma_w`, `scales`, `taper`. Core is a multiple of `inner_chunk` and `lattice_factor`; `halo` of `lattice_factor`; `k` odd. |
| `features.mind` | on for anatomix | Add the 12 MIND-SSC channels. |
| `features.sample_channels` | profile's count | Channels registered per pass, drawn anew each pass; `null` registers all (28 with MIND). |
| `features.anatomix_window` | `null` | Run anatomix in N-voxel windows (multiple of 32, at least 64). Windows are anchored to the run grid. |

`profile_overrides: {"halo": 64, "scales": [4, 2, 1]}` is the 80 GB "deep" variant: a
clamp of 16 voxels with a deeper in-chunk pyramid (`configs/profiles.yaml`).

## `levels`

| Key | Default | Meaning |
|---|---|---|
| `run` | `all` | `all`, `listed` (coarsest plus every tuned level), or a list of levels. The coarsest always runs. |
| `stop_at` | none | Finest level to run: a number or a spacing such as `"200um"`. |
| `caps` | `[2, 3, 4, 6, 8]` | Maximum passes per level, listed from level 0 outward; the last value repeats. |
| `coarsest_stages` | moments, rigid, affine, greedy `[8,4,2,1]` | Stages of the coarsest (single-chunk) level. |
| `seeded_stages` | greedy `[2,1]`, 50 and 30 iterations | Stages of every other level. |
| `level_stages` | none | `{"<ℓ>": [stages]}`: replace a level's whole stage list. |
| `level_params` | none | `{"<ℓ>": {...}}`: tune a level's stages. Settable: `scales`, `iterations`, `lr`, `translation_lr`, `smooth_grad_sigma`, `smooth_warp_sigma`, `cc_kernel`, `loss`, `tolerance`. |
| `anneal` | none | `{"<ℓ>": {"passes": N, "smooth_grad_sigma": [a, b], "smooth_warp_sigma": [a, b]}}`: smoothing falls geometrically from `a` to `b` over N passes, then holds. |
| `lattice_factor` | profile's (2) | `{"<ℓ>": f}`: store level ℓ's field on a lattice `f` times coarser than the level (`1` = every voxel). Worth it at the small coarse levels; must divide the core and halo. |
| `shape_update_step` | 0.25 | Fraction of the mean displacement removed from the template each pass. |
| `sharpen_laplacian_levels` | `null` | Levels whose template is sharpened. `null` is the two coarsest. |
| `min_tissue_fraction` | 0.02 | A chunk with less foreground emits its seed unchanged. |
| `coarsest_max_disp_frac` | 0.25 | Clamp of the single-chunk level, as a fraction of the field of view. |

A stage is `"moments"`, `{"greedy": {...}}` or `{"kind": "greedy", ...}`. Kinds:
`moments`, `rigid`, `affine`, `greedy`, `syn`. Stage keys are the settable ones above
plus `gradient_checkpointing` (default on). Defaults: `scales [2,1]`,
`iterations [50,30]`, `loss cc`, `cc_kernel 7`, `lr 0.5`, `smooth_grad_sigma 1.0`,
`smooth_warp_sigma 0.5`, `tolerance 1e-6`. Scales run coarse to fine and match
`iterations` in length.

### `levels.stop`

| Key | Default | Meaning |
|---|---|---|
| `residual_frac` | 0.5 | Estimated template: stop when `d99` is below this fraction of the next level's clamp. |
| `ubar_vox` | 1.0 | Estimated template: and the recentring step is under this many voxels. |
| `residual_vox` | 0.5 | Held template: stop when `d99` is under this many voxels. |
| `percentile` | 99.0 | Which percentile of the residual is `d99`. |
| `require_subvoxel` | false | Held template: do not stop merely because the residual fits the next halo. |
| `plateau` | none | `{"passes": N, "rel": 0.05, "after": null}`. Stop when the level's convergence signal (`d99` for a held template, the recentring step for an estimated one) fell by less than `rel` over `N` passes, counted after the `anneal` schedule (or from pass `after`). |

## `ingest`

| Key | Default | Meaning |
|---|---|---|
| `dtype` | `auto` | Storage type. `auto` keeps integers and stores floats as float32. |
| `median_radius` | none | Optional median denoise, in voxels. A scan with one is always copied. |
| `percentiles` | `[0.5, 99.5]` | Intensity window, measured at the coarsest level. |
| `link` | true | Read a zarr scan already on the run grid in place; `false` copies it. |

## `retry`, `retention`, `scratch`

| Key | Default | Meaning |
|---|---|---|
| `retry.fold_frac` | 0.001 | Fraction of folded voxels that sends a chunk down the retry ladder. |
| `retry.ladder` | `sigma_w_x2`, `clamp_x0.5`, `emit_seed` | Steps tried in order on a fold or divergence. |
| `retention.delete_task_arrays` | true | Delete task arrays once blended (early on multi-GPU runs). |
| `retention.delete_accumulators` | true | Delete the template accumulators after the update pass. |
| `retention.residual_map_max_voxels` | none | Keep a level's residual map only if its lattice has at most this many voxels. |
| `retention.keep_history_levels` | `null` | Accepted; the pipeline does not currently read it. |
| `scratch` | `<root>/scratch` | Directory for task arrays and accumulators, absolute or relative to `root`. |

## Compute

| Key | Default | Meaning |
|---|---|---|
| `device` | `auto` | `cuda`, `cuda:N`, `cpu`, `torch-cpu` (testing) or `auto` (a GPU if visible). `cuda` refuses to start without one. |
| `engine` | `auto` | `fireants`, `demons` (CPU reference), or `auto`. |
| `gpus` | `all` | GPUs a local run uses: `all` or a count. |
| `workers_per_gpu` | 1 | Workers per GPU, or `auto` (2 if two tasks fit in 80% of `gpu_mem_gb`). |
| `gpu_mem_gb` | 80 | Device memory the planner assumes. |
| `bytes_per_voxel_channel` | 90 | Memory model; `setup` measures it. |
| `throughput_ch_vox_per_s` | 1.5e6 | Speed model; `setup` measures it. |

## `runner` and `slurm`

| Key | Default | Meaning |
|---|---|---|
| `runner` | `local` | `local` (this process, GPU workers on this node) or `slurm`. |
| `slurm.partition` | `gpu` | Partition. |
| `slurm.account` | none | Account. |
| `slurm.poll_seconds` | 30 | How often `sacct` is asked for array state. |
| `slurm.max_wait_s` | none | Fail instead of waiting longer than this for a pass. |
| `slurm.register`, `.blend`, `.update` | see below | Per-pass resources: `gpus`, `cpus`, `mem_gb`, `time`, `chunks_per_task` (16). |

Defaults: `register` 1 GPU, 8 CPUs, 64 GB, 04:00:00; `blend` 0 GPUs, 8 CPUs, 32 GB,
02:00:00; `update` 0 GPUs, 8 CPUs, 32 GB, 01:00:00.

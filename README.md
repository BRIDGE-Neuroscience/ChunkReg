# chunkreg

Co-registers a cohort of 3D volumes that are too large for one GPU. The output
is an unbiased group template and one displacement field per subject.

How it works is in [docs/PLAN.md](docs/PLAN.md). This page is how to use it.

## Install

```bash
pip install -e ".[store,io]"        # core, zarr, TIFF/NIfTI readers
pip install -e ".[gpu]"             # GPU path: torch, MONAI, SimpleITK
pip install git+https://github.com/neel-dey/anatomix.git
bash anatomix/registration/registration_backend/install_fireants.sh
```

The CPU reference engine needs only the first line. On Windows ARM64 there is
no `numcodecs` wheel, so use an x64 Python for zarr.

## Quick start

1. Copy [configs/example.json](configs/example.json) and list your subjects.
2. Prepare everything and check the plan:

   ```bash
   chunkreg setup run.json
   ```

3. Run it:

   ```bash
   chunkreg run run.json
   ```

Both commands are safe to repeat. `setup` skips subjects already converted, and
`run` resumes from the last finished pass. Use `chunkreg status run.json` to see
progress.

## Subjects

Each subject needs an `id` and a `source`:

```json
"subjects": [
  {"id": "s01", "source": "raw/s01.ome.zarr"},
  {"id": "s02", "source": "raw/s02.zarr"}
]
```

`setup` converts each source into a sharded zarr store at
`<root>/subjects/<id>.zarr`, one chunk core at a time, so a subject never has to
fit in memory. Set `path` to put the store somewhere else.

Accepted sources:

- **OME-Zarr**, v0.4 or v0.5. The full-resolution dataset is used, and the voxel
  size is read from its metadata.
- **Plain zarr**, either an array or a group holding one array.
- **TIFF** stacks or folders of slices, and **NIfTI**. These are loaded whole.

Every subject must be a single 3D volume with isotropic voxels. All subjects
must share one shape and spacing, so pad or resample them first. `setup` stops
with a clear message if any of this is not true, or if the config's
`spacing_mm` disagrees with a file's own metadata.

## Checking the plan

`setup` ends by printing three things:

- **The pyramid.** Levels halve in resolution from native until the volume fits
  one chunk. The depth is worked out for you. A 2500³ volume at 50 µm gives
  levels at 800, 400, 200, 100 and 50 µm, numbered 0 to 4.
- **The stages per level.** These are the registration settings each level will
  actually use.
- **The cost.** This is the work per level and the estimated GPU-hours.

While editing a config, skip the slow measurements:

```bash
chunkreg setup run.json --no-calibrate --no-probe
```

## Tuning each level

Coarse levels need stiff regularisation, and fine levels need freedom for
detail such as cerebellar folia. `level_params` sets parameters per level,
using the level numbers from the pyramid table:

```json
"levels": {
  "seeded_stages": [{"greedy": {"scales": [2, 1], "iterations": [50, 30]}}],
  "level_params": {
    "1": {"iterations": [60, 40],  "lr": 0.5,  "smooth_grad_sigma": 1.2, "smooth_warp_sigma": 0.7},
    "4": {"iterations": [100, 70], "lr": 0.25, "smooth_grad_sigma": 0.6, "smooth_warp_sigma": 0.3, "cc_kernel": 5}
  }
}
```

You can set `scales`, `iterations`, `lr`, `translation_lr`,
`smooth_grad_sigma`, `smooth_warp_sigma`, `cc_kernel`, `loss` and `tolerance`.
A misspelt key is an error, and a level number the pyramid lacks is reported.
To replace a level's stages entirely, use `level_stages` with the same keys.

Level 0 uses `level0_stages`, which by default are moments, rigid, affine and
then greedy. Every other level starts from `seeded_stages`.

## Other config settings

| Key | Meaning |
|---|---|
| `root` | Where everything is written. |
| `profile` | Chunk profile: `a16`, `a8`, `a4` (anatomix features) or `i1` (raw intensity). |
| `spacing_mm` | Voxel size. Optional when the sources state it. |
| `runner` | `local` or `slurm`. |
| `levels.caps` | Maximum passes per level. Levels usually stop earlier on their own. |
| `ingest.median_radius` | Optional median denoise at ingest. |
| `template_subject` | Hold one subject fixed as the target instead of building a template. |
| `slurm.*` | Partition and per-pass resources, plus `max_wait_s` to fail instead of waiting forever. |

Configs can be JSON or YAML.

## Outputs

| What | Where |
|---|---|
| Final template | `<root>/levels/L<last>/template.zarr` |
| Final fields | `<root>/fields/<id>.zarr` |
| Per-pass records | `<root>/levels/L<k>/pass_it<n>.json` |

Fields map template coordinates to subject coordinates, in millimetres. To use
them:

```bash
chunkreg apply subjects/s01.zarr fields/s01.zarr s01_in_template.zarr
chunkreg export levels/L4/template.zarr template.nii.gz
```

## Pairwise registration

```bash
chunkreg pair run.json --fixed s01 --moving s02
```

This registers one subject to another, using the same pipeline with the fixed
subject as the template.

## On a cluster

Set `"runner": "slurm"` and run `chunkreg run run.json` on a login node. Each
pass is submitted as a job array, and each element calls
`chunkreg run-task`. Edit the templates in [slurm/](slurm/) to load your
environment.

## Tests

```bash
python -m pytest tests/ -q
```

The full suite takes about 10 minutes, most of it in
`tests/test_pipeline.py`. The zarr tests are skipped when zarr is not
installed.

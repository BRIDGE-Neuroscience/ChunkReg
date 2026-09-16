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
`run` resumes from the last finished pass (see [Resuming](#resuming)). Use
`chunkreg status run.json` to see progress.

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

Every subject must be a single 3D volume. Subjects can differ in shape, field of
view and voxel size, and voxels can be anisotropic: `setup` resamples every scan
onto one run grid (see [The run grid](#the-run-grid)).

Each scan's voxel size comes from, in order:

1. The subject's own `spacing_mm`: a number, or `[z, y, x]` for anisotropic
   voxels. This overrides the file, for files whose metadata is missing or wrong.
2. The file's metadata.
3. The top-level `spacing_mm`, for files that do not state one. If a file does
   state a size and it disagrees, `setup` stops rather than guess.

```json
"spacing_mm": 0.05,
"subjects": [
  {"id": "s01", "source": "raw/s01.zarr"},
  {"id": "s02", "source": "raw/s02.nii.gz"},
  {"id": "s03", "source": "raw/s03.zarr", "spacing_mm": [0.2, 0.05, 0.05]}
]
```

## The run grid

Every subject is stored on one isotropic grid, which is the finest pyramid level.
The `grid` block chooses that grid:

```json
"grid": {"spacing_mm": "finest", "align": "centre"}
```

| Key | Meaning |
|---|---|
| `grid.spacing_mm` | The finest resolution the run can register at: `finest` (default, the finest voxel in the cohort), `coarsest`, or a number in mm. |
| `grid.shape` | Optional `[z, y, x]`. By default the grid is the smallest box that holds every scan. A scan larger than a set shape is cropped. |
| `grid.align` | `centre` (default) centres every scan in the grid. `corner` lines up their first voxels. |

Resampling depends on how each scan compares with the grid:

- **Same spacing:** the scan is copied, shifted by whole voxels.
- **Finer than the grid:** the scan is averaged down (anti-aliased), streamed
  one block at a time.
- **Coarser than the grid:** the scan is interpolated up. This adds voxels but no
  detail, and `setup` warns about it.

A scan whose voxel size divides the grid spacing is shifted by at most half a
grid voxel so its voxels line up with the grid's. A 50 µm scan on a 100 µm grid
is then an exact 2×2×2 average.

To register 50 µm scans at 100 µm, set `"grid": {"spacing_mm": 0.1}`. The stores
are then an eighth of the size, and every pyramid level is coarser by one step.

`setup` prints the run grid and how each scan reaches it, and writes both to
`<root>/grid.json`. Each store also records where its scan sits on the grid. The
grid depends on every scan, so adding a larger or finer scan later moves it.
`setup` then asks for `--reingest`.

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

## Choosing which levels run

By default every level runs. To stop early, name the finest level to run:

```json
"levels": {"stop_at": "200um"}
```

`chunkreg run run.json --stop-at 200um` does the same without editing the
config. The final fields are written at that level's resolution. Running again
with a finer `stop_at`, or none, carries on from there without redoing
anything.

`levels.run` picks an arbitrary subset:

```json
"levels": {"run": [0, "400um", "200um"]}
```

- **Stop early** by leaving out the finest levels. The example above stops at
  200 µm, and the final fields are written at that resolution.
- **Skip a level** by leaving it out of the middle. The result is passed
  straight to the next level that runs.
- **Name levels** by number, or by spacing such as `"200um"` or `"0.2mm"`.
  A spacing must match a level in the pyramid table exactly.
- **Run only tuned levels** with `"run": "listed"`. This runs level 0 plus every
  level named in `level_params` or `level_stages`.

Level 0 always runs, because it aligns the whole volume. `setup` marks skipped
levels in the pyramid table and leaves them out of the cost. `setup --stop-at`
costs a run that stops early.

## Resuming

`chunkreg run` never starts over on its own. Each finished pass writes
`levels/L<k>/pass_it<n>.json`, and each level writes `levels/L<k>/seeded.json`
once it has its starting template and fields. A second start of the same run:

- **replays** the finished passes through the stopping rule, without redoing
  them;
- **carries on** from the first pass that has no record, reusing any register
  tasks that pass had already finished;
- **seeds or promotes** a level only if that has not already been done.

So a job killed at its time limit is continued by starting it again. If the
rules have changed since, a level may add passes, but only until the next
level has been seeded from it.

To redo work on purpose, use `chunkreg run run.json --from-level N`. This
discards the progress at level N and every finer level, then runs them again.

## Other config settings

| Key | Meaning |
|---|---|
| `root` | Where everything is written. |
| `profile` | Chunk profile: `a16`, `a8`, `a4` (anatomix features) or `i1` (raw intensity). |
| `spacing_mm` | Voxel size of sources that do not state one. |
| `grid.*` | The run grid; see [The run grid](#the-run-grid). |
| `levels.stop_at` | Finest level to run; see [Choosing which levels run](#choosing-which-levels-run). |
| `runner` | `local` or `slurm`. |
| `device` | `cuda`, `cuda:N`, `cpu`, or `auto` (the default: a GPU if one is visible). |
| `engine` | `fireants`, `demons`, or `auto` (FireANTs on a GPU). |
| `gpus` | How many GPUs a local run uses. Default `all`. |
| `levels.caps` | Maximum passes per level. Levels usually stop earlier on their own. |
| `ingest.median_radius` | Optional median denoise at ingest. |
| `template_subject` | Hold one subject fixed as the target instead of building a template. |
| `slurm.*` | Partition and per-pass resources, plus `max_wait_s` to fail instead of waiting forever. |

Configs can be JSON or YAML.

## Running on GPUs

With `"device": "cuda"`, every pass runs on the GPU and the host only reads and
writes files. If no GPU is visible, the run refuses to start rather than fall
back to the CPU. `"engine": "auto"` picks FireANTs on a GPU.

With several GPUs in one job, `chunkreg run` starts one worker per GPU listed in
`CUDA_VISIBLE_DEVICES`. Set `"gpus"` to a number to use fewer. `"device": "cpu"`
keeps the NumPy reference path, and the `CHUNKREG_DEVICE` environment variable
overrides the config for one process.

Ingest still resamples on the CPU when a scan is not already on the run
spacing. It does this once per scan, one block at a time.

## Outputs

| What | Where |
|---|---|
| Final template | `<root>/levels/L<k>/template.zarr`, for the last level run |
| Final fields | `<root>/fields/<id>.zarr`, at the last level run's resolution |
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

On a Grid Engine cluster, build the Apptainer image on a Linux machine with
`bash containers/build.sh`; its `--help` explains how to pick the CUDA
version. Then set `"runner": "local"` and `"device": "cuda"`, fill in the
settings at the top of [sge/chunkreg.qsub](sge/chunkreg.qsub), and submit it
with `qsub`.

The whole run happens in that one job, with one worker per GPU it holds (set by
`-pe gpu N`). If the job hits its time limit, submit it again and the run
resumes. The script's `STOP_AT` setting stops at a chosen level:
`qsub -v STOP_AT=200um sge/chunkreg.qsub`.

## Tests

```bash
python -m pytest tests/ -q
```

The full suite takes a few minutes, most of it in
`tests/test_pipeline.py`. The zarr tests are skipped when zarr is not
installed.

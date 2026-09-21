<p align="center">
  <img src="docs/chunkreg-logo.png"
       alt="ChunkReg — chunked metadata wrapper for registration in massive volumes"
       width="440">
</p>

Registers 3D volumes that are too large for one GPU. Nothing is ever held in
memory whole: every volume is a chunked store on disk, and every pass works one
chunk at a time with a fixed memory footprint.

There are two ways to run it, and they are the same pipeline underneath:

| | What you give it | What you get |
|---|---|---|
| [**Registration**](#registration) | two volumes, a fixed and a moving one | the moving volume on the fixed one's own lattice, plus the displacement field |
| [**Coregistration**](#coregistration) | a cohort of volumes | an unbiased group template, plus one displacement field per subject |

Start at [Setup](#setup). How it works, rather than how to use it, is in
[docs/PLAN.md](docs/PLAN.md).

---

# Setup

## Install

```bash
pip install -e ".[store,io]"        # core, zarr, TIFF/NIfTI readers
pip install -e ".[gpu]"             # GPU path: torch, MONAI, SimpleITK
pip install git+https://github.com/neel-dey/anatomix.git
bash anatomix/registration/registration_backend/install_fireants.sh
```

The CPU reference engine needs only the first line. On Windows ARM64 there is
no `numcodecs` wheel, so use an x64 Python for zarr.

That is all [Registration](#registration) needs — it takes two file paths and
fills in everything below for you. The rest of this section is for
[Coregistration](#coregistration), and for `chunkreg pair`, which both work
from a config file.

## Subjects

Each subject needs an `id` and a `source`:

```json
"subjects": [
  {"id": "s01", "source": "raw/s01.ome.zarr"},
  {"id": "s02", "source": "raw/s02.zarr"}
]
```

`setup` makes an OME-Zarr store for each subject at
`<root>/subjects/<id>.zarr`, holding the scan at every pyramid level. Set
`path` to put the store somewhere else.

- **A zarr already exactly on the run grid is read in place.** Only the coarser
  levels are written (about 1/7 of the scan's size), and the scan itself is
  opened read-only and never changed. `setup` prints the scan's chunk layout.
  Set `"ingest": {"link": false}` to copy it instead, into one file per
  256-voxel block, if the scan's storage is slow or badly chunked.
- **Anything else is copied**, resampled onto the run grid one block at a
  time, so a subject never has to fit in memory.

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
| `grid.reference` | A subject whose own shape and voxel size *are* the grid. It is then copied onto the grid unchanged and every other scan is resampled onto it and cropped to its field of view. Cannot be combined with `spacing_mm` or `shape`, which would ask for a different grid. |

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

`grid.reference` is for registering **onto** a volume rather than building a
template from a group. The default rule takes the finest voxel in the cohort,
which for a pair means a finer moving scan drags the whole run to its
resolution and the fixed scan is interpolated up. Naming the fixed subject
keeps the run — and the result — on its lattice:

```json
"grid": {"reference": "s01"}
```

`setup` prints the run grid and how each scan reaches it, and writes both to
`<root>/grid.json`. Each store also records where its scan sits on the grid. The
grid depends on every scan, so adding a larger or finer scan later moves it.
`setup` then asks for `--reingest`.

## Checking the plan

```bash
chunkreg setup run.json
```

`setup` ingests, verifies the displacement conventions, measures, and ends by
printing three things:

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

`setup` is safe to repeat: it skips subjects already converted. It ingests four
subjects at a time, since a scan is a long read off shared storage and the
subjects are independent. `--jobs N` changes that, and `--jobs 1` ingests them
in turn.

---

# Registration

One moving volume onto one fixed volume. Neither has to fit in memory, and no
cohort or config file is involved:

```python
import chunkreg

result = chunkreg.register(
    fixed="fixed.ome.zarr",
    moving="moving.ome.zarr",
    root="work/",
    progress=print,
)
result.export_moving("moving_in_fixed.nii.gz")
```

That ingests both volumes, holds the fixed one as the target, runs the level
loop, and writes the moving volume onto the fixed one's own lattice. Call it
again with the same arguments to resume a run that stopped.

`result.field` is the displacement, on the run grid, in millimetres, mapping
fixed coordinates to moving ones. `result.warp_moving(out)` writes the warped
volume as OME-Zarr instead of a file. The named arguments are `grid`,
`profile`, `spacing_mm`, `stop_at`, `device`, `engine`, `gpus` and `workers`;
any other config key goes through as a keyword:

```python
chunkreg.register(..., levels={"caps": [4, 3, 2]}, features={"mind": False})
```

## The fixed volume's grid

By default `grid="fixed"`: the fixed volume's shape and voxel size *are* the
run grid, so it is copied onto the grid unchanged and the moving volume is
resampled onto it. This is what a cohort run cannot do, and it matters when
the two differ:

| registering 96³ at 25 µm onto 48³ at 50 µm | `grid="fixed"` | `grid="finest"` (the cohort rule) |
|---|---|---|
| the run grid | 48³ at 50 µm | 96³ at 25 µm |
| the fixed volume | copied | interpolated up, adding voxels but no detail |
| the result | on the fixed volume's lattice | on the moving volume's, at 8× the voxels |

Eight times the voxels at every level, for detail the fixed volume does not
have. `grid` also takes `"finest"`, `"coarsest"`, a spacing in millimetres, or
a whole grid block as a mapping. In a config file the same setting is
`"grid": {"reference": "s01"}`, and it works for any run, not just a pair.

## Two subjects of a cohort

When both volumes are already subjects of a config, register one onto the other
from the command line:

```bash
chunkreg pair run.json --fixed s01 --moving s02
```

This is the same pipeline with the fixed subject held as the template. It
derives a two-subject config, writes it to
`<root>/pairs/<fixed>__<moving>/config.json`, and runs there — so the pair gets
its own levels, records and fields rather than resuming from the cohort's, and
so the run reaches the same GPUs and the same cluster `chunkreg run` does.
`--stop-at`, `--from-level` and `--root` work as they do for `run`.

A config that sets `template_subject` does the same thing through
`chunkreg run`, with no derived config at all. The named subject gets no field
of its own, its data is the template at every level, and the unbiasing update
pass is skipped — there is nothing to unbias when the target is one particular
anatomy rather than a population mean.

## What a fixed target changes

Holding a volume fixed changes when a level stops. A run that estimates a
template converges when the template stops moving; a fixed template never
moves, so what converges instead is the residual — the part of the
correspondence each pass could not already explain. A level ends once that is
below `levels.stop.residual_vox` (half a voxel by default) or small enough for
the next level's halo to absorb it.

---

# Coregistration

A cohort of volumes into an unbiased group template, plus one field per
subject.

1. Copy [configs/example.json](configs/example.json) and list your subjects.
2. Prepare everything and check the plan — see [Setup](#setup):

   ```bash
   chunkreg setup run.json
   ```

3. Run it:

   ```bash
   chunkreg run run.json
   ```

Both commands are safe to repeat. `run` resumes from the last finished pass
(see [Resuming](#resuming)). Use `chunkreg status run.json` to see progress.

Each pass registers every subject to the current template, averages the warped
subjects, and recentres the template by the mean displacement so the group
shape bias decays rather than accumulates.

## Outputs

| What | Where |
|---|---|
| Final template | `<root>/levels/L<k>/template.zarr` (OME-Zarr), for the last level run |
| Final fields | `<root>/fields/<id>.zarr` (OME-Zarr), at the last level run's resolution |
| Per-pass records | `<root>/levels/L<k>/pass_it<n>.json` |

Fields map template coordinates to subject coordinates, in millimetres. To use
them:

```bash
chunkreg apply subjects/s01.zarr fields/s01.zarr s01_in_template.zarr
chunkreg export levels/L4/template.zarr template.nii.gz
```

Everything the pipeline writes is on the run grid, which is chosen for the
cohort rather than for any one scan. `--on` puts a result back on the lattice
a scan arrived on, taken from that scan's store:

```bash
chunkreg apply subjects/s02.zarr fields/s02.zarr s02_in_s01.zarr --on subjects/s01.zarr
chunkreg export s02_in_s01.zarr s02_in_s01.nii.gz --on subjects/s01.zarr
```

For `apply` the warp and the resample are one streaming pass, so the
intermediate on the run grid is never written. Anisotropic voxels are carried
in the exported file's header, where a chunkreg store could not represent them.

`export` reads a group of planes at a time rather than the whole volume. A
TIFF is written straight out that way. A NIfTI is assembled through a memory
map beside the output first, because the format is a header followed by the
whole array — the process stays bounded, but the volume passes through the
page cache and needs the space on disk twice for the duration. NIfTI also
stores each dimension in 16 bits, so an axis over 32767 voxels is refused
rather than truncated; past that size, write a TIFF or keep the OME-Zarr.

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

---

# Tuning

These apply to both kinds of run.

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

## Features: anatomix with MIND-SSC

Anatomix profiles register on anatomix's 16 learned channels together with the
12 MIND-SSC channels, as anatomix's own `anatomix+mindssc` registration does.
On a GPU, MIND-SSC is anatomix's own GPU implementation.

That is 28 channels, but by default each template pass registers a random 16
of them, so memory and per-pass cost stay what 16 channels cost. The draw:

- **changes every pass,** so over a level every channel contributes;
- **is split between anatomix and MIND** in proportion (9 and 7 of 16);
- **is the same for every chunk and every GPU** in a pass, because it depends
  only on `seed`, the level and the pass.

```json
"features": {"mind": true, "sample_channels": 16}
```

| Key | Meaning |
|---|---|
| `features.mind` | Add the MIND-SSC channels. Default: on for anatomix profiles. |
| `features.sample_channels` | Channels registered per pass. Default: the profile's count (16) when MIND is on. `null` registers all 28, about 100 GB per chunk. |

`setup` and the plan show the resulting spec, for example
`anatomix+mindssc@16`.

---

# Running it

## On GPUs

With `"device": "cuda"`, every pass runs on the GPU and the host only reads and
writes files. If no GPU is visible, the run refuses to start rather than fall
back to the CPU. `"engine": "auto"` picks FireANTs on a GPU.

With several GPUs in one job, `chunkreg run` starts one worker per GPU listed in
`CUDA_VISIBLE_DEVICES`. Set `"gpus"` to a number to use fewer. `"device": "cpu"`
keeps the NumPy reference path, and the `CHUNKREG_DEVICE` environment variable
overrides the config for one process.

A worker's GPU is idle whenever that worker is reading a chunk or writing a
task array. `"workers_per_gpu": 2` puts a second worker on each card so one
covers the other's reads, which is worth having at a low channel count and
impossible at a high one: two tasks have to fit at once. The default is 1, and
`"auto"` takes 2 only when the profile's memory estimate says both fit inside
80% of `gpu_mem_gb`. That estimate is only a model until `chunkreg setup`
measures `bytes_per_voxel_channel` on your backbone, so raise it once you have
the measured figure rather than before.

Register and blend are dispatched together rather than one after the other. A
core's blend reads the chunks whose padded boxes reach into it and nothing
else, so it starts as soon as those are registered instead of waiting for the
whole register pass. What that removes is the barrier at the end of every pass
where the GPUs go idle one by one behind the slowest task.

On a GPU, chunkreg turns on TF32 and cuDNN autotuning: the registration inner
loop is fp32 convolution, chunk shapes are fixed by the profile so a plan is
tuned once and reused, and ten bits of mantissa are far more than a
displacement field resolves. `CHUNKREG_TF32=0` and `CHUNKREG_CUDNN_BENCHMARK=0`
turn them off to bisect a numerical difference against an earlier run.

Ingest still resamples on the CPU when a scan is not already on the run grid.
It does this once per scan, one block at a time. A scan that is already on the
grid needs no resampling, only the coarser levels.

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

---

# Reference

## From Python

Everything the CLI does is a function call. For a cohort, that is
`chunkreg.load_config` plus `chunkreg.build_template`:

```python
import chunkreg

cfg = chunkreg.load_config("run.json")
print(chunkreg.plan(cfg, native_grid).format())

runner = chunkreg.get_runner(
    "multigpu", cfg=cfg, gpus=["0", "1"], device="cuda", config_path="run.json"
)
try:
    result = chunkreg.build_template(cfg, runner=runner, config_path="run.json")
finally:
    runner.close()

print(result.summary())
chunkreg.apply_field("subjects/s01.zarr", "fields/s01.zarr", "s01_in_template.zarr")
chunkreg.export_store("levels/L4/template.zarr", "template.nii.gz")
```

For two volumes it is [`chunkreg.register`](#registration), which needs no
config at all. Ingest is a function call too, so a script can prepare a run as
well as drive one:

```python
chunkreg.ingest_subjects(cfg, say=print)
chunkreg.save_config(cfg, "derived.json")
```

Also exported: `register`, `PairResult`, `register_pair`, `pair_config`,
`ingest_subjects`, `save_config`, `resample_to_scan`, `scan_view`,
`Placement`, `calibrate`, `probe`, `selftest`, `get_profile`, `RunConfig`,
`Volume` and `Field`. A multi-GPU runner needs `config_path`,
because it addresses each task by config file rather than shipping a closure to
another process. These are imported on first use, so `import chunkreg` still
costs nothing more than the grid geometry.

## Config settings

| Key | Meaning |
|---|---|
| `root` | Where everything is written. |
| `profile` | Chunk profile: `a16`, `a8`, `a4` (anatomix features) or `i1` (raw intensity). |
| `spacing_mm` | Voxel size of sources that do not state one. |
| `grid.*` | The run grid, including `grid.reference`; see [The run grid](#the-run-grid). |
| `levels.stop_at` | Finest level to run; see [Choosing which levels run](#choosing-which-levels-run). |
| `runner` | `local` or `slurm`. |
| `device` | `cuda`, `cuda:N`, `cpu`, or `auto` (the default: a GPU if one is visible). |
| `engine` | `fireants`, `demons`, or `auto` (FireANTs on a GPU). |
| `gpus` | How many GPUs a local run uses. Default `all`. |
| `workers_per_gpu` | Worker processes per GPU. Default 1; `auto` allows 2 when two tasks fit the device. |
| `levels.caps` | Maximum passes per level. Levels usually stop earlier on their own. |
| `levels.stop.*` | What "earlier on their own" means: `residual_frac` and `ubar_vox` for a run that estimates a template, `residual_vox` for one holding a subject fixed, and `percentile`. |
| `ingest.median_radius` | Optional median denoise at ingest. A scan with one is always copied. |
| `ingest.link` | Read zarr scans that are already on the run grid in place. Default `true`. |
| `features.*` | MIND-SSC and channel sampling; see [Features](#features-anatomix-with-mind-ssc). |
| `seed` | Seeds the per-pass channel draw. |
| `template_subject` | Hold one subject fixed as the target instead of building a template. That subject gets no field, its own data is the template at every level, and the unbiasing update pass is skipped. |
| `slurm.*` | Partition and per-pass resources, plus `max_wait_s` to fail instead of waiting forever. |

Configs can be JSON or YAML.

## Storage format

Every store is OME-Zarr 0.5 (zarr v3), so napari, neuroglancer and other
OME-Zarr viewers open subjects, templates and fields directly:

- **Volumes:** dataset `0` is full resolution and each further dataset halves
  it. The scale and translation of each dataset are in millimetres.
- **Fields:** one dataset with axes `c, z, y, x`. The three channels are the x,
  y and z displacements in millimetres.
- **chunkreg's own metadata** (intensity window, provenance, and where a
  linked scan lives) is in the group's `chunkreg` attribute.
- **Blocks:** every level chunkreg writes has one file (shard) per 256-voxel
  block, so parallel workers never write the same file.

Stores written by earlier versions (`meta.json` beside `s0 … sK`) still open.

## Tests

```bash
python -m pytest tests/ -q
```

The full suite takes a few minutes, most of it in
`tests/test_pipeline.py`. The zarr tests are skipped when zarr is not
installed.

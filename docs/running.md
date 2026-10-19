# Running

Every run is two commands against one config. Both are safe to repeat.

```bash
chunkreg setup run.json    # ingest, check conventions, measure, print the plan
chunkreg run run.json      # run the level loop; resumes where it stopped
```

`setup` ingests each subject that names a `source`, then prints the run grid, the
resolved pyramid (one row per level), the stages each level will use, and the cost
in GPU-hours. Read that before `run`. While editing a config, skip the slow
measurements with `--no-calibrate --no-probe`. If you add a larger or finer scan to
an existing run, the grid moves and `setup` asks for `--reingest`.

## One GPU

Set `"device": "cuda"` (or leave `auto`, which uses a GPU if one is visible).
With `cuda` the run refuses to start rather than fall back to the CPU. Pick the card
with `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=0 chunkreg run run.json
```

`"device": "cpu"` runs the NumPy reference path with the `demons` engine. It exists
for tests and small volumes. `CHUNKREG_DEVICE` in the environment overrides the
config's device for one process.

## Several GPUs on one node

`"gpus": "all"` (the default) starts one worker per GPU in `CUDA_VISIBLE_DEVICES`.
Set a number to use fewer. Nothing else changes: the same command, the same config.

```json
{"runner": "local", "device": "cuda", "gpus": "all", "workers_per_gpu": 1}
```

- `workers_per_gpu: 2` puts a second worker on each card so that one covers the
  other's reads and writes. It only fits at a low channel count: two tasks must fit
  in memory at once. `auto` takes 2 only when the memory model says both fit in 80%
  of `gpu_mem_gb`.
- Register and blend are dispatched together, and each task array is deleted as soon
  as its last blend is done.
- TF32 and cuDNN autotuning are on. `CHUNKREG_TF32=0` and
  `CHUNKREG_CUDNN_BENCHMARK=0` turn them off, to bisect a numerical difference.
  `CHUNKREG_PROGRESS=1` shows the engine's own progress.

## Grid Engine (SGE)

Grid Engine has no ChunkReg runner, so the whole run happens inside one job with
`"runner": "local"` and `"device": "cuda"`. The job starts one worker per GPU it
holds (`-pe gpu N`). [`sge/chunkreg.qsub`](https://github.com/BRIDGE-Neuroscience/ChunkReg/blob/main/sge/chunkreg.qsub)
is the template.

1. On a Linux machine build the Apptainer image: `bash containers/build.sh --help`
   explains how to pick `--cuda` from the cluster's driver version.
2. Edit the `#$` resource lines and the SETTINGS block at the top of the script:
   `SIF`, `CONFIG`, `BINDS` (every directory the config touches), and optionally
   `STAGES` (`setup run status`), `STOP_AT`, `ENGINE`, `SETUP_ARGS`, `RUN_ARGS`,
   `CHUNKREG_SRC` (a checkout to use instead of the code in the image).
3. `qsub sge/chunkreg.qsub`.

Settings can be overridden at submit time:

```bash
qsub -v STAGES=run,STOP_AT=200um sge/chunkreg.qsub
```

The script checks the GPUs and the config first, so a bad setup fails in seconds,
and exits with status 99 (requeue) if Grid Engine started the job without
allocating GPUs. A job that hits its time limit is continued by submitting it again.

## SLURM

Set `"runner": "slurm"` and run `chunkreg run run.json` on a login node. The driver
stays alive. Each pass is submitted as a job array, one element per task, chained
with `afterok` and polled with `sacct`. Each element runs `chunkreg run-task`, which
rebuilds its work from the config file, so the config must be on a shared path.

- Edit the templates in `slurm/` (`register.sbatch`, `blend.sbatch`, `update.sbatch`)
  to load your environment.
- Resources come from `slurm.register`, `slurm.blend` and `slurm.update`.
  `slurm.max_wait_s` fails a pass instead of waiting forever.
- Scripts and per-element logs go under `<root>/slurm` and `<root>/logs`.

## Two volumes

```bash
chunkreg pair run.json --fixed s01 --moving s02
```

This derives a two-subject config at `<root>/pairs/<fixed>__<moving>/config.json`,
holds `s01` as the template, and runs there, on the same GPUs or cluster `run` would
use. A config that sets `template_subject` does the same through `chunkreg run`.
From Python with two paths and no config, use `chunkreg.register` ([API](api.md)).

## Resuming

`chunkreg run` never starts over on its own. Each finished pass writes
`levels/L<ℓ>/pass_it<n>.json`, and each level writes `levels/L<ℓ>/seeded.json`
once it has its starting template and fields. A second start:

- replays the finished passes through the stopping rule without redoing them;
- continues from the first pass with no record, reusing register tasks that pass
  had already finished;
- seeds or promotes a level only if that has not been done.

So a job killed at its time limit is continued by starting it again. If the rules
changed since, a level may add passes, but only until the next level has been seeded
from it. `chunkreg status run.json` shows tasks done per level and pass, and the
ids still outstanding.

## Stopping early, and redoing

Levels are numbered from full resolution (0); see
[Concepts](concepts.md#the-pyramid-and-level-numbering). For a five-level run,
levels 0 to 4 are 50, 100, 200, 400 and 800 µm.

| To | Use |
|---|---|
| Stop at 200 µm | `chunkreg run run.json --stop-at 2` or `--stop-at 200um`, or `levels.stop_at` |
| Go finer later | run again with a smaller `--stop-at`, or none. Nothing is redone |
| Run an arbitrary subset | `"levels": {"run": [4, "400um", "200um"]}` |
| Run only tuned levels | `"levels": {"run": "listed"}`: the coarsest plus every level in `level_params` or `level_stages` |
| Redo from a level | `chunkreg run run.json --from-level 2` discards levels 2, 1 and 0, then reruns them |

The final fields are written at the finest level that ran. Skipping a middle level
hands its seed straight to the next finer one that runs. The coarsest level always
runs, because it aligns the whole volume. A spacing must match a level exactly.
`chunkreg setup --stop-at` costs a run that stops early.
`--from-level` takes a level number only, and `N` must be a level the config runs.

A run root started by a ChunkReg that numbered levels from the coarsest is refused
rather than resumed. Finish it with the version that started it, or start a new root.

# Command line

```
chunkreg [--version] <command> [options]
```

Most commands take the path of a run config. `LEVEL` is a level number (0 is full
resolution) or a spacing such as `200um` or `0.2mm`. A bare decimal such as `0.2` is
refused as ambiguous. On failure a command prints `chunkreg <command>: <message>` to
stderr and exits with status 1.

| Command | Purpose |
|---|---|
| `setup` | Ingest subjects, check conventions, measure, print the plan. |
| `run` | Run the level loop for a cohort, or a pair if `template_subject` is set. |
| `pair` | Register one subject of a config onto another. |
| `status` | Progress from the task markers. |
| `apply` | Warp a volume through a field. |
| `export` | Write a store as NIfTI or TIFF. |
| `features` | Render feature channels as RGB images. |
| `selftest` | Check the displacement conventions. |
| `run-task`, `clean` | What a scheduler array element runs; apply the retention policy. Not typed by hand. |

## `chunkreg setup CONFIG`

| Option | Meaning |
|---|---|
| `--reingest` | Re-ingest subjects whose store exists. Needed when the run grid has moved. |
| `--no-ingest` | Skip ingest. |
| `--jobs N` | Subjects ingested at once (default 4); `1` ingests them in turn. |
| `--no-selftest` | Skip the conventions check. |
| `--no-calibrate` | Skip measuring receptive field, memory and throughput. |
| `--no-probe` | Skip measuring how far apart the cohort is. |
| `--pairs N` | Pairs to probe (default 3). |
| `--gpus N [N ...]` | GPU counts to cost the run at (default `8 64`). |
| `--engine NAME` | Engine for the checks: `fireants` or `demons`. |
| `--out PATH` | Where to write `calibration.json` (default `<root>/calibration.json`). |
| `--stop-at LEVEL` | Plan as if the run stops at this level. |

## `chunkreg run CONFIG`

| Option | Meaning |
|---|---|
| `--stop-at LEVEL` | Finest level to run. Running again with a finer level carries on. Overrides `levels.stop_at`. |
| `--from-level N` | Redo level `N` and every finer level (`N-1` down to 0), discarding their progress. |
| `--workers N` | Worker threads on the CPU path (default 1). Ignored on a single GPU; multi-GPU workers come from `gpus`. |
| `--engine NAME` | Registration engine. |

## `chunkreg pair CONFIG --fixed ID --moving ID`

Same options as `run` (`--stop-at`, `--from-level`, `--workers`, `--engine`), plus
`--root DIR` for the pair's output (default `<root>/pairs/<fixed>__<moving>`). Writes the
derived config to `<root>/pairs/<fixed>__<moving>/config.json` first.

## `chunkreg status CONFIG`

Prints a table of level, pass, tasks done, whether the template exists, and fields
written, then the task ids still outstanding in the unfinished pass.

## `chunkreg apply VOLUME FIELD OUT`

| Option | Meaning |
|---|---|
| `--level N` | Volume level to read (default: the one matching the field's grid). |
| `--profile NAME` | Profile for the output store (default `a16`). |
| `--on STORE` | Resample the result onto the lattice the scan behind `STORE` arrived on. |
| `--device DEV` | `auto`, `cuda`, `cuda:N` or `cpu` (default `auto`). |

## `chunkreg export STORE OUT`

`OUT` ending in `.nii.gz`, `.nii` or a TIFF suffix sets the format.

| Option | Meaning |
|---|---|
| `--level N` | Level to write (default 0, full resolution). A template store has one level. |
| `--on STORE` | Write on the lattice the scan behind `STORE` arrived on. |

## `chunkreg features CONFIG`

Feature-channel RGB sheets per subject per level, to judge whether the features line
up before spending GPU-days.

| Option | Meaning |
|---|---|
| `--out DIR` | Output directory (default `<root>/qc/features`). |
| `--at Z,Y,X` | Box centre in world mm (default: the volume centre). |
| `--size-mm MM` | Box width, the same at every level (default: one core at native resolution). |
| `--levels L,L` | Level numbers (default all). `--subjects a,b` likewise. |
| `--plane P` | `axial`, `coronal`, `sagittal` or `all`. |
| `--extractor SPEC` | Override the profile's features, e.g. `mindssc`, `anatomix`, `intensity`. |
| `--warped` | Pull each subject through its current field. |
| `--basis B` | `shared` (columns comparable) or `per-level` (not comparable). |
| `--tile-px N` | Tile size in pixels (default 192). |

## Others

`chunkreg selftest [--engine demons]` drives known shifts through the conventions.
`chunkreg run-task CONFIG --level N --iter N --pass register|blend|update --id N [--engine NAME]`
runs one task from its manifest. `chunkreg clean CONFIG --level N --iter N` removes a
pass's transient objects.

## Environment variables

| Variable | Effect |
|---|---|
| `CUDA_VISIBLE_DEVICES` | GPUs a local run can see; one worker per entry with `gpus: all`. |
| `CHUNKREG_DEVICE` | Overrides the config's `device` for one process. |
| `CHUNKREG_TF32=0`, `CHUNKREG_CUDNN_BENCHMARK=0` | Turn off TF32 and cuDNN autotuning. |
| `CHUNKREG_PROGRESS=1` | Show the engine's own progress. |
| `CHUNKREG_TEMPLATE_LINK_MIN_VOXELS` | Size from which a held template is linked, not copied (default 2^30). |

## Scripts

| Script | Use |
|---|---|
| `scripts/warp_final.py CONFIG` | Normalised warped subjects, coverage masks and the template, as NIfTI. See [Outputs and QC](outputs-qc.md). |
| `scripts/bigbrain_s2146.py {prepare,estimate,memcheck,check80}` | The BigBrain onto S-21-46 study at 20 µm. See [Large scans](large-scans.md). |

# Notes: the 50 µm N=10 cohort run

What tonight's changes mean for `configs/atlas_v3_n10.json` (ten HiP-CT brains,
anatomix + MIND-SSC with 8 of 28 channels per pass, core 256, halo 56, estimated
template, 1600 → 100 µm), and what else would make it run better. Written
2026-10-20 against commit `fa2d4b4` and later. Confirm the level numbers
against the pyramid `chunkreg setup` prints: the notes assume six levels
(ℓ = 5 is 1600 µm, ℓ = 0 is 50 µm), which is what the migrated config encodes.

## Apply now (no quality risk) — in `configs/atlas_v3_n10_v2.json`

| Change | Why | Effect |
|---|---|---|
| `slurm.register.chunks_per_task: 1 → 10` | A register task extracts the template's features once and reuses them for every subject *in that task*. At 1 entry per task the template chunk's anatomix+MIND features are computed ten times per chunk. | ~⅓ less feature work per pass; a tenth of the task arrays and files. |
| `levels.lattice_factor: {"5": 1, "4": 1}` | The 1600 and 800 µm fields are tiny; storing them at full resolution measured +0.4 Dice points at the coarsest level of the BigBrain pair (0.973 → 0.977) where the lattice was the only difference. | Better coarse seed, no measurable cost. |
| `retention.residual_map_max_voxels: 2**30` | The residual QC map is one float32 lattice volume per subject per level: at 100 µm, 10 × 1.7 GB; at 50 µm it would be 10 × 16 GB. The cap keeps the maps to 200 µm and coarser, where they are looked at. | ~150 GB less on the shared filesystem for a 50 µm run. |
| Early task-array deletion (automatic with the multi-GPU runner) | Arrays are deleted as soon as their last blend has read them, not at the end of the pass. | Pass peak disk from "all task arrays" to a thin front. |
| `scratch` on node-local disk | Task arrays and accumulators never touch `/SAN`. UCL H100/A100 nodes expose `tscratch`; request it in the qsub (`-l tscratch=300G`) and point `scratch` at `/scratch0/$USER/atlas_n10`. All workers of a local run are on one node, which is what makes it safe. | Shared-filesystem writes per pass drop by ~1 TB (at 50 µm). |

## Worth a two-level trial before adopting

| Change | Why | How to test |
|---|---|---|
| `features.anatomix_window: 128` | With whole-chunk anatomix a voxel's features depend on the chunk that contains it; neighbouring chunks disagree about shared voxels before registering anything (measured |ΔF| 0.045 against |F| 0.22 at window 64 with the old tiling). Windows anchored to the run grid make features a function of position alone (|ΔF| 1e-5). 128 voxels is 6.4 mm of context at 50 µm. | `--stop-at 400um` with and without; compare the per-level QC figures (`scripts/qc_levels.py`) and `eval_field.py` seam ratios. Windowed extraction costs roughly twice the bare-box pass, so expect a slower register pass; `chunks_per_task: 10` recovers most of it. |
| `levels.anneal` at the coarsest level, e.g. `{"5": {"passes": 8, "smooth_grad_sigma": [1.5, 0.6], "smooth_warp_sigma": [0.8, 0.3]}}` | The first passes register against a voxelwise-mean template that is badly blurred; heavy smoothing early keeps the fields from chasing blur, lighter later lets them sharpen. The cross-modal pair converged with such a schedule where a fixed one did not. For same-modality HiP-CT the gain is unproven. | Same two-level trial; watch `step mm` per pass. |

## Code changes that would help, not done tonight

- ~~Plateau stop for an estimated template~~ — done: `levels.stop.plateau` now reads the recentring step for an estimated template, so `{"passes": 3, "rel": 0.05}` ends a cohort level once the template has stopped moving; add it to the v2 config once the caps have been seen to bind.
- ~~Generic `estimate`~~ — done: `scripts/bigbrain_s2146.py estimate` counts tissue chunks from the template subject or the first subject; `--seconds-per-chunk` from `calibration.json` gives hours per level.
- **Two workers per GPU at 8 channels.** 103 B × 8 channels × 368³ voxels ≈ 41 GB per task, so two workers need 82 GB: just over an 80 GB card. `sample_channels: 6` would fit two per card and nearly double throughput at the cost of 2 of 8 channels per pass — a trial, not a default.
- **`retention.keep_history_levels`** is parsed and never read. Either implement (delete coarse-level fields and templates once a finer level has been seeded) or drop the key.

## What tonight's run tells you about the cohort

- Whether `calibrate` on an H100 confirms ~75–90 bytes per voxel-channel for FireANTs on anatomix at a 392³ chunk (the cohort's 368³ at 8 channels is comfortably inside 80 GB either way).
- The seconds per chunk at 16 channels and (40, 20) iterations; the cohort's (150–300, 100–200) iteration schedules are 5–8× longer per chunk, which is where its time goes. If the coarse levels converge in fewer passes with the lattice-1 seed and (optionally) annealing, the caps can come down.

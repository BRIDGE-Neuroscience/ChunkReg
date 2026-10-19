# chunkreg changes: warm starts, OME-ordered levels, OME-Zarr 0.6, stopping and fixes

Plan written 2026-10-19 after registering BigBrain (histology, 20 um) onto S-21-46 (HiP-CT, 20 um), so the fine
levels can run on the 8xH100 cluster without first resampling BigBrain onto S-21-46's 20 um lattice.

## Revision of 2026-10-19 (evening): what was actually built for tonight's run

The hackathon window is one run, 01:00 to 14:00. Sections 2 (level renumbering), 3 (OME 0.6 reader) and most of 4
were taken off the critical path; what replaced them is smaller, exact, and tested against the 80 um data.

**Decisions**

- *No resample and no run-time field warm start.* The moving subject is a volume whose every level is an
  `AffineArray` (`chunkreg/affine_view.py`): each block read maps the run grid's voxel centres through one 4x4
  (run world mm -> S-21-46 physical um -> inv(T_pair) -> BigBrain dataset voxel) and samples the BigBrain dataset
  whose voxel is the largest not coarser than the level's, trilinear, zero outside, with BigBrain's white background
  (>= 250, and its 255 fill) set to 0 first. The store holds only metadata (`Volume.create(affine=spec)`), every
  level counts as linked, and nothing downstream changes: `check_volume`, `native_grid`, `build_template`, register,
  blend, promote and QC all see an ordinary pyramid on the run grid. This removes all ten shared-grid assumptions of
  Appendix A at once. At 80 um the view reproduces `BigBrain2015_on_S-21-46_linear_80um.ome.zarr` **exactly**
  (MAE 0.00, corr 1.0000 on 192^3 blocks; `scripts/bigbrain_s2146.py check80`), and the affine it derives for the
  20 um grid matches `cluster_warmstart/bigbrain_affine_runmm_to_scanmm_zyx.json` to 1e-14. The 1280 um field seed
  is not carried: re-running the coarsest level costs ~10-15 min on the cluster and avoids a frame-conversion risk.
- *The fixed store's own pyramid is linked at every level* (`store.link_pyramid`): S-21-46's datasets 0-5 have
  exactly the shapes `GridSpec.coarsened` produces (ceil, centred), so `chunkreg setup` builds nothing. Levels a
  scan lacks are still built from the level above. Preparing a 20 um run root takes ~70 s and writes no voxels.
- *A held template is linked, not copied, above 2^30 voxels* (`promote.template_from_subject`): the copy was the
  fixed level normalised to float32 -- 1.6 TB written by the driver, one core at a time, before the 20 um level could
  start. The linked template carries the subject's intensity window as its normalisation, so every read returns
  the same values the copy held. Small levels keep the copy (so `template.zarr/0` still exists for the QC scripts
  at 640/320 um).
- *Blend skips the template accumulators for a held template* (`blend.run_blend_task`, `groupwise._run_pass`):
  they feed only the update pass, which a pairwise run never runs. At 20 um that was a second read of the moving
  volume and ~1.6 TB of float32 writes per pass, deleted again by retention.
- *Anatomix windows anchored to the run grid* (`AnatomixFeatures.aligned_box`, used by `register_chunk` and
  `calibrate`). The sliding-window tiling used to start at each padded chunk's own origin, so neighbouring chunks
  tiled a shared voxel differently and disagreed about its features before registering anything: measured on
  two overlapping 192^3 boxes, mean |dF| 0.045 against |F| 0.22, max 0.50. The box is now widened to hold every
  grid-aligned window reaching into it (size - 64 a multiple of the 48 stride, which is the condition under
  which MONAI places windows at the stride), features are extracted over it and cropped back. The same
  measurement then gives mean |dF| 1e-5 (bf16 noise), at no extra time. Chunks now disagree only about the
  registration; that is what the per-chunk residual map shows, and what the blend's Hann overlap smooths.
- *Fix 1a done:* field attrs now say `z, y, x displacement in millimetres (array axis order)` with
  `component_order: ["z", "y", "x"]`. *Fix 1b done:* `levels.stop.plateau: {passes, rel, after}` -- with a
  held template, stop once d99 has fallen by less than `rel` over `passes` passes, counted only after the
  level's anneal schedule (`after` overrides); a rise counts as a plateau. *Section 3 done* for reading: the
  ingest reader (`io_formats._resolve_zarr`) now takes NGFF 0.6 axes from `coordinateSystems` and flattens
  `sequence` transforms, ignoring multiscale-level transforms into other systems. 1d (per-level lattice) remains
  deferred.
- *Section 4 as a config feature* (`subjects[].warm_start`, `config.py` / `ingest.py` / `promote.py`):
  `{"affine_um": <4x4 | 3x4 | JSON path>, "background": [250, 0], "field": <field store>}`. `affine_um` is the
  pairwise affine as estimated, moving physical um -> fixed physical um (`T_pair_zyx_um.json` works as is; the
  first 4x4/3x4 in the file is taken), which is grid-independent, so the run-mm form of the original plan is not
  needed. With it, `chunkreg setup` leaves the scan out of the run-grid decision, derives the run-mm -> source
  map from the fixed scan's OME frame, and writes an affine-read store; a plain OME source whose pyramid matches
  the run's is linked level by level (`link_pyramid`) instead of rebuilt. `field` seeds the coarsest level by
  resampling the warm field's lattice onto it (`promote._seed_from_warm_start`), refusing a field whose world box
  is not the run grid's. `scripts/bigbrain_s2146.py prepare` remains as the one-command path; the config route
  is `tests/test_warm_start.py`.
- *Chunk core:* 256 is first choice (6 levels: 640, 320, 160, 80, 40, 20 um; 26,100 chunks at 20 um, ~8,100 with
  tissue); the memory model says 72-87 GB for a 392^3 chunk at 16 channels, which may not fit an 80 GB card. The run
  root is prepared for 256 and 192 (`<root>/c256`, `<root>/c192`) and `sge/bigbrain_s2146_20um.qsub` runs
  `chunkreg setup` on c256 first -- its calibration registers one real padded chunk -- and falls back to c192
  (7 levels from 1280 um, 328^3 chunks, ~42-51 GB) if that fails. `scripts/bigbrain_s2146.py memcheck --core N`
  does the same check with no data, so it can run on the cluster while the stores download.
- *Schedule* (`scripts/bigbrain_s2146.py prepare`, by spacing): coarsest level deep pyramid (8,4,2,1)
  (240,160,100,60) annealed 3.0/2.0 -> 1.5/1.0 over 25 passes, cap 30; next level (200,100) annealed, cap 15;
  >100 um (100,60) cap 4; 80 um (80,40) cap 2; 40 um (60,30) cap 1; 20 um (40,20) cap 1. lr 1.0, smoothing 1.5/1.0,
  profile a16 + `halo 68, r_f 32, sigma_g 3, sigma_w 2`, `anatomix_window 64`, percentiles (0, 99.99),
  `require_subvoxel`. Iterations and caps for 80/40/20 um are flags.

**Time budget.** Measured locally (A6000, 640 um, core 128): 54 s per ~7.5 M-voxel chunk at (200,100) iterations,
i.e. ~0.27 G voxel-channel-iterations/s at scale 1. Scaled to a 392^3 chunk at (40,20) on an H100 (assumed 3x):
~30 s, so one 20 um pass is ~8,100 x 30 s / 8 GPUs = **8.5 h**, plus ~1 h for the 18,000 empty chunks (seed
emitted, nothing registered) and 2-3 h for 640-40 um. That is the whole window. `chunkreg setup`'s calibration prints
the real seconds per chunk for this schedule; `scripts/bigbrain_s2146.py estimate <config> --seconds-per-chunk S
--gpus N` turns it into hours per level. If it does not fit, lower `--iters-20um` (the scale-1 count dominates) or
run with `STOP_AT=40um` and continue later; the run resumes.

**Disk.** Per 20 um field store ~300 GB uncompressed float16 (level field + final field), residual QC map ~200 GB
float32 per level, task arrays ~1.2 TB uncompressed per pass (deleted after the blend). Budget >= 1 TB on the run
filesystem; everything compresses, the empties to nothing.

**Launch record (2026-10-19, pryor.cs.ucl.ac.uk).** Job 7558152, `sge/bigbrain_s2146_20um.qsub`, 2 H100 on
seymour2 (the node's GPU prolog refused 4 and 3 although SGE showed 4 free slots; the rest of the 80 GB estate was
full). Core 256 ran out of memory in calibration (78.4 of 79 GB) and the job fell back to the c192 root: 7 levels,
1280 um coarsest, 328^3 chunks, 47 GB per task. Calibration: 14.1 s per chunk at the 20 um schedule, 122 bytes
per voxel-channel, receptive field 32. Fixed = the masked `S-21-46_brain_20um.ome.zarr` already on the cluster
(all six levels linked); moving = BigBrain pulled from the public bucket with `scripts/fetch_public_gcs_zarr.sh`
(265 objects verified). The code on the cluster is main at 093b4c9 (pre-renumbering); the run root is numbered the
old way and must finish under that code.
At 640 um passes 2-3, 6 of 8 chunks exhausted the fold ladder and emitted their seed: the fold fraction was
judged over the padded box, halo included, where the registration is least constrained and the blend's weight is
zero. Changed (in main and as a one-line patch of the cluster copy, backup `register.py.bak_padded_folds`) to judge
the core only. The level had already closed at pass 4 under the old criterion (8 of 8 gave up), so the
job was resubmitted as 7558158 with `--from-level 1`, redoing 640 um and finer under the new one.
The 640 um QC figure then showed fold lines exactly along the chunk seams: the blend smoothed each core
within itself with its own edge replicated, a derivative kink at every seam. The blend now accumulates and
smooths over the core plus a margin and crops (seam ratios 1.01/1.23/1.02 -> 0.84/1.04/0.90 on the local smoke);
installed in the cluster copy too (backup `blend.py.bak_core_smoothing`) and the job resubmitted again as
7558168 from level 1. Before that restart, 320 um showed 12-13 of 27 chunks giving up on folds: lr 1.0 at the
fine levels was never run locally (the finished 80 um levels used lr 0.25); `prepare --lr-fine 0.5` is the
fallback if the smooth seed does not cure it.

**Not done tonight, still wanted:** section 2 (finished on branch `ome-level-order`, worktree `scalable-reg-ome`,
uncommitted, 520 tests passing -- review and merge after the run), section 3 (the 0.6 reader; `affine_view.
ome_datasets` parses 0.6 datasets for the two new store constructors, `io_formats` still does not), fixes 1b/1d,
the field warm start.

## Context and findings that motivate this

- Test pair: S-21-46 80 um (fixed, the 80 um level of the preprocessed 20 um store) and BigBrain 80 um resampled once
  onto S-21-46's lattice by a pairwise mutual-information affine
  (`/mnt/passport_user/BigBrain_S2146_80um/pair/T_pair_zyx_um.json`).
- Scored at 1280 um (Dice of tissue masks / NMI; linear only 0.919 / 1.023):
  - anatomix-register with its defaults: 0.978 / 1.062 (UNet backbone), 0.974 / 1.062 (ViT).
  - chunkreg with the same settings: 0.973 / 1.055 (lattice factor 2), 0.977 / 1.061 (lattice factor 1).
  - chunkreg, anatomix in 64-voxel windows, lr 1.0, smoothing annealed 3.0/2.0 -> 1.5/1.0, 43 passes at 1280 um:
    plateau 0.980-0.981 / 1.054 from about pass 10; no folds anywhere.
- Field stores hold their components in (z, y, x) = array-axis order, although their attrs say "x, y, z". An
  evaluation that trusted the label swapped z and x (all such numbers from before the fix were wrong).
- The residual (d99) stays 2-5 mm per pass for this pair even after Dice has plateaued, so the sub-voxel stop never
  fires; levels run to their caps.
- chunkreg's default coarsest-level stages (moments, rigid, affine, greedy) collapse an already-aligned cross-modal pair
  (Dice about 0.56: the affine stage shrinks the volume). anatomix-register's default is deformable-only.
- anatomix computed over a whole brain at a coarse level is dominated by whole-brain context; in 64-voxel windows the
  two brains' channels agree locally 2.4x more (local NCC at alignment 0.30 vs 0.12) and the loss localises 3.5x
  sharper.

## Already done and tested (main checkout, uncommitted)

| Addition | Where | Notes |
|---|---|---|
| `features.anatomix_window` | `features/__init__.py` (`anatomix:wN` spec, `anatomix_window()`, `describe`), `config.py` (`_with_features`, `_features_config_block`) | anatomix in N-voxel sliding windows; written into the feature spec so every worker builds the same extractor; round-trips through the saved config |
| `levels.anneal` | `config.py` (`LevelPolicy.anneal`, `stages_for_pass`, load/save), `passes/register.py` | per-level smoothing schedule over passes: `start * (end/start) ** min(k/(N-1), 1)`, then held; applies to the fold retries too |
| `levels.stop.require_subvoxel` | `config.py`, `pipelines/groupwise.py` `_should_stop` | with a held template, a level ends only when sub-voxel or at its cap, not when the residual "fits the next halo" |
| Stop reason for fold give-ups | `pipelines/groupwise.py` `_should_stop` | a pass where every tissue chunk gave up on folds is reported as such, not as "residual is sub-voxel" |
| Tests | `tests/test_anneal_window.py` (9) | full suite passes except the pre-existing `test_the_coarse_rerun_config_resolves_as_intended` (configs/atlas_v3.json says 100um, the test expects 0.8) |

## 1. Fixes

| # | Problem | Change | Files |
|---|---|---|---|
| 1a | Field components stored (z, y, x) but labelled "x, y, z" | Label "z, y, x displacement in millimetres (array axis order)" and add `component_order: ["z", "y", "x"]`; test that component 0 moves along z | `store.py` (Field attrs) |
| 1b | Residual-based stop never fires for weak cross-modal pairs | `levels.stop.plateau: {passes: N, rel: r}` -- stop once the level's residual percentile has fallen by less than `r` (relative) over the last `N` passes. d99 comes from fixed histogram bins (`stats.EDGES`), so compare bin indices, not raw floats | `pipelines/groupwise.py` (`_should_stop`), `config.py` |
| 1c | Default coarsest stages collapse already-aligned cross-modal pairs | Keep the default, but use a deformable-only coarsest level when any subject has a warm-start affine (section 4), and warn when the coarsest stages include `affine` for a template-held pairwise run | `config.py` |
| 1d (optional) | A field lattice coarser than its level costs accuracy at the coarse levels (0.977 -> 0.973 Dice at 1280 um) | Per-level `lattice_factor` (e.g. 1 at the coarse levels, 2 at the fine ones) | `config.py`, `store.Field`, `passes/promote.py`, `passes/blend.py` |

## 2. Level numbering matching OME-Zarr (in progress)

Branch `ome-level-order`, worktree `/home/andrew/scripts/scalable-reg-ome` (uncommitted).

- Level `l` = OME-Zarr dataset `l`: 0 = full resolution (the run grid), K = coarsest (the whole volume fits one chunk).
  The loop still runs coarse to fine (K, K-1, ..., 0); spacing of level `l` = native spacing x 2^l.
- `Volume.array(l)` reads dataset `l` with no translation.
- Config keys indexed by `l`: `levels.level_params`, `levels.level_stages`, `levels.anneal`, int `levels.stop_at`, int
  entries of `levels.run`. `levels.caps` is listed from full resolution towards coarser, padded with its last value
  for deeper pyramids.
- Renamed: `level0_stages` -> `coarsest_stages`, `level0_max_disp_frac` -> `coarsest_max_disp_frac`; the old names
  raise a ConfigError that explains the rename and the numbering.
- CLI `--stop-at N` and `--from-level N` take `l` (`--from-level N` redoes level N and every finer level).
- On disk and in logs: `levels/L<l>`, scratch `L<l>_it<n>`, manifests, pass records, QC titles.
- Each run records `level_order: "ome"`; resuming a run root made with the old numbering is refused with an
  explanation.
- Migrated: `configs/*.json`, `configs/sweep400/*`, `sge/*.qsub`, `slurm/*`, `README.md`, `docs/PLAN.md`, tests.

Review before merging: read the diff; compare the test baseline before and after; check every migrated config's level
meaning by hand; run a two-level smoke test on the 80 um pair and confirm folders/logs read L0 = 80 um.

## 3. OME-Zarr 0.6 input

- Extend the OME source reader in `ingest.py` to parse NGFF 0.6: `multiscales[].coordinateSystems`, per-dataset
  `coordinateTransformations` that may be `scale`, `translation`, `identity` or a `sequence` of them, with
  `input`/`output` names. Voxel size and origin come from the transform into the intrinsic system (the one every
  dataset outputs to, usually `physical`).
- Additional systems and multiscale-level transforms (for example the `MNI152NLin2009bAsym` affine BRIDGE writes) are
  carried as metadata, never applied to the registration grid.
- Tests: a BRIDGE-written 0.6 store; the S-21-46 0.6 root (`gs://.../S-21-46_HiP-CT_20um.ome.zarr/zarr.json`).
- Until this lands: pass the level-0 array path (`.../store.ome.zarr/0`) with `spacing_mm`.

## 4. Warm starts (no 20 um resample)

### Config (per subject)

```json
{"id": "moving",
 "source": ".../BigBrain2015_coronal_20um.ome.zarr/0",
 "spacing_mm": 0.02,
 "warm_start": {"affine": "bigbrain_affine_runmm_to_scanmm_zyx.json",
                "field": "init_field_1280um.zarr"}}
```

Meaning: the moving subject is sampled at `A (p + u(p))`, where `p` is a fixed-grid point in run-grid millimetres
(z, y, x), `u` is chunkreg's field (seeded with the warm-start field), and `A` is a 4x4 map from run-grid mm to the
subject's own scan mm (z, y, x; level-0 voxel index x spacing + origin). Both parts are optional; with neither the
behaviour is unchanged.

### Implementation

| Piece | Change |
|---|---|
| `config.py` | `SubjectSpec.warm_start` (affine: 4x4 list or JSON path with key `affine_zyx_mm`; field: store path); validation; load/save round-trip; part of the fingerprint |
| `ingest.py` | A subject with an affine is not resampled onto the run grid: link level 0 in place in its own frame and write dyadic coarser levels in that frame (block means, about 1/7 of the scan). Window percentiles as now. `grid.json` placement records `frame: native`, the affine and the scan grid. Subjects without one are unchanged |
| `store.py` `Volume` | `read_through(level, origin, shape, disp_mm)`: run-grid voxel centres of the block + displacement -> `A` -> scan coordinates -> index into the subject level whose spacing is closest without being coarser than the run level -> read the bounding box (+ margin) -> GPU trilinear sampling, zero outside. Optionally a sampler for arbitrary point sets (delivery) |
| `passes/_common.py` `warped_subject_block` | Use `read_through` when the subject has an affine, the existing `read_padded` + `warp_offset` otherwise. Registration, template building, QC and residual maps sample subjects through this function |
| `passes/promote.py` | Seed the coarsest level from the warm-start field: resample its lattice onto the level-0 lattice in run-grid mm (fields are mm, so only sampling positions change); refuse when its extent or frame does not match within a tolerance |
| Delivery (`twoimage.PairResult.warp_moving` / `export_moving`, `apply.py`, CLI `apply`) | Same sampler, so the final moving volume on the fixed lattice comes from the native scan in one interpolation |
| Defaults | A warm-start affine implies a deformable-only coarsest level (fix 1c) |

Conventions to pin in docstrings and tests: fields pull fixed -> moving, components (z, y, x) in mm; `A` is applied
after the field; the affine is in run-grid mm, i.e. the run grid's own centred frame (`grid.json` `origin_mm`).

### Tests

- Synthetic: a volume and an affinely transformed copy; registering the copy with the warm-start affine equals
  registering the pre-resampled original to within interpolation error.
- Field seeding: a known field as warm start; pass 0 starts from it (the read moving block equals the block warped by
  that field).
- Config round-trip; refusal of a mismatched field frame.

### Real-data check (local, before the cluster)

At 80 um: BigBrain's native 80 um store (`BigBrain2015_coronal_80um.ome.zarr`), the local-frame version of the
affine and `init_field_1280um.zarr` (shifted back into the local frame) must reproduce the coarsest-level result of
the pre-resampled run (Dice about 0.98 at 1280 um) without the pre-resampled input.

Cost: reading through the affine needs a bounding box about 1.2-1.5x the chunk (BigBrain's scale factors relative to
S-21-46 are 0.95-1.21), far cheaper than writing a 7375 x 7465 x 7465 resampled copy.

### Prepared inputs

`/mnt/passport_user/BigBrain_S2146_80um/cluster_warmstart/`:

- `bigbrain_affine_runmm_to_scanmm_zyx.json` -- cluster run-grid mm (run grid = S-21-46 20 um store, centred:
  centre (73.74, 74.64, 74.64) mm) -> BigBrain 20 um scan mm. The run-grid centre maps to (75.0, 57.2, 67.4) mm in
  BigBrain (scan centre (74.0, 57.1, 65.7) mm).
- `init_field_1280um.zarr` -- the converged 1280 um field (43 passes; Dice 0.980 / NMI 1.053 at 1280 um), translation
  shifted by (+0.010, +0.030, +0.030) mm from the 80 um run's frame into the 20 um run's frame, label corrected to
  z, y, x.

## 5. Chunk size

| core (voxels) | edge at 20 um | chunks | with tissue >= 2% | padded chunk / core volume (halo 68) |
|---|---|---|---|---|
| 128 (current) | 2.56 mm | 201,898 | 58,736 | 264^3 = 8.8x |
| 256 | 5.12 mm | 26,100 | 8,115 | 392^3 = 3.6x |
| 384 | 7.68 mm | 8,000 | 2,586 | 520^3 = 2.5x |
| 512 | 10.24 mm | 3,375 | 1,184 | 648^3 = 2.0x |

The core is one setting for every level (one chunk geometry), so core 256 makes the coarsest level 640 um (234^3 fits
one chunk); the 640 um single-chunk tests reached the same Dice as 1280 um. Keeping a 1280 um coarsest level with big
fine-level chunks would need a per-level core (a larger change; deferred). Before the cluster run: a one-chunk memory
check of a 392^3 chunk with 16 windowed anatomix channels on an 80 GB card.

## 6. Cluster run once 2-4 are in

1. Pull both stores: `gs://hip_ct_zarr_vector_03987646472fethdsvdvdfg/zarr_vectors_test/BigBrain_registration_2026/`
   `S-21-46_HiP-CT_20um.ome.zarr` (its root already carries the linear-to-MNI affine, NGFF 0.6) and
   `BigBrain2015_coronal_20um.ome.zarr`.
2. Config: fixed = S-21-46 20 um; moving = native BigBrain 20 um with `warm_start` (affine + 1280 um field);
   profile a16 with `features: {mind: false, sample_channels: null, anatomix_window: 64}`; lr 1.0; coarse levels
   annealed 3.0/2.0 -> 1.5/1.0 with many passes and the plateau stop; fine levels 1-2 passes at smoothing 1.5/1.0;
   core 256; ingest percentiles (0, 99.99).
3. `chunkreg run config.json` (one worker per GPU).
4. QC per level with the corrected evaluation scripts in `/mnt/passport_user/BigBrain_S2146_80um/`
   (`eval_all_1280.py`, `eval_metrics.py`, `qc_level0.py`, `eval_field.py` -- Jacobian, folds, chunk-seam ratio) at
   640 and 320 um before committing to 40 and 20 um.

## 7. Order and estimates

| Step | Time |
|---|---|
| Review and merge the level renumbering (2) | 30 min |
| Fixes 1a-1c | 45 min |
| OME-Zarr 0.6 reader (3) | 45 min |
| Warm start (4) with tests | 2.5-3.5 h |
| Real-data warm-start check at 80 um | 30 min |
| Cluster package and one-chunk memory check (core 256) | 45 min |
| Optional: per-level lattice factor (1d) | 1 h |

If time is short: defer 1d; use the `/0` array path instead of the 0.6 reader; the field seed alone (about 45 min)
avoids re-running 1280 um, but only the affine part avoids the 20 um resample.

## Appendix A: implementation map (read from the code, 2026-10-19; line numbers are the main checkout with the
## uncommitted additions, before the level renumbering)

### Conventions to keep
- `GridSpec` (`grid.py:45-108`) is isotropic and axis-aligned: `world = origin_mm + spacing_mm * voxel`, origin = centre
  of voxel 0, array order (Z, Y, X). `coarsened(f)`: shape `ceil(n/f)`, spacing `s*f`, origin `+ s*(f-1)/2`.
  `pyramid(native, profile)` (`grid.py:284-290`), depth `pyramid_depth(shape, core)` (`grid.py:273-281`).
- Displacements (`fields.py:1-26`): `(3, Z, Y, X)` mm in the run-grid level's world frame, component d = array axis d
  (z, y, x), pull fixed -> moving `warped(x) = M(x + u(x))`; images sampled with 0 outside, fields edge-padded.
  `store.py:610` mislabels them "x, y, z" (fix 1a). External ITK/ANTs fields are xyz-ordered with LPS signs: convert.

### 3. OME-Zarr 0.6 -- where
- `io_formats.py` `_resolve_zarr` (198-235) is used both by ingest (`open_source` 307-348 -> `_open_zarr` 238-241) and at
  run time for linked levels (`open_zarr_in_place` 275-284): one fix covers both.
- Today NGFF 0.6 axes (under `coordinateSystems[*].axes`) are missed, the reader falls into its NGFF <= 0.3 branch and
  spacing comes back `None`. `_ome_spacing` (153-181) multiplies `scale`s only and ignores `translation`, `sequence`,
  `input`/`output`. No format parses an origin (`VolumeSource` 63-104 has no origin field; `_open_nifti` reads zooms
  only) -- warm starts need the scan origin, so add `origin_mm` to `VolumeSource` while doing 0.6.
- Test fixture to extend: `tests/test_io_zarr.py` `write_ome` (44-71).

### 4. Warm starts -- where, and what assumes a shared grid
Hook points:
- Moving read during registration: `passes/_common.py` `warped_subject_block` (13-34) assumes subject level k and run
  level k share voxel indices; give it the run-level `GridSpec` (every caller has one: `manifest.grid`,
  `fld.level_grid`, featurereport's `grid`). Model for an affine read box: `scripts/warp_final.py:80-113` (box from the
  field's min/max). GPU coordinate sampler exists: `gpu_ops.sample` (72-107); CPU side uses `ndimage.map_coordinates`.
- Other subject reads through a field: `passes/blend.py:228-232` (template intensity mean), `apply.py:50-138`
  (`_Warped`, `apply_field`; used by `twoimage.PairResult.warp_moving`/`export_moving` and CLI `cmd_apply`),
  `featurereport.py:133-173`, `scripts/warp_final.py`, `scripts/compare_sweep.py`.
- Subject reads without a field (need the affine too): `promote.seed_level_zero` mean (353-361),
  `promote.template_from_subject` (288-307), `probe.py:71-90`, `scripts/anatomix_level_pngs.py:285`.
- Init field: `promote.seed_level_zero` (332-372) creates the level-0 fields (zeros by zarr fill) -- write the warm-start
  field there (resample onto `Field.lattice_grid`, `write_lattice`, or `write_dense` at level resolution).
  `register._seed_block` (41-63) then picks it up at level 0 pass 0 automatically. Promotion/settling
  (`promote._carry_fields` 158-188, `_carry_core` 41-69, `settle_fields` 191-205) is pure resampling of mm fields, so
  keeping the affine outside the field (`scan = A(x + u(x))`) means promote/blend/update/recentre need no change.
- Level 0 still runs `level0_stages` (moments/rigid/affine/greedy, `config.py:208-213`) on the pre-warped moving; with
  a warm start use deformable-only (fix 1c). `Engine.register` already takes `init_affine` (`engines/base.py:74-82`,
  `fireants.py:240-266`) but `register_chunk` never passes it -- not needed with the affine outside the field.
- Config: `SubjectSpec` (`config.py:439-452`), parsing in `load_config` (1426-1445; allowed keys
  `{"id","path","source","spacing_mm"}`), round-trip in `to_config_dict` (941-956), cross-subject checks in
  `RunConfig.validate` (683-901); relative paths made absolute in `pipelines/groupwise.pair_config` (557-570);
  `twoimage.register` builds subject dicts at 203-216 (add `moving_affine` / `moving_init_field`).

Shared-grid assumptions a natively placed subject breaks (each needs handling):
1. `ingest.native_grid(cfg)` (`ingest.py:51-65`) takes the first subject's grid as the run grid (feeds `cli.py` setup
   398, run 446, pair 543) -- read `grid.json` instead (written at `ingest.py:234-244`, never read back today).
2. `ingest_subjects` (133-346): every subject feeds `resolve_run_grid` (`cohort.py:119-225`) and would change the grid's
   extent/spacing; the kept-store check (211-224) raises for stores off the run grid; placement and `link_source`
   (`store.py:1045-1081`, passed the run spacing/origin) assume resampling onto the run grid. A warm-start subject
   should be excluded from `resolve_run_grid` and linked with its own spacing/origin.
3. `check_subjects` (`ingest.py:349-381`) requires `same_grid`; `check_volume` (`store.py:1118-1175`) requires the run's
   pyramid depth.
4. `groupwise.build_template` (206-216) raises if any subject's `native_grid` differs, and builds the pyramid from the
   first subject.
5. Subject level k == run level k is assumed in register (122), blend (229), seed mean (358). For a native subject,
   choose its level by spacing (affine scale x run spacing), not by index; `warp_offset` has no prefilter, so avoid
   reading a finer level than needed.
6. `GridSpec` is isotropic: an anisotropic scan cannot be a native `Volume` unless its voxel scaling moves into the
   affine (treat the store as voxel-index space).
7. `apply.matching_level` (33-47) matches levels by shape -> fails for a native moving subject (`warp_moving`,
   `export_moving`, `cli apply`).
8. `featurereport.build_feature_report` (193-208) takes native/grids from the first subject, requires equal `n_levels`.
9. `probe.py:71-90` reads every subject at level 0 on the first subject's grid.
10. Normalisation: `Volume.compute_normalisation` (`store.py:450-478`) takes percentiles over the whole coarsest level;
    a resampled subject includes zero padding outside the scan, a native one does not, so its window differs -- set
    percentiles explicitly or mask.
11. Accumulators and QC stores are sized from the run grid -- fine while all warped output stays on the run grid.
12. Tests encoding these assumptions: `tests/test_io_zarr.py:382-419`, `tests/test_scan_space.py`,
    `tests/test_twoimage.py`, `tests/test_cohort.py`, `tests/test_store.py`.

### Revised estimate for section 4
The shared-grid assumptions (items 1-10) are the bulk of the work: about 4-5 h for the warm-start affine with tests,
not 2.5-3.5 h. The init-field seed alone (promote.seed_level_zero + config) stays about 45 min and is independent.

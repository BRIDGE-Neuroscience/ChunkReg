# Large scans

At 20 µm a HiP-CT or histology scan is 7,000 voxels or more per side. One copy of
it is hundreds of gigabytes, and a naive run makes several copies per pass. The
features below remove those copies. Most are automatic; the table says which.

| Feature | What it does | Use it when | Switch |
|---|---|---|---|
| Linked scan | A zarr already exactly on the run grid is read in place. Only the coarser levels are written (about 1/7 of the scan). | the scan is on storage you can read fast | `ingest.link` (default on) |
| Linked pyramid | An OME-Zarr that carries its own mean pyramid is used at every level whose shape matches. Each level is checked against a 2x2x2 mean of the level below, and built here if it is not one. | the source has a pyramid, such as an existing 20 µm HiP-CT store | automatic with `ingest.link` |
| `warm_start` | The moving scan stays in its own frame and is sampled through a known affine on demand. Nothing is resampled on disk. | you have a pairwise affine and the scan is huge | `subjects[].warm_start` |
| Linked template | A held template of 2^30 voxels or more is read from the subject's level through its intensity window instead of copied as float32. | pairwise runs at fine levels | automatic; `CHUNKREG_TEMPLATE_LINK_MIN_VOXELS` moves the threshold |
| Settled field link | With a held template there is no template motion to fold in, so `fields/<id>.zarr` is a symbolic link to the level's field, not a second copy. | pairwise runs | automatic where the filesystem has links |
| Early task-array deletion | On a multi-GPU run each task array is deleted as soon as the last blend that reads it is done, not at the end of the pass. | a level's task arrays reach a terabyte | automatic if `retention.delete_task_arrays` |
| `scratch` | Task arrays and accumulators go to a directory of your choice, such as node-local disk. | the node has local scratch and the shared filesystem is slow or small; local runner only, since every worker must see the directory | `scratch` |
| Residual map cap | Residual QC maps are written only for levels whose lattice is small enough. | the finest level's map would be hundreds of gigabytes | `retention.residual_map_max_voxels` |
| Grid-anchored anatomix windows | Windowed anatomix is tiled from the run grid, not from each chunk's own origin, so a voxel has the same features in every chunk that holds it. Costs about twice the feature compute; the engine, which dominates, is unchanged. | cross-modal or whole-brain scans | `features.anatomix_window` |

Linking needs a zarr scan exactly on the run grid, with no `median_radius` and no
dtype change; any other scan is copied onto the grid one block at a time.

## `warm_start`: the moving scan in its own frame

Resampling a 7,500³ scan onto the fixed volume's lattice writes hundreds of
gigabytes for a copy that is read once per pass. With a known pairwise affine,
skip it:

```json
{"id": "moving", "source": "/data/BigBrain2015_coronal_20um.ome.zarr",
 "warm_start": {"affine_um": "pair/T_pair_zyx_um.json",
                "background": [250, 0],
                "field": "init_field_1280um.zarr"}}
```

- `affine_um` maps the moving scan's physical µm to the fixed scan's physical µm,
  the direction a pairwise affine is normally estimated in. It is a 4x4 or 3x4
  matrix, or a JSON file holding one (the first such matrix in the file is used).
- The subject's store holds only metadata. Every level is sampled when a block is
  read, trilinearly, zero outside the scan, from the scan's OME dataset whose
  voxel is the largest not coarser than the level's. The source must be an
  OME-Zarr (v0.4, 0.5 or 0.6).
- `background` sets voxels at or above a threshold to a value before
  interpolating, so a white histology background and "outside the scan" agree.
- The fixed subject's own sampling is the run grid. Name it with `grid.reference`
  (or `template_subject`). The warm-start subject takes no part in choosing the grid.
- `field` seeds the coarsest level from a ChunkReg field store made earlier in the
  same world frame. Vectors are millimetres, so only the lattice is resampled; a
  field from another frame is refused.
- The scans are already aligned, so give the coarsest level a deformable-only
  stage list. The default `coarsest_stages` include an affine stage, which can
  shrink an already-aligned cross-modal pair.

Nothing downstream knows. Register, blend, promote and QC see an ordinary pyramid
on the run grid. `chunkreg apply` and `export` work on the result as usual.

### An example: histology onto HiP-CT at 20 µm

Abridged from `scripts/bigbrain_s2146.py prepare`. The run has 6 levels (640 to
20 µm, `K` = 5). Passes are few at the fine levels and many at the coarse ones.

```json
{
  "root": "/data/bb_s2146",
  "subjects": [
    {"id": "fixed",  "source": "/data/S-21-46_HiP-CT_20um.ome.zarr"},
    {"id": "moving", "source": "/data/BigBrain2015_coronal_20um.ome.zarr",
     "warm_start": {"affine_um": "pair/T_pair_zyx_um.json", "background": [250, 0]}}
  ],
  "template_subject": "fixed",
  "grid": {"reference": "fixed"},
  "profile_overrides": {"halo": 68, "r_f": 32, "sigma_g": 3, "sigma_w": 2},
  "features": {"mind": false, "sample_channels": null, "anatomix_window": 64},
  "ingest": {"percentiles": [0, 99.99]},
  "scratch": "/scratch0/me/bb_s2146",
  "retention": {"residual_map_max_voxels": 1073741824},
  "levels": {
    "caps": [1, 1, 2, 4, 18, 32],
    "coarsest_stages": [{"greedy": {"scales": [8, 4, 2, 1], "iterations": [240, 160, 100, 60], "lr": 1.0}}],
    "stop": {"require_subvoxel": true, "plateau": {"passes": 3, "rel": 0.05}},
    "anneal": {"5": {"passes": 25, "smooth_grad_sigma": [3.0, 1.5], "smooth_warp_sigma": [2.0, 1.0]},
               "4": {"passes": 12, "smooth_grad_sigma": [2.5, 1.5], "smooth_warp_sigma": [2.0, 1.0]}}
  }
}
```

## Budgeting disk

Sizes for a 7,500³ scan at 20 µm, uncompressed. Tissue-free chunks compress to
almost nothing, so real stores are smaller.

| Object | Size at level 0 | Lifetime | Lever |
|---|---|---|---|
| Subject store, linked | metadata, plus coarser levels (about 1/7 of the scan) | the run | `ingest.link`, `warm_start` |
| Subject store, copied | the scan again | the run | |
| Field, per registered subject | 3 x 3750³ x 2 B = 316 GB; each coarser level is an eighth of the one below | kept | |
| Task arrays, per pass | about 1 TB at the peak | deleted after the blend | `scratch`, early deletion |
| Residual map, per subject | 3750³ x 4 B = 211 GB | overwritten every pass | `residual_map_max_voxels` |
| Template, if copied | 7500³ x 4 B = 1.7 TB | the level | linked template |

`residual_map_max_voxels: 1073741824` (2^30) keeps maps down to 80 µm here and drops
the 40 and 20 µm ones. The per-pass QC sheet then says no map was written.

Put the run root on a filesystem with room for the fields, and `scratch` on the
fastest disk you have. A resumed run on a different node just redoes the pass that
was unfinished.

## Budgeting time

`chunkreg setup` prints GPU-hours at 8 and 64 GPUs (`--gpus` changes the counts)
from a measured throughput. It counts every chunk at its cap, so it is an upper
bound. Two helper commands in `scripts/bigbrain_s2146.py` show the finer budget.
They are written for the BigBrain and S-21-46 study (its profile overrides, and
a subject called `fixed`), so copy and edit them for another study.

```bash
# Does one padded chunk fit this card, and how long does it take? Needs no data.
python scripts/bigbrain_s2146.py memcheck --core 256

# Chunks per level, chunks with tissue, and hours, from a measured seconds per chunk.
python scripts/bigbrain_s2146.py estimate /data/bb_s2146/c256/config.json \
    --seconds-per-chunk 30 --gpus 8
```

`memcheck` runs one padded chunk through the real extractor and engine, prints peak
memory against the card (`FITS` under 85%, otherwise `TOO BIG` and exit status 2)
and the seconds per chunk. `estimate` prints, per level, the chunks, the chunks
with tissue, the cap, and hours per pass and per level:

```
pass time  = (tissue chunks x s/chunk + empty chunks x 1.5 s) / GPUs
level time = pass time x cap          (+ 10 to 15 % for promotes, settling and QC)
```

Seconds per chunk is scaled at coarser levels by that level's deformable iteration
count relative to the finest level's. For the S-21-46 scan at core 256 the planning
figures were 26,100 chunks at 20 µm, about 8,100 with tissue, and an assumed 30 s per
chunk on an H100. One 20 µm pass on 8 GPUs is then 8,100 x 30 / 8 = 8.4 h, plus about
0.9 h for the 18,000 empty chunks.

If the total does not fit:

- Lower the iteration counts at the finest level; they dominate.
- Stop early with `--stop-at 40um` and continue later. The run resumes.
- Use a smaller `core` (192 with a halo of 68 gives a 328³ padded chunk and one more level).
- Lower `caps` at the fine levels. A seeded level starts close, so one pass is often enough.

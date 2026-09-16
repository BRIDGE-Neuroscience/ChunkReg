# chunkreg

Hierarchical co-registration of N volumes that are individually too large for a
GPU. Produces an unbiased group template and one diffeomorphic displacement
field per volume.

The design note is [docs/PLAN.md](docs/PLAN.md). This file is how to run it.

## The idea in one paragraph

Work on a resolution pyramid with **one fixed chunk geometry at every level**.
The coarsest level is by construction the level at which a whole volume fits
inside a single chunk, so there is no separate whole-volume code path: one
memory footprint, one scheduler resource request, top to bottom. Each finer
level is seeded by the level above and solves only a residual, which is what
keeps chunked registration honest. Chunk solutions are merged with a partition
of unity. Similarity is computed on anatomix feature channels registered by
FireANTs. Every volumetric object is a sharded zarr array, and every pass is a
pure function of `(config, task id)`, so the same code runs on a laptop and on a
thousand-node cluster.

## Install

```bash
pip install -e ".[store,io,qc]"     # core plus zarr, format IO, QC tables
pip install -e ".[gpu]"             # torch, MONAI, SimpleITK
pip install git+https://github.com/neel-dey/anatomix.git
bash anatomix/registration/registration_backend/install_fireants.sh
```

Only the core is needed to run the CPU reference engine and the whole test
suite. The GPU extras are needed for the production path.

## Run

```bash
chunkreg ingest raw/s01.tif subjects/s01.zarr --spacing 0.05    # once per subject
chunkreg selftest                                               # conventions
chunkreg calibrate run.yaml                                     # r_f, memory, throughput
chunkreg probe run.yaml                                         # how far apart the cohort is
chunkreg plan run.yaml                                          # pyramid, clamps, cost
chunkreg features run.yaml                                      # are the features worth it?
chunkreg template run.yaml                                      # the run
chunkreg status run.yaml                                        # progress, outstanding ids
```

Run `plan` before committing to anything. It resolves the pyramid depth from
the grid, the displacement clamp from the halo, and prints the work in padded
channel-voxels with the levels ranked by cost.

## The four things worth knowing

**The halo defines the clamp, not the other way round.** With the chunk
geometry fixed, the halo bound

```
h >= D_max/spacing + s_max*[(k-1)/2 + 3*max(sigma)] + r_f
```

is solved for `D_max`. At the default profile that is 12 voxels at every level,
so 4.8 mm at 400 um and 0.6 mm at 50 um. A configuration cannot ask for more
than its halo can justify; `chunkreg plan` refuses one that tries.

**The receptive field is measured, not assumed.** `r_f` is subtracted from the
halo before anything is left for displacement, so a wrong value silently
overdraws the budget. `chunkreg calibrate` measures it by perturbing a voxel and
watching how far the change propagates, and warns when the profile understates
it.

**Levels stop on a measured condition.** A level keeps iterating until its
residual fits comfortably inside the next level's clamp and the template has
stopped moving by more than a voxel. Caps are a backstop, not the schedule.

**Look at the features before paying for them.** `chunkreg features` renders the
same physical box at every pyramid level, for several subjects, as intensity
above and feature channels as RGB below, on one sheet. One projection is shared
by every panel, so a colour means the same thing everywhere; `--basis per-level`
trades that for per-level contrast when you want to know whether a level that
renders flat has structure in its own subspace. Three numbers sit under each
tile: contrast (is there structure at this scale), effective rank out of the
channel count (has the descriptor collapsed), and the fraction of feature
variance the picture actually shows. That is the evidence gate G2 needs.

**Channel count is the cost lever.** Registration work is linear in channels.
The `a16`, `a8`, `a4` and `i1` profiles are the same geometry at 16, 8, 4 and 1
channels, spanning a 16x range in both cost and device class. `i1` also gets
three times the clamp, because raw intensity has no receptive field to pay for.

## Layout

```
chunkreg/
  grid.py         GridSpec, Profile, Chunk, pyramid, tile, window
  fields.py       displacement conventions, compose, warp, jacobian, engine bridge
  store.py        Volume, Field, TaskArray, ingest
  backend.py      zarr (production) and memory (tests)
  stats.py        mergeable histograms for exact distributed percentiles
  features/       intensity, anatomix, MIND-SSC, PCA projection
  featureviz.py   feature-to-RGB projection, metrics, contact sheets
  featurereport.py  samples the same world box at every level, per subject
  engines/        demons (CPU reference), fireants (production)
  passes/         manifest, register, blend, update, promote
  pipelines/      the level loop and its stopping rule
  runners/        local pool, SLURM arrays
  planner.py probe.py calibrate.py status.py selftest.py cli.py
```

## Conventions

Arrays are `(Z, Y, X)`. Displacements are `(3, Z, Y, X)` float16 in
**millimetres**, components in **array-axis order** `(z, y, x)`, mapping
**template coordinates to subject coordinates**. Composition is
`compose(inner, outer)` giving `inner(x) + outer(x + inner(x))`.

Storing in millimetres is what makes promoting a seed between levels exact
rather than approximate: resampling changes the lattice, never the vectors.
Storing components in axis order rather than `grid_sample` order confines the
reversal to one function, `fields.grid_to_disp_mm`, instead of spreading it
through every block read.

`chunkreg selftest` drives known shifts through all of this, whole-chunk and
tiled, and is the first thing to run against a new engine build.

## Testing

```bash
python -m pytest tests/ -q          # about 2.5 minutes
```

The suite runs entirely on the CPU reference engine and the in-process memory
backend, so it needs neither a GPU nor zarr. All block arithmetic, lattice
conversion and sharding logic sits above the backend seam, so exercising it on
the memory backend exercises the zarr path too.

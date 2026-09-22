# chunkreg: hierarchical co-registration of N massive volumes, anatomix-backed

Implementation plan for an agent. `chunkreg` co-registers any group of N
volumes that are individually too large for a GPU, producing an unbiased
group template and one diffeomorphic field per volume. It works on a
resolution pyramid with **one fixed chunk geometry at every level**: the
coarsest level is simply the level at which the whole volume fits inside one
chunk, so there is one code path, one memory footprint and one SLURM resource
spec from top to bottom. **Every volumetric object is a zarr array**, scalar
`(Z,Y,X)` or vector `(3,Z,Y,X)`; nothing else touches disk after ingest.
Similarity is computed on **anatomix** feature channels registered by
**FireANTs**, with channel count as the primary cost control and raw intensity
as the one-channel degenerate case.

Starting points in this repo:

- `scripts/hipct_patchreg.py`: reusable tiling, Hann partition-of-unity
  windows, blockwise warp and resample helpers, composition, template update
  math. Its storage layout and single-node pool are replaced.
- `scripts/median3d.py`: streaming 3D median for TIFF stacks; usable in ingest.

External code, imported not vendored:

- anatomix `anatomix/registration/registration_infrastructure/`:
  `features.py` (load_backbone, minmax_normalize, prepare_feature_channels,
  normalize_features), `register.py` (run_registration, stage specs,
  `reextract_moving`, `_compose_grids`, `_compose_linear_grid`), `warp_io.py`
  (grid conventions, get_warp_parameters, warp_volume, invert_grid),
  `_fireants.py` (FakeBatchedImages). Backend install:
  `registration_backend/install_fireants.sh` pins the neel-dey/FireANTs fork at
  commit c72d2ef and builds fused_ops.
- FireANTs: `fireants/registration/{greedy,syn,affine,rigid,moments}.py`,
  `fireants/io/image.py` (Image, BatchedImages, FakeBatchedImages,
  torch2phy/phy2torch), `fireants/scripts/template/` (reference template loop,
  Laplacian sharpening).

---

## 1. Scope and shape of the solution

Inputs: N scalar volumes of the same modality class (any of TIFF stacks,
NIfTI, zarr) with known voxel spacing, on the same or different grids, roughly
the same field of view. Outputs: `template.zarr` at every level, one
`fields/<sid>.zarr` per volume mapping template coordinates to that volume,
and a QC record. Pairwise registration is the same pipeline with N = 1 and a
fixed template: `template_subject` names the volume to hold, its own data is
the template at every level, the unbiasing update pass does not run, and the
level loop converges on the residual instead of on template motion (see
[The stopping rule](#the-stopping-rule)). `grid.reference` then takes the run
grid from that volume, so the result lands on the lattice it is expected on
rather than on a grid chosen for a cohort.

The pipeline in one paragraph. Ingest resamples each volume onto a common
grid as a sharded, multiscale zarr store. The pyramid has `K + 1` levels where
`K = ceil(log2(max(shape) / core))`; at level 0 the whole volume is one chunk.
Level 0 aligns every volume to an initial template by moments, rigid and
affine, then deformably, and iterates the template to convergence. Each finer
level is seeded by the upsampled template and fields, tiles the grid into
fixed 256³ cores with 48-voxel halos, registers every (volume, chunk) pair as
an independent GPU task on anatomix features, blends chunk residuals with a
partition of unity into per-volume field stores, averages the warped volumes
into a new template, recentres it, and repeats until the leftover residual is
small enough for the next level's halo to absorb. Every pass is a pure
function over zarr shards, so it runs identically under a local GPU pool or a
SLURM array.

```
                       ┌──────────────── level k (k = 0 … K) ────────────────┐
  subjects/*.zarr ──►  │  register ──► task.zarr ──► blend+average ──► update │ ──► promote ──► level k+1
  (multiscale s0..sK)  │  (N × chunks GPU tasks)     (per shard)     (per shard)│      (upsample)
                       │        ▲                                         │     │
                       │        └──────── not converged: next iteration ◄─┘     │
                       └──────────────────────────────────────────────────────┘
  stores per level: template.zarr · fields/<sid>.zarr · scratch/isum.zarr · scratch/wsum.zarr
```

---

## 2. Theory

### 2.1 Transform model and conventions

All volumes live on a common isotropic, axis-aligned grid
`G = (shape, spacing, origin)`, arrays ordered `(Z, Y, X)` to match zarr, TIFF,
FireANTs `.array` and MONAI. A transform is a displacement field `u` in
millimetres, stored as `(3, Z, Y, X)` float16 with component order `(x, y, z)`
(the order of a `grid_sample` grid's last axis). `φ(x) = x + u(x)` maps
**template (fixed) coordinates to subject (moving) coordinates**;
`(M ∘ φ)(x) = M(x + u(x))`. Composition on one grid:
`(φ_a ∘ φ_b)(x): u(x) = u_b(x) + u_a(x + u_b(x))`, i.e.
`compose(outer=u_a, inner=u_b)`. Because `u` is physical, resampling it onto
a finer grid or a coarser storage lattice changes only the lattice. The two
functions `grid_to_disp_mm` / `disp_mm_to_grid` are the only bridge to an
engine's normalised coordinates and carry a selftest with known shifts.

### 2.2 Residual (seeded) registration

With `φ0` the seed (previous level upsampled, or previous iteration), pre-warp
`M0 = M ∘ φ0`, register `F` to `M0` for a residual `ψ`, and set
`φ = φ0 ∘ ψ`. The residual is small by construction, so the halo need only
cover it, the subject is interpolated once per level, and non-equivariant
feature extractors (anatomix) are evaluated on `M0`, on the template grid,
which is anatomix's own `reextract_moving` rule.

### 2.3 Locality and the halo bound

For a local similarity of kernel width `k`, Gaussian regularisation sigmas
`σ_g, σ_w` (in the voxels of the pyramid scale they run at), a feature
extractor with receptive-field radius `r_f`, and a residual clamped to
`D_max`, the solution inside a chunk core depends only on data within

```
h  ≥  D_max/spacing  +  s_max · [ (k−1)/2 + 3·max(σ_g, σ_w) ]  +  r_f      (native voxels)
```

where `s_max` is the coarsest in-chunk pyramid scale. The `s_max` factor is
essential: an LNCC window of 7 at scale 8 spans 56 native voxels.

**With a fixed halo, this bound is read the other way round.** `h`, `k`, `σ`,
`s_max` and `r_f` are fixed by the chunk profile, so the bound *defines the
clamp*:

```
D_max(level)  =  ( h − s_max·[(k−1)/2 + 3σ] − r_f ) · spacing(level)
```

With the default profile (`h = 48, k = 7, σ = 1, s_max = 2, r_f = 24`) this is
12 native voxels at every level: 4.8 mm at 400 µm, 0.6 mm at 50 µm. The
coarser level's job is to leave a residual that fits under the finer level's
clamp (2.6). The chunk cannot be asked to do more than its halo allows, and the
configuration cannot express a request that violates the bound.

### 2.4 The aperture problem and why the pyramid exists

A halo fixes data dependence; it cannot fix the aperture. A deformation whose
correlation length exceeds the chunk width `L = core · spacing` looks like a
translation inside a chunk and is invisible to a local similarity until it
exceeds the anatomy's correlation length. Only a level whose `L` covers the
deformation resolves it. With a fixed core, `L` doubles per level up the
pyramid, and level 0 has `L ≥` the whole volume. That is why the pyramid
always starts at level 0: it costs one registration per subject of a ≤ 256³
volume, and it is the only place the global component can be estimated.
In-chunk multiscale (`scales: [2,1]` or `[4,2,1]`) resolves ambiguity only up
to `L`. Template iteration propagates information between chunks through the
template, but slowly and at full cost per pass.

### 2.5 Partition-of-unity blending

Cores tile the grid without overlap; each is padded by the halo. Each chunk
carries a Hann window `w_i`, 1 on the core interior and tapering over the halo,
with `Σ w_i > 0` everywhere. `u = Σ w_i u_i / Σ w_i`. Where one chunk
contributes, `u = u_i` exactly; in overlaps, `u` is a convex combination and
deviates from each contributor by at most their disagreement `ε`, which the
halo bound keeps small. Convex combinations of diffeomorphic fields are not
guaranteed diffeomorphic, so the blend is followed by a one-lattice-voxel
Gaussian smoothing and a fold check per shard; folds in an overlap indicate a
violated bound and trigger the chunk retry ladder (section 5). The denominator
is analytic and never stored. Because shards are chosen equal to cores, the
blend task for shard `s` reads exactly the results of chunk `s` and the ≤ 26
neighbours whose halos reach into it.

### 2.6 Groupwise unbiased template and the per-level stopping rule

```
min_{T, φ_i}  Σ_i [ D(T, I_i ∘ φ_i) + λ R(φ_i) ]   s.t.   mean_i(u_i) = 0
```

Alternate: register each `φ_i` with `T` fixed (embarrassingly parallel);
average `T̂ = (1/N) Σ I_i ∘ φ_i`; recentre `T ← T̂ ∘ (id − ε ū)`,
`ū = (1/N) Σ u_i`, `ε = 0.25`, and subtract `ε ū` from every seed. Only two
accumulators exist, `isum` (scalar) and `wsum` (vector), written shard by
shard by the shard's owner. Bias decays as `(1 − ε)^k` per pass.

**Stopping rule per level** (replaces hand-set iteration counts):

```
stop at level k when   d99_residual(k) < 0.5 · D_max(k+1)   and   ε·‖ū‖₉₉ < spacing(k)
```

i.e. the leftover residual fits comfortably under the next level's clamp and
the template has stopped moving by more than a voxel. Each level also has a
cap (default `[8, 6, 4, 3, 2]` from coarse to fine) and a floor of 1. A level
that inherits a converged shape exits after one pass; the first level runs
until the global shape is right. The last level stops on `‖ū‖` alone.

**With a fixed template the rule is different, because `ū` does not exist.**
No update pass runs, so the recentring step is identically zero and the
`ε‖ū‖ < spacing` half passes vacuously — on the last level, where it is the
whole rule, that exits every level after one pass however far from converged
it is. What converges instead is the residual itself, the part of the
correspondence each pass could not already explain:

```
stop at level k when   d99_residual(k) < residual_vox · spacing(k)
                or     d99_residual(k) < 0.5 · D_max(k+1)
```

`residual_vox` defaults to 0.5. Below half a voxel at the 99th percentile the
pass is correcting less than the level can represent and the next level will
resolve it better than another pass here.

### 2.7 Inner iteration schedule

For an intensity LNCC of width `k`, the coarsest in-chunk scale must satisfy
`s_max ≳ 2·D_max/(k·spacing)`; with `D_max = 12` voxels and `k = 7` that is
`s_max ≳ 3.4`. But `s_max` also eats halo (2.3): at `s_max = 4` the smoothing
term is 24 voxels and nothing is left for `D_max`. The default profile resolves
this by running seeded levels at `scales: [2,1]` and relying on two facts: the
seed already places the residual within `D_max`, and anatomix features have a
capture range several times wider than an intensity window (measured by
`calibrate`, which is what lets the planner accept `[2,1]`). Level 0 (single
chunk, unclamped after affine) uses `[8,4,2,1]`, where halo is irrelevant. Iterations per scale `≈ d(s)/δ + N_conv`: each finer octave has 2
to 4× less displacement to cover, so counts decrease geometrically toward a
convergence tail of 20 to 30. Configure them as caps with FireANTs
`tolerance` early stopping, and after the first pass at any chunked level set
each cap to the 95th percentile of the achieved-iteration histogram from QC.

### 2.8 Feature channels

anatomix maps an intensity chunk to 16 L2-normalised channels; multichannel
LNCC then approximates a local cosine similarity between feature vectors. Dense
features are never stored: each register task extracts them from its two
padded chunks and discards them. Normalisation is per volume and global
(percentiles from the coarsest level, stored in attrs). The extractor's
receptive-field radius `r_f` enters the halo bound; it is **measured** by
`calibrate`, not assumed (default 24 until measured). Channel count is the
primary cost lever: registration work scales linearly with channels, so a PCA
projection to `C_pca` channels fitted once at level 0 on a sample of feature
voxels is offered as a control, with intensity (`C = 1`) as the floor.

### 2.9 Memory and work model

Fixed chunk profile ⇒ fixed peak memory:

```
mem  ≈  B · C · (core + 2h)³   + ~2 GB        B ≈ 90 B per voxel-channel (75 with checkpointing)
```

| Profile | core / halo | padded | C | Peak memory | GPU class |
|---|---|---|---|---|---|
| `a16` | 256 / 48 | 352³ | 16 | 63 GB (52 ckpt) | 80 GB |
| `a8` | 256 / 48 | 352³ | 8 PCA | 31 GB | 40 GB |
| `a4` | 256 / 48 | 352³ | 4 PCA | 16 GB | 24 GB |
| `i1` | 256 / 48 | 352³ | 1 intensity | 4 GB | any |

Work, summed over levels and reported by the planner:

```
work = Σ_k  V_k · overhead · C · iters_k · N ,    overhead = ((core+2h)/core)³ = 2.60
```

Overhead is a constant 2.60 for all chunked levels; that is the price of a
fixed 48-voxel halo on a 256 core, and it buys a single memory footprint and
resource spec. Everything else is linear in N.

### 2.10 Storage model

- Every volumetric object is zarr v3 with the sharding codec. Images: inner
  chunk 64³, **shard 256³ = one chunk core**. Fields: inner 32³, shard 128³ on
  the lattice (= one core at lattice factor 2). One shard is one file.
- Subject stores are multiscale groups `s0 … sK` built at ingest by 2× mean
  pooling; attrs hold normalisation percentiles, mask statistics, provenance.
- Fields are float16 mm on a lattice coarser than the level by
  `lattice_factor` (default 2), upsampled trilinearly on read.
- Task results are zarr too: `task_<t>.zarr` of shape `(n, 3, 176, 176, 176)`
  float16 (padded chunk on the lattice), one shard, written to node-local
  scratch then copied. A bounding-box index in the level manifest maps shards
  to the task arrays that touch them.
- Owner computes: every pass with a chunked output is parallel over output
  shards; a task reads with margins and writes only its own shards. No locks.
- Idempotent tasks with complete markers in zarr attrs; explicit retention.
- Only ingest reads foreign formats; only `export` writes them.

---

## 3. Chunk anatomy

```
        ◄────────────── padded 352 ──────────────►
        ┌──────────────────────────────────────────┐
        │ halo 48        (read, never trusted)     │
        │    ┌──────────────────────────────┐      │       halo budget (native voxels):
        │    │ taper (Hann, 48)             │      │         D_max          12
        │    │   ┌──────────────────────┐   │      │       + s_max·(3+3)    12   (scales [2,1], s_max = 2)
        │    │   │ core 256 = 1 shard   │   │      │       + r_f            24   (measured by calibrate)
        │    │   │ owned by this chunk  │   │      │       = 48
        │    │   └──────────────────────┘   │      │
        │    └──────────────────────────────┘      │
        └──────────────────────────────────────────┘
        neighbour cores' halos reach 48 voxels into this core: blend for this
        shard reads this chunk's result plus up to 26 neighbours' task arrays.
```

Note on `s_max`: with `scales: [4,2,1]` the coarsest scale is 4 and the
smoothing term is `4·6 = 24`, which leaves `48 − 24 − 24 = 0` voxels for
`D_max`. The default profile therefore runs seeded levels at `scales: [2,1]`
(`s_max = 2`, term 12, `D_max = 12`) and relies on the seed for anything
coarser; 2.7's `s_max ≳ 3.4` estimate is conservative for anatomix features,
whose capture range is wider than an intensity LNCC window. `calibrate`
measures the actual capture range and the planner accepts `[4,2,1]` only if the
measured `r_f` and capture leave `D_max ≥ 8`. Users on 80 GB GPUs may instead
choose profile `a16-h64` (halo 64, padded 384³, 82 GB with checkpointing) to
run `[4,2,1]` with `D_max = 16`.

---

## 4. Level schedule for an arbitrary grid

```
K        = ceil(log2(max(shape_native) / core))
spacing_k = spacing_native · 2^(K−k)          k = 0 (coarsest) … K (native)
shape_k   = ceil(shape_native / 2^(K−k))
chunks_k  = Π_axis ceil(shape_k / core)       chunks_0 = 1 by construction
```

For the HiP-CT example (1920 × 2560 × 2560 at 50 µm): `K = 4`, levels 800,
400, 200, 100, 50 µm with 1, 4, 18, 100, 800 chunks per subject. For a
1 mm brain MRI of 256³, `K = 0` and the pipeline is a single whole-volume
anatomix registration. For a 8000³ volume, `K = 5`. Nothing else changes.

Level 0 stages: moments → rigid → affine → greedy `[8,4,2,1]`, unclamped, on
anatomix. The affine is composed into the field as dense displacement so every
downstream consumer sees only vector zarr. Levels 1..K: greedy `[2,1]`,
clamped to `D_max(k)`, seeded.

---

## 5. Controls (robustness and scale)

| Control | Default | Guards against | Where |
|---|---|---|---|
| Fixed chunk profile | `a16` (256/48, 16 ch) | memory surprises; per-level retuning | `config.profile` |
| Derived clamp `D_max(k)` | 12 native voxels | halo bound violation; runaway chunk solutions | planner, register |
| Halo bound validation | from profile + measured `r_f` | silently under-sized halos | `config.validate` |
| Per-level stopping rule | `d99 < 0.5·D_max(k+1)` and `ε‖ū‖ < spacing`; with a fixed template, `d99 < residual_vox·spacing` instead | too few passes (aperture leftovers) or too many (cost) | `pipelines.groupwise` |
| Iteration caps + early stop | `[8,6,4,3,2]` passes; per-scale caps; `tolerance 1e-6` | unbounded runtime | config, engine |
| Chunk retry ladder | folds > 0.1 % or loss diverged → (1) `σ_w × 2`, (2) clamp × 0.5, (3) emit seed and flag | one bad chunk poisoning a level | register |
| Tissue-fraction skip | < 2 % → emit seed | registering air | register |
| Global per-volume normalisation | percentiles 0.5 / 99.5 at level 0 | feature drift across chunks | ingest, features |
| Measured `r_f` and capture | `calibrate` | wrong halo budget for a given backbone | calibrate |
| Channel count `C` | 16; PCA to 8 / 4; intensity 1 | cost; memory class | config, G2 |
| Level QC gate | similarity not worse than previous level; folds < 0.1 % | promoting a regression | `passes.qc` |
| Owner-computes shards | shard = core | write conflicts, locks | store, passes |
| Idempotent tasks | complete marker in attrs | partial reruns, duplicate work | tasks, runners |
| Resubmit by id | `status` lists failed ids; `submit --retry` | array element failures | runners.slurm |
| Node-local scratch | `$TMPDIR` then copy | shared-FS small-write storms | register |
| Retention | keep template, fields, QC; delete task arrays after blend, accumulators after update | disk growth | `clean` |
| Provenance | config hash, git commits, backbone id in every store's attrs | irreproducible outputs | store |
| Work model | channel-voxels per level | uninformed cost decisions | planner |

---

## 6. Repository layout

```
scalable-reg/
  pyproject.toml                  package "chunkreg", console script "chunkreg"
  environment.yml
  chunkreg/
    grid.py                       GridSpec, Chunk, pyramid(), tile(), window()
    store.py                      Volume (scalar multiscale zarr), Field (vector zarr), TaskArray,
                                  read_padded, write_shard, ingest, export
    fields.py                     compose, resample, clamp, jacobian, grid_to_disp_mm / disp_mm_to_grid
    features/                     base.py (protocol) · anatomix.py · pca.py · intensity.py · mindssc.py
    engines/                      base.py (protocol, StageSpec) · fireants_greedy.py · fireants_linear.py
    passes/                       manifest.py · register.py · blend.py · update.py · promote.py · qc.py
    pipelines/                    groupwise.py (level loop + stopping rule, pairwise entry) · planner.py
    twoimage.py                   register(fixed, moving, root): the cohort filled in from two paths
    ingest.py                     run grid resolution, placement, link-or-copy, provenance
    cohort.py                     Placement, ResampledSource (scan -> run grid, and back)
    apply.py                      apply_field, resample_to_scan, scan_view, export_store
    runners/                      base.py · local.py · slurm.py · multigpu.py · runner_for
    config.py                     dataclasses, YAML, validation (halo bound, profile, shard alignment)
    calibrate.py                  measure r_f, capture range, B, throughput
    probe.py                      measure d99 at level 0; report
    selftest.py                   known-shift convention test, full and chunked
    cli.py
  configs/                        profiles.yaml · example_hipct.yaml · example_pairwise.yaml
  slurm/                          register.sbatch · blend.sbatch · update.sbatch · promote.sbatch
  scripts/                        legacy
  tests/
  docs/PLAN.md
```

---

## 7. Data types and API

```python
@dataclass(frozen=True)
class GridSpec: shape: tuple[int,int,int]; spacing_mm: float; origin_mm: tuple[float,float,float]
@dataclass(frozen=True)
class Chunk:    id: int; core_origin; core_shape; pad_origin; pad_shape; shard_id: tuple[int,int,int]
@dataclass(frozen=True)
class Profile:  core: int = 256; halo: int = 48; inner_chunk: int = 64; lattice_factor: int = 2
                channels: int = 16; backbone: str = "anatomix"; r_f: int = 24
                k: int = 7; sigma_g: float = 1.0; sigma_w: float = 0.5; scales: tuple = (2, 1)
                def d_max_vox(self) -> int   # h − s_max·((k−1)/2 + 3σ) − r_f
                def padded(self) -> int; def overhead(self) -> float; def mem_gb(self, B) -> float

def pyramid(native: GridSpec, profile: Profile) -> list[GridSpec]        # K+1 levels, coarse → fine
def tile(grid: GridSpec, profile: Profile) -> list[Chunk]

class Volume:   # multiscale scalar zarr group
    open(path) / create(path, native_grid, levels, dtype="u2")
    level(k) -> zarr.Array;  grid(k) -> GridSpec
    read_padded(k, origin, shape, fill=0.0) -> ndarray
    attrs: norm_lo, norm_hi, mask, provenance

class Field:    # vector zarr (3,Z,Y,X) float16 mm, lattice factor in attrs
    open(path) / create(path, grid, profile)
    read_dense(origin, shape) -> Tensor;  write_shard(shard_id, block);  shards()

class TaskArray:   # vector zarr (n,3,p,p,p) float16, single shard, + bbox sidecar in attrs
    create(path, n, profile);  write(i, chunk_id, block);  complete()

class FeatureExtractor(Protocol):
    channels: int; r_f: int
    def setup(self, device) -> None
    def __call__(self, patch, spacing_mm, mask=None) -> Tensor        # (C,z,y,x) normalised

class Engine(Protocol):
    def register(self, fixed, moving, spacing_mm, stages, init_affine=None, device="cuda") -> RegResult
    # RegResult.disp_mm (3,z,y,x) fixed→moving · .affine · .loss_curve · .iters_per_scale

# public
plan(config) -> Plan
calibrate(config, device) -> Calibration         # r_f, capture, B, throughput
probe(config, pairs=3) -> ProbeReport            # d99 at level 0, expected passes per level
build_template(volumes, config, runner) -> (Volume, dict[str, Field])
register_pair(config, fixed, moving, runner) -> RunResult   # a pair out of a cohort
pair_config(config, fixed, moving) -> RunConfig             # the derived config it runs under
register(fixed, moving, root, grid="fixed") -> PairResult    # two paths, no cohort
ingest_subjects(config, say=None) -> (ingested, kept)
save_config(config, path) -> Path                # the inverse of load_config
apply_field(volume, field, out_path, level=0, target=None) -> Volume
resample_to_scan(volume, target, out) -> Volume  # run grid -> a scan's own lattice
export(volume_or_field, path, fmt, target=None)  # nifti | tiff | ome-zarr
selftest(config) -> bool
```

---

## 8. Passes and execution

| Pass | Parallel over | Reads | Writes |
|---|---|---|---|
| `register` | ranges of (subject, chunk) | template chunk, subject chunk (padded), seed field | one `task_<t>.zarr` |
| `blend` | field shards (= cores) | task arrays touching the shard; subject intensity with margin | `fields/<sid>.zarr` shard; `isum`, `wsum` shard |
| `update` | template shards | `isum`, `wsum` with margin | `template.zarr` shard (recentred, optionally sharpened) |
| `qc` | subjects | template, fields, subjects at a coarse level | metrics in level attrs + PNGs |
| `promote` | shards of level k+1 | template and fields of level k | level k+1 template and seed fields |

Register task: load backbone once; per (subject, chunk): read padded chunks,
apply stored normalisation, skip if tissue < 2 %, read and upsample seed,
pre-warp subject, extract features for both, run engine, clamp, compose with
seed, fold-check, retry ladder if needed, mean-pool to lattice, write into the
task array; write to `$TMPDIR`, copy, mark complete.

Level loop (`pipelines/groupwise.py`):

```
for k in 0..K:
    seed ← promote(k−1) or (identity, initial template = mean of moments-aligned volumes) if k == 0
    for it in 1..cap[k]:
        register; blend; update; qc
        if stop_rule(k): break
    gate(k)            # similarity ≥ previous level, folds < 0.1 %; else halt with report
    clean(k, it)
```

Runners: `LocalRunner` (one process per GPU, work queue) and `SlurmRunner`
(one sbatch per pass, arrays chained with `afterok`, `sacct` polling, retry by
id). Resource spec is constant because the profile is: register = 1 GPU of the
profile's class, 8 CPUs, 64 GB RAM; blend/update = 8 CPUs, 32 GB RAM.

---

## 9. Configuration

```yaml
root: /scratch/proj/cohort
runner: slurm
profile: a16                      # from configs/profiles.yaml; fixes core/halo/channels/memory
backbone: anatomix                # anatomix | anatomix-dev | anatomix-dev-vit
channels: 16                      # 16 | pca:8 | pca:4 | intensity
subjects: [{id: s01, path: subjects/s01.zarr}, ...]
grid: {spacing_mm: 0.05}          # shape/origin derived at ingest (union bbox after moments) or given

levels:                           # optional overrides; everything below is derived by default
  caps: [8, 6, 4, 3, 2]           # template passes per level, coarse → fine
  level0_stages: [moments, rigid, affine, {greedy: {scales: [8,4,2,1], iterations: [120,80,50,30]}}]
  seeded_stages: [{greedy: {scales: [2,1], iterations: [50,30]}}]
  stop: {residual_frac: 0.5, ubar_vox: 1.0}
  shape_update_step: 0.25
  sharpen_laplacian_levels: [0, 1]

retry: {fold_frac: 0.001, ladder: [sigma_w_x2, clamp_x0.5, emit_seed]}
retention: {keep_history_levels: [0, 1], delete_task_arrays: true}
slurm:
  partition: gpu
  register: {gpus: 1, cpus: 8, mem_gb: 64, time: "04:00:00", chunks_per_task: 16}
  blend:    {cpus: 8, mem_gb: 32, time: "02:00:00"}
  update:   {cpus: 8, mem_gb: 32, time: "01:00:00"}
```

Validation: profile halo bound with the calibrated `r_f`; `D_max ≥ 8` voxels;
shard = core; channel count fits the GPU class; stage lists well-formed.

---

## 10. CLI

```
chunkreg setup     CONFIG [--stop-at L]   # ingest onto the run grid, conventions, calibration, cohort spread, plan
chunkreg selftest  CONFIG
chunkreg run       CONFIG [--stop-at L] [--from-level k]   # the run itself; resumes on its own,
                                                 # --from-level k redoes k and finer
chunkreg pair      CONFIG --fixed ID --moving ID [--root DIR] [--stop-at L] [--from-level k]
                                                 # derives a two-subject config under
                                                 # <root>/pairs/<fixed>__<moving>/ and runs there
chunkreg run-task  CONFIG --level k --iter i --pass P --id T      # what sbatch invokes
chunkreg status    CONFIG [--retry]
chunkreg apply     VOLUME FIELD OUT [--on STORE]  # --on delivers on that scan's own lattice
chunkreg export    STORE OUT.nii.gz|OUT.tif [--on STORE]
chunkreg clean     CONFIG --level k --iter i
```

---

## 11. Worked example: HiP-CT cohort, N = 10, 50 µm, profile a16

| Level | Spacing | Shape (Z,Y,X) | Voxels | Chunks/subject | D_max | Passes (cap) |
|---|---|---|---|---|---|---|
| 0 | 800 µm | 120 × 160 × 160 | 3.1 M | 1 | unclamped | ≤ 8 |
| 1 | 400 µm | 240 × 320 × 320 | 24.6 M | 4 | 4.8 mm | ≤ 6 |
| 2 | 200 µm | 480 × 640 × 640 | 197 M | 18 | 2.4 mm | ≤ 4 |
| 3 | 100 µm | 960 × 1280 × 1280 | 1.57 G | 100 | 1.2 mm | ≤ 3 |
| 4 | 50 µm | 1920 × 2560 × 2560 | 12.6 G | 800 | 0.6 mm | ≤ 2 |

Work at the caps' typical realised passes (6, 4, 3, 2, 1), overhead 2.60,
N = 10, and GPU-hours at 1.5 × 10⁶ channel-voxels/s (to be replaced by the
calibrated throughput):

| Channels | Work (10¹² ch-vox) | GPU-hours | 8 GPUs | 64 GPUs |
|---|---|---|---|---|
| 16 (anatomix) | 6.8 | 1,260 | 6.6 d | 20 h |
| 8 (PCA) | 3.4 | 630 | 3.3 d | 10 h |
| 4 (PCA) | 1.7 | 315 | 39 h | 5 h |
| 1 (intensity) | 0.43 | 79 | 10 h | 1.2 h |

Level 4 is 76 % of the work; level 3 is 19 %. Feature extraction (two 352³
UNet passes per chunk) is independent of `C` and adds roughly 15 GPU-hours per
level-4 pass. G2 picks the channel count on 20 seeded level-4 chunks.

Storage, steady state: subject stores ~1,350 files / ~200 GB; fields ~1,250
files / ~110 GB; templates ~130 files / ~30 GB; transient per level-4 pass
≤ 600 task arrays / ~260 GB. Total ~3,000 files, ~350 GB, versus ~216,000 files
and 3.8 TB per pass for the legacy script.

---

## 12. QC and gates

Per (level, pass, subject) in the level's zarr attrs and a parquet mirror:
masked similarity to template, fold fraction on the lattice, residual `d99`
(the stopping-rule input), `ε‖ū‖₉₉`, template gradient energy, achieved
iterations per chunk per scale, retry-ladder events. Thumbnails: three mid
slices of template and checkerboard per level and pass.

- **G0** `calibrate` and `probe` run; `plan` prints the pyramid, clamps and work.
- **G1** `selftest` passes, full and chunked, features on and off.
- **G2** Channel count: 2 subjects at level 1 and 20 seeded level-4 chunks;
  16 / pca:8 / pca:4 / intensity by similarity gain, folds, 20 landmarks.
- **G3** One pass at level 2 on the cluster: throughput, memory, achieved
  iteration histogram → caps; resubmission of a killed element works.
- **G4** Pilot: 3 subjects to level 3; file count and bytes against budget.
- **G5** Full run; deliver only if the level-4 gate passes.

---

## 13. Work breakdown

1. `grid.py` (pyramid, tile, window), `fields.py`, tests; port from legacy.
2. `store.py`: Volume, Field, TaskArray, ingest, export. Accept: 12.6 Gvox
   ingest ≤ 140 files; padded reads across shard edges; task array round-trip.
3. `features/anatomix.py`, `pca.py`, `intensity.py`; `engines/`;
   `selftest.py` (G1).
4. `config.py`, `Profile`, `planner.py`; `calibrate.py`, `probe.py` (G0).
5. `passes/manifest.py`, `register.py` with retry ladder; `LocalRunner`.
   Accept: chunked selftest end to end.
6. `blend.py`, `update.py`, `promote.py`; `pipelines/groupwise.py` with the
   stopping rule and gate. Accept: 3 synthetic subjects (known smooth
   deformations of one volume) recover inverse fields within 0.5 voxel and the
   loop stops at the predicted pass count.
7. `runners/slurm.py`, sbatch templates, `status --retry`, `clean`.
8. `passes/qc.py`; G2, G3.
9. Worked example (G4, G5).

Environment: Python 3.11; torch matching the cluster CUDA; zarr ≥ 3,
numcodecs, MONAI, SimpleITK, tifffile, nibabel, pyarrow, PyYAML; anatomix from
git; FireANTs via `install_fireants.sh` with fused ops built on a GPU node. Pin
in `environment.yml`; record commits and backbone id in every store's attrs.

---

## 14. Implementation notes: where the code differs from this design

Written against the implementation in `chunkreg/`. Each item is a place where
building the thing revealed the design note to be wrong, under-specified, or
right in principle but unworkable as stated. The code is the authority; this
section exists so the difference is recorded rather than discovered later.

**Displacement components are stored in `(z, y, x)`, not `(x, y, z)`.**
Section 2.1 proposed matching `grid_sample`'s last axis. In practice axis order
is used by every function in `fields.py` and every blockwise read in `store.py`,
while `grid_sample` is touched in exactly one place. Storing components in
array-axis order confines the reversal to `grid_to_disp_mm` and
`disp_mm_to_grid` and removes a whole class of silent transposition bugs.

**Task arrays are one array per task but one shard per entry.** Section 2.10
said one file per task. A blend task needs the residuals of its own chunk plus
up to 26 neighbours, and those neighbours are scattered across many task arrays
because tasks are ranges over (subject, chunk) while neighbours are adjacent in
3D. Sharding per task would make a blend task decompress a whole task array to
reach one entry, roughly fifteen-fold read amplification at the production
profile. Sharding per entry makes the read exact. The task still writes one
array: one directory, one metadata document, one completion marker.

**The seed must be recentred, and this is not optional.** Section 2.6 mentions
subtracting `ε ū` from every seed in passing. It is load-bearing. Without it the
template moves each pass but the stored fields still point at where the template
used to be, so the fields chase the template and the mean displacement grows
instead of decaying by `(1 - ε)` per pass. Implemented as a per-level
`recentre.zarr` that the update pass writes and the next pass's seed read
composes in. Measured effect at the finest level of the synthetic cohort: the
template step fell about fourfold once it was added.

**Stopping statistics come from merged histograms, at the 99th percentile.**
Two corrections. First, percentiles do not average and no task sees a whole
level, so each task returns a fixed-size histogram over shared logarithmic bin
edges (`chunkreg/stats.py`) and the pipeline reads the percentile off the sum.
Taking a maximum across tasks, which is what the first implementation did, turns
a percentile into an outlier detector. Second, the design's 99.9th percentile is
bounded below by the clamp itself for as long as the clamp is active, because a
few voxels sit pinned at it, so the rule can never clear. The default is now the
99th, configurable as `levels.stop.percentile`.

**The stopping signal is the residual, not the field.** The total field is the
full correspondence and never shrinks. What has to fall below the next level's
clamp is the part a pass could not already explain, which is the clamped
residual the engine returned before it was composed onto the seed.

**Level 0 is bounded by the aperture, not left unclamped.** Section 4 says the
coarsest level runs unclamped because it has no halo. That leaves nothing at all
bounding it, and a noisy solution can run away. It is now clamped to
`levels.level0_max_disp_frac` (default 0.25) of the volume's shortest extent: a
displacement approaching the size of the volume cannot be a correspondence,
whatever the similarity says.

**Accumulator and recentre creation is shared state.** Writing them is
owner-computes, but every blend task of a pass needs the same two objects to
exist. The pipeline creates them once before dispatching and the tasks tolerate
losing the race, so the pass is correct whether tasks start together on a
scheduler or one at a time in a pool.

**`r_f` is measured, not declared.** The halo budget subtracts the feature
receptive field before anything is left for displacement, so a wrong value
overdraws the budget silently. `chunkreg setup` measures it directly by
perturbing one voxel and finding how far the feature response changes, and warns
when the profile understates it.

**Manifests are real files, not backend objects.** A scheduler script references
them by path and a person reads them, so they stay on the filesystem even when
arrays go through a backend. The consequence is that a config with `root: "."`
writes into the working directory, which the test suite isolates per test rather
than merely discouraging.

**A CPU reference engine exists.** Not in the design at all. `engines/demons.py`
is a multi-scale symmetric-forces demons implementation in numpy and scipy. It
makes the whole pipeline runnable and testable without a GPU, a CUDA toolkit or
a FireANTs build, and it is what the 143-test suite exercises. It is a stand-in,
not a competitor: the production thresholds in this document are calibrated for
FireANTs, and the reference engine does not reach them on the synthetic cohort.
Tests therefore verify the stopping *mechanism* directly rather than requiring
the stand-in to converge to production tolerances.

**A backend seam exists below the stores.** `chunkreg/backend.py` abstracts
array creation and opening, with zarr as the production backend and an
in-process memory backend for tests. All block arithmetic, lattice conversion
and sharding logic sits above the seam, so exercising it on the memory backend
exercises the zarr path too. Zarr remains the only backend a real run should
use.

**Passes are not always barriers.** Section 8 lists `register` and `blend` as
consecutive passes, and the level loop ran them that way: every task of one
finished before any task of the other started. On a node with several GPUs that
barrier is where most of them go idle, one by one, behind the slowest register
task of the pass, and it is paid twice per pass per level. But a blend task's
read set is already known exactly -- it is the property section 2.5 relies on,
that a core reads its own chunk and at most 26 neighbours -- so a core's blend
can start as soon as those are registered rather than when the level is. The
dependency is computed from the chunk lattice before the pass
(`passes/blend.py: blend_depends`) and the multi-GPU runner dispatches both
passes as one graph. A failed register task abandons the blends that read it
instead of letting them open a task array that was never written. `update` is
still a barrier, and correctly so: it reads the accumulators with a margin, so
it depends on every blend rather than on a known few.

**A GPU can hold more than one worker.** Section 2.9 sizes a task's memory so
that one fits a device of the profile's class, which quietly assumes the
device's occupancy is the task's to fill. It is not: a worker spends part of
every task reading a chunk, writing a task array and decompressing shards,
with the GPU idle throughout. `workers_per_gpu` puts a second worker on a card
to cover the first's reads. It is off by default, because whether two tasks fit
depends on `bytes_per_voxel_channel`, which is a declared constant until
`setup` measures it, and an out-of-memory failure mid-level costs more than the
idle gap.

**TF32 and cuDNN autotuning are on.** Not in the design, which does not discuss
precision below the choice of float16 for stored fields. The registration inner
loop is fp32 convolution -- the local correlation and its Gaussian
regularisation -- and both defaults leave throughput on the floor for an
Ampere-or-later device. Ten bits of mantissa is far more than a displacement
field resolves, and the features are already extracted in bfloat16; chunk
geometry is fixed by the profile, so autotuning sees a handful of shapes per
level and reuses each plan for every chunk of that shape. Set in
`xp.configure`, so every entry point gets them, and both are switchable from
the environment for bisecting a numerical difference.

### Verified by the test suite

Conventions (direction, component order, composition, the engine bridge) under
known integer shifts; cores tiling exactly with a blend denominator never below
one; the read set of a blend task being at most 27 chunks; pyramid depth,
chunk counts and clamps matching this document's worked example; store round
trips including the lattice; task idempotency; retention actually deleting
scratch; identical results from serial and parallel runners, including when the
parallel one fuses register and blend; the fused dependency map agreeing
exactly with a scan of every chunk against every other, and a failed register
task abandoning the blends that read it and no others; recovery of a
three-subject synthetic cohort with fields staying diffeomorphic; and, in
`chunkreg selftest`, that a shift solved in tiles and blended agrees with the
same shift solved whole.

**Feature quality is now inspectable, and MIND-SSC exists.** Neither was in the
design. The note assumes throughout that learned features earn their sixteen-fold
cost at every level and provides no way to check, which leaves gate G2 with
nothing to look at. `chunkreg features` renders the same world box at every
level for several subjects, intensity above and features as RGB below, under one
shared projection so the colours are comparable across every panel, with
contrast, effective rank and shown-variance printed per tile. MIND-SSC was
implemented alongside it (twelve channels, pure numpy) so the multichannel path
can be judged on a machine with no GPU, and because its receptive field is
exactly dilation plus patch radius it doubles as a known-answer test for
`chunkreg setup`.

Building it found two defects worth recording. The report originally derived the
pyramid from the config profile rather than from the store, so a volume ingested
under a different profile was indexed by the wrong depth: level 0 of a
three-level store was read as if it were native, the box landed outside the
array, and every tile came back all-zero with zero contrast and a confident
colour map fitted to nothing. It now reads the levels the store actually has and
refuses subjects whose stores disagree. Separately, the contact sheet's legend
said "shared projection" unconditionally, which on a per-level sheet invites
precisely the cross-column comparison that mode cannot support.

### Not yet built

the parquet mirror and gate of `passes/qc.py` (a residual map per subject per level and a slice sheet per pass now exist in `chunkreg/qc.py`; the per-subject similarity, gradient energy and the level gate do not), and
`engines/fireants.py` is written against the documented FireANTs API but has
never been executed, since this machine has no GPU. Run `chunkreg selftest
--engine fireants` first on a machine that does; it is designed to catch
exactly the sign, scale and axis-order mistakes an unexecuted adapter makes.

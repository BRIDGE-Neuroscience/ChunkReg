# Python API

Everything the CLI does is a function call. `import chunkreg` is cheap: only the
grid geometry is imported eagerly, and the rest on first use. Levels are numbered
as in [Concepts](concepts.md#the-pyramid-and-level-numbering): 0 is full
resolution, and every `level` argument and every list indexed by level follows it.

## Two volumes

```python
import chunkreg

result = chunkreg.register(
    fixed="fixed.ome.zarr", moving="moving.ome.zarr", root="work/", progress=print
)
result.export_moving("moving_in_fixed.nii.gz")
```

`register(fixed, moving, root, grid="fixed", profile="a16", spacing_mm=None,
fixed_spacing_mm=None, moving_spacing_mm=None, stop_at=None, device="auto",
engine="auto", gpus="all", workers=1, runner=None, reingest=False,
from_level=None, progress=None, **overrides) -> PairResult`

Ingests both volumes, holds the fixed one as the template, runs the level loop, and
returns. Call it again with the same arguments to resume. `grid` is `"fixed"` (the
fixed volume's sampling is the run grid), `"finest"`, `"coarsest"`, a spacing in mm,
or a `grid` block as a mapping. `overrides` reach any other config key:
`levels={"caps": [2, 3, 4]}` (listed from level 0), `features={"mind": False}`.

| `PairResult` | |
|---|---|
| `.field` | The displacement on the run grid, in mm, mapping fixed to moving coordinates. |
| `.fixed`, `.moving` | The subject `Volume`s. |
| `.config`, `.config_path`, `.root` | The config the run used, where it was saved, and the run root. |
| `.summary()` | The per-pass table. |
| `.warp_moving(out, on_fixed_lattice=True, profile=None) -> Volume` | Write the moving volume on the fixed one, as OME-Zarr. |
| `.export_moving(out, on_fixed_lattice=True) -> Path` | The same as NIfTI or TIFF, keeping the intermediate store under the root. |

## A cohort

```python
cfg = chunkreg.load_config("run.json")
chunkreg.ingest_subjects(cfg, say=print)
print(chunkreg.plan(cfg, native_grid).format())

runner = chunkreg.get_runner(
    "multigpu", cfg=cfg, gpus=["0", "1"], device="cuda", config_path="run.json"
)
try:
    result = chunkreg.build_template(cfg, runner=runner, config_path="run.json")
finally:
    runner.close()
print(result.summary())
```

`native_grid` is the run grid; `chunkreg.ingest.native_grid(cfg)` reads it.

| Function | Returns |
|---|---|
| `load_config(source)` | `RunConfig` from a JSON or YAML path, or a mapping. Validates; raises `ConfigError`. |
| `save_config(cfg, path)` | The path written; a file `load_config` reads back. |
| `get_profile(name, **overrides)` | A `Profile` from a preset (`a16`, `a8`, `a4`, `i1`). |
| `ingest_subjects(cfg, reingest=False, jobs=4, say=None)` | `(ingested, kept)` subject ids. |
| `plan(cfg, native_grid)` | A `Plan`; `.format(n_gpus=(8, 64))` is the table `setup` prints. |
| `calibrate(cfg, engine=None, stages=None)` | A `Calibration`: receptive field, memory and speed. |
| `probe(cfg, pairs=3, engine=None)` | A `ProbeReport`: residual at the coarsest level. |
| `selftest(engine="demons", features="intensity", spacing_mm=0.05, verbose=False)` | `True` if the displacement conventions hold. |
| `get_runner(name, **kwargs)` | A runner: `"local"`, `"multigpu"` or `"slurm"`. A multi-GPU runner needs `config_path`, since it addresses tasks by config file. |
| `build_template(cfg, runner=None, engine=None, from_level=None, progress=None, config_path=None)` | `RunResult`. Resumes unless `from_level` is given. |
| `register_pair(cfg, fixed, moving, runner=None, engine=None, root=None, config_path=None, progress=None, from_level=None)` | `RunResult` for two subjects of a cohort. |
| `pair_config(cfg, fixed, moving, root=None)` | The derived two-subject `RunConfig`. |

`RunResult` has `.template` (a `Volume`), `.fields` (subject id to `Field`), `.levels`
(a `LevelReport` per level, each with its `PassReport`s), `.n_passes` and `.summary()`.

## Applying and exporting

| Function | Does |
|---|---|
| `apply_field(volume, field, out, level=None, profile="a16", backend=None, target=None)` | Warp a volume through a field into a new store. `target` delivers on a scan's own lattice. |
| `export_store(store, out, level=None, backend=None, target=None)` | Write a level (default 0) as NIfTI or TIFF, a group of planes at a time. |
| `resample_to_scan(volume, target, out, level=None, profile="a16", backend=None)` | Write a store back onto the lattice a scan arrived on. |
| `scan_view(volume, target, level=None, backend=None)` | The same as an array-like view that streams. |

`volume`, `field` and `store` are paths or opened objects.

## Stores and geometry

| Name | |
|---|---|
| `Volume` | A scalar multiscale store. `Volume.open(path)`, `.n_levels`, `.grid(level)`, `.array(level)`, `.normalisation`. |
| `Field` | A displacement store, `(3, Z, Y, X)` in mm. `Field.open(path)`, `.level_grid`. |
| `GridSpec(shape, spacing_mm, origin_mm)` | An isotropic grid, `(z, y, x)`. `.coarsened(f)` gives the grid mean-pooled by `f`. |
| `Profile` | Core, halo, channels, kernel and smoothing; `.d_max_vox()` is the clamp. |
| `pyramid(native, profile)` | The `GridSpec` of every level, indexed by level (index 0 is native). |
| `pyramid_depth(shape, core)` | `K`, the coarsest level number. |
| `tile(grid, profile)`, `window`, `chunks_touching`, `shard_grid` | Chunk geometry and blend windows. |
| `Placement` | Where a scan sits on the run grid. |
| `RunConfig` | The frozen config; `.levels`, `.grid`, `.retention` and the other blocks mirror [Configuration](configuration.md). |

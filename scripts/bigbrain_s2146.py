"""BigBrain (histology) onto S-21-46 (HiP-CT) at 20 um: prepare, check, launch.

Nothing is resampled on disk. The fixed store's own OME pyramid is linked level
by level, and BigBrain is read through the pairwise affine from its own frame
every time a block is needed (see ``chunkreg.affine_view``). Preparing a run
root therefore writes metadata only and takes seconds, whatever the resolution.

    # the run root(s), one per chunk core to try, plus their configs
    python scripts/bigbrain_s2146.py prepare --root /data/bb_s2146 \
        --fixed /data/S-21-46_HiP-CT_20um.ome.zarr \
        --moving /data/BigBrain2015_coronal_20um.ome.zarr \
        --pair pair/T_pair_zyx_um.json --core 256 192

    # per-level chunk counts and a time projection for a measured seconds/chunk
    python scripts/bigbrain_s2146.py estimate /data/bb_s2146/c256/config.json --seconds-per-chunk 30 --gpus 8

    # memory and speed of one padded chunk on this GPU (needs no data)
    python scripts/bigbrain_s2146.py memcheck --core 256

    # local check at 80 um against the resampled volume made the old way
    python scripts/bigbrain_s2146.py check80 --root /tmp/bb80 \
        --fixed .../S-21-46_HiP-CT_80um.ome.zarr --moving .../BigBrain2015_coronal_80um.ome.zarr \
        --pair pair/T_pair_zyx_um.json --reference .../BigBrain2015_on_S-21-46_linear_80um.ome.zarr/0

Then ``chunkreg setup <root>/c256/config.json --no-probe`` (its calibration is
the one-chunk memory check on the real card) and ``chunkreg run`` the same
config; ``sge/bigbrain_s2146_20um.qsub`` does both and falls back to the
smaller core if the larger one does not fit.

Conventions: ``T_pair_zyx_um.json`` holds ``T_pair_bigbrain_to_s2146_zyx_um``,
BigBrain physical um -> S-21-46 physical um; the run grid is the fixed scan's
own sampling centred on the world origin, exactly as ``chunkreg setup`` places
a ``grid.reference`` subject.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from chunkreg import affine_view as _av
from chunkreg import backend as _backend
from chunkreg.config import get_profile, load_config, save_config
from chunkreg.grid import GridSpec, pyramid, tile
from chunkreg.store import Volume, link_affine_source, link_pyramid

PROFILE_OVERRIDES = {"halo": 68, "r_f": 32, "sigma_g": 3.0, "sigma_w": 2.0}
FEATURES = {"mind": False, "sample_channels": None, "anatomix_window": 64}
PERCENTILES = (0.0, 99.99)
BACKGROUND = (250.0, 0.0)  # BigBrain: white and its 255 fill become zero

COARSEST = {
    "kind": "greedy", "scales": [8, 4, 2, 1], "iterations": [240, 160, 100, 60],
    "loss": "cc", "cc_kernel": 7, "lr": 1.0, "smooth_grad_sigma": 1.0, "smooth_warp_sigma": 0.5,
}
COARSEST_ANNEAL = {"passes": 25, "smooth_grad_sigma": [3.0, 1.5], "smooth_warp_sigma": [2.0, 1.0]}
SECOND_ANNEAL = {"passes": 12, "smooth_grad_sigma": [2.5, 1.5], "smooth_warp_sigma": [2.0, 1.0]}


def _seeded(iters, sigma_g=1.5, sigma_w=1.0):
    return {
        "kind": "greedy", "scales": [2, 1], "iterations": list(iters), "loss": "cc",
        "cc_kernel": 7, "lr": 1.0, "smooth_grad_sigma": sigma_g, "smooth_warp_sigma": sigma_w,
    }


def level_plan(spacings_um, args):
    """Caps and stages per level, chosen by the level's spacing, coarse first.

    The coarsest level is one chunk and runs the deep in-chunk pyramid with
    annealed smoothing; the next level is annealed too (it is still a handful
    of chunks). Everything finer refines a seed, and from 80 um down the
    iteration counts and caps are what fits a night: the plan is one or two
    passes at 40 um and one at 20 um.
    """
    caps, stages, anneal = [], {}, {}
    for k, um in enumerate(spacings_um):
        if k == 0:
            caps.append(args.cap_coarsest)
            stages[k] = [COARSEST]
            anneal[k] = COARSEST_ANNEAL
        elif k == 1:
            caps.append(args.cap_second)
            stages[k] = [_seeded((200, 100), 1.0, 0.5)]
            anneal[k] = SECOND_ANNEAL
        elif um > 100:
            caps.append(4)
            stages[k] = [_seeded((100, 60))]
        elif um > 50:
            caps.append(args.cap_80um)
            stages[k] = [_seeded(args.iters_80um)]
        elif um > 25:
            caps.append(args.cap_40um)
            stages[k] = [_seeded(args.iters_40um)]
        else:
            caps.append(args.cap_20um)
            stages[k] = [_seeded(args.iters_20um)]
    return caps, stages, anneal


def run_grid(fixed_root: Path, dataset: str | None):
    """The fixed scan's sampling, centred on the world origin; and its OME transform."""
    datasets = _av.ome_datasets(fixed_root)
    ds = (min(datasets, key=lambda d: max(d["scale_um"])) if dataset is None
          else next(d for d in datasets if d["path"] == str(dataset)))
    shape = tuple(int(n) for n in ds["shape"])
    spacing_mm = float(max(ds["scale_um"])) / 1000.0
    origin = tuple(-spacing_mm * (n - 1) / 2.0 for n in shape)
    return GridSpec(shape, spacing_mm, origin), ds


def read_pair(path: Path) -> np.ndarray:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    key = next(k for k in doc if k.startswith("T_pair"))
    return np.asarray(doc[key], dtype=np.float64)


def prepare_one(root: Path, core: int, args, grid: GridSpec, fixed_ds: dict,
                world_to_src: np.ndarray, say=print) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    profile = get_profile("a16", core=core, **PROFILE_OVERRIDES)
    grids = pyramid(grid, profile)
    say(f"[{root.name}] core {core}: {len(grids)} levels, "
        + ", ".join(f"{g.spacing_mm * 1000:g} um {'x'.join(map(str, g.shape))}" for g in grids))

    t0 = time.perf_counter()
    fixed_path = root / "subjects" / "fixed.zarr"
    fixed = link_pyramid(
        args.fixed, fixed_path, grid.spacing_mm, profile, origin_mm=grid.origin_mm,
        overwrite=True, percentiles=PERCENTILES,
        provenance={"source": str(args.fixed), "dataset": fixed_ds["path"], "placement": {
            "shape": list(grid.shape), "voxel_mm": [grid.spacing_mm] * 3, "origin_mm": list(grid.origin_mm)}},
        native_dataset=fixed_ds["path"],
        check=not args.trust_pyramid,
    )
    linked = [k for k in range(fixed.n_levels) if fixed.is_linked(k)]
    say(f"  fixed: {fixed.n_levels} levels, linked {linked}, window {fixed.normalisation}, "
        f"{time.perf_counter() - t0:.0f} s")
    for k, why in sorted(((fixed.meta.get("provenance") or {}).get("pyramid_levels_not_linked") or {}).items()):
        say(f"      level {k} of the scan's pyramid not linked: {why}; built (pass --trust-pyramid to link anyway)")

    t0 = time.perf_counter()
    spec = _av.build_spec(args.moving, world_to_src, background=BACKGROUND)
    moving_path = root / "subjects" / "moving.zarr"
    moving = link_affine_source(
        spec, moving_path, grid, profile, overwrite=True, percentiles=PERCENTILES,
        provenance={"source": str(args.moving), "pair_affine": str(args.pair), "placement": {
            "shape": list(grid.shape), "voxel_mm": [grid.spacing_mm] * 3, "origin_mm": list(grid.origin_mm)}},
    )
    say(f"  moving: affine view of {args.moving}, datasets "
        f"{[moving.array(k).dataset['path'] for k in range(moving.n_levels)]}, "
        f"window {moving.normalisation}, {time.perf_counter() - t0:.0f} s")

    _backend.write_json(root / "grid.json", {
        "shape": list(grid.shape), "spacing_mm": grid.spacing_mm, "origin_mm": list(grid.origin_mm),
        "align": "centre",
        "placements": {sid: {"shape": list(grid.shape), "voxel_mm": [grid.spacing_mm] * 3,
                             "origin_mm": list(grid.origin_mm)} for sid in ("fixed", "moving")},
    })

    spacings = [g.spacing_mm * 1000 for g in grids]
    caps, stages, anneal = level_plan(spacings, args)
    raw = {
        "root": str(root),
        # spacing_mm is informational here (the stores carry their grids) but
        # the evaluation scripts read it off the config.
        "subjects": [{"id": "fixed", "path": "subjects/fixed.zarr", "spacing_mm": [grid.spacing_mm] * 3},
                     {"id": "moving", "path": "subjects/moving.zarr", "spacing_mm": [grid.spacing_mm] * 3}],
        "profile": "a16",
        "profile_overrides": {"core": core, **PROFILE_OVERRIDES},
        "features": FEATURES,
        "template_subject": "fixed",
        "device": "cuda",
        "gpus": "all",
        "gpu_mem_gb": args.gpu_mem_gb,
        "grid": {"reference": "fixed"},
        "ingest": {"percentiles": list(PERCENTILES)},
        "levels": {
            # Plateau: once the anneal schedule has run, stop when d99 has not
            # fallen by 5% over the last 3 passes (it wanders once converged).
            "stop": {"require_subvoxel": True, "plateau": {"passes": 3, "rel": 0.05}},
            "caps": caps,
            "min_tissue_fraction": 0.02,
            "level_stages": {str(k): v for k, v in stages.items()},
            "anneal": {str(k): v for k, v in anneal.items()},
        },
    }
    cfg = load_config(raw)
    for w in cfg.validate():
        say(f"  warning: {w}")
    path = save_config(cfg, root / "config.json")
    say(f"  config {path}")
    return path


def cmd_prepare(args) -> int:
    grid, fixed_ds = run_grid(Path(args.fixed), args.fixed_dataset)
    T = read_pair(Path(args.pair))
    world_to_src = _av.world_to_source_um(grid, fixed_ds["scale_um"], fixed_ds["translation_um"], T)
    print(f"run grid {'x'.join(map(str, grid.shape))} @ {grid.spacing_mm:g} mm, origin {grid.origin_mm}")
    as_mm = world_to_src.copy()
    as_mm[:3] /= 1000.0
    print("run world mm -> BigBrain physical mm:")
    print(np.array2string(as_mm, precision=5, suppress_small=True))
    root = Path(args.root)
    for core in args.core:
        prepare_one(root / f"c{core}", core, args, grid, fixed_ds, world_to_src)
    return 0


# --------------------------------------------------------------------------- #
# Estimates
# --------------------------------------------------------------------------- #
def chunk_counts(cfg, say=print):
    """Chunks per level, and how many of them hold tissue, from the coarsest level."""
    fixed = Volume.open(cfg.subject_path("fixed"), cfg.backend)
    coarse = np.asarray(fixed.array(0)[:]) > 0
    grids = fixed.grids()
    rows = []
    for k, g in enumerate(grids):
        step = 2**k  # level-k voxels per coarsest voxel
        chunks = tile(g, cfg.profile)
        with_tissue = 0
        for c in chunks:
            lo = [o // step for o in c.pad_origin]
            hi = [-(-(o + s) // step) for o, s in zip(c.pad_origin, c.pad_shape)]
            box = coarse[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            if box.size and box.mean() >= cfg.levels.min_tissue_fraction:
                with_tissue += 1
        rows.append((k, g.spacing_mm * 1000, len(chunks), with_tissue, cfg.levels.cap(k)))
    return rows


def cmd_estimate(args) -> int:
    cfg = load_config(args.config)
    rows = chunk_counts(cfg)
    print(f"{'lvl':>3} {'um':>6} {'chunks':>7} {'tissue':>7} {'cap':>4} {'pass h':>8} {'level h':>8}")
    total = 0.0
    for k, um, n, t, cap in rows:
        stages = cfg.levels.stages(k)
        iters = sum(sum(s.iterations) for s in stages if s.is_deformable)
        # seconds per chunk scales with iterations relative to the finest level's schedule
        fin = cfg.levels.stages(rows[-1][0])
        fin_iters = sum(sum(s.iterations) for s in fin if s.is_deformable)
        per_chunk = args.seconds_per_chunk * (iters / max(fin_iters, 1)) if n > 1 else args.seconds_per_chunk
        empty = n - t
        pass_s = (t * per_chunk + empty * args.seconds_per_empty) / args.gpus
        level_s = pass_s * cap
        total += level_s
        print(f"{k:>3} {um:>6g} {n:>7} {t:>7} {cap:>4} {pass_s / 3600:>8.2f} {level_s / 3600:>8.2f}")
    print(f"total at the caps: {total / 3600:.1f} h on {args.gpus} GPU(s) "
          f"(+ promotes, settling and QC, ~10-15%)")
    return 0


def cmd_memcheck(args) -> int:
    """One padded chunk through the real extractor and engine on this GPU."""
    import tempfile

    from chunkreg import xp
    from chunkreg.calibrate import calibrate

    with tempfile.TemporaryDirectory() as tmp:
        raw = {
            "root": tmp,
            "subjects": [{"id": "a", "path": "a.zarr"}, {"id": "b", "path": "b.zarr"}],
            "profile": "a16",
            "profile_overrides": {"core": args.core, **PROFILE_OVERRIDES},
            "features": FEATURES,
            "template_subject": "a",
            "device": "cuda",
            "gpu_mem_gb": args.gpu_mem_gb,
        }
        cfg = load_config(raw)
        dev = xp.configure("cuda")
        stage = _seeded(args.iters_20um)
        from chunkreg.config import StageSpec

        cal = calibrate(cfg, stages=[StageSpec(**stage)])
        print(cal.format())
        import torch

        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        peak = cal.bytes_per_voxel_channel * cfg.profile.channels * cfg.profile.padded**3 / 1024**3
        print(f"  peak for a {cfg.profile.padded}^3 chunk: {peak:.1f} GB of {total:.0f} GB on {dev}")
        print(f"  {'FITS' if peak < 0.85 * total else 'TOO BIG'} (needs < 85% of the card)")
        print(f"  seconds per 20 um chunk with iterations {args.iters_20um}: {cal.seconds_per_chunk:.1f}")
        return 0 if peak < 0.85 * total else 2


# --------------------------------------------------------------------------- #
# Local check at 80 um
# --------------------------------------------------------------------------- #
def cmd_check80(args) -> int:
    """Does the affine view reproduce the resampled volume made the old way?"""
    import zarr

    grid, fixed_ds = run_grid(Path(args.fixed), args.fixed_dataset)
    T = read_pair(Path(args.pair))
    world_to_src = _av.world_to_source_um(grid, fixed_ds["scale_um"], fixed_ds["translation_um"], T)
    spec = _av.build_spec(args.moving, world_to_src, background=BACKGROUND)
    view = _av.AffineArray(spec, grid)
    ref = zarr.open_array(str(args.reference), mode="r")
    print(f"view {view}; reference {ref.shape} {ref.dtype}")
    if tuple(ref.shape) != view.shape:
        print("SHAPE MISMATCH"); return 1
    rng = np.random.default_rng(0)
    n = int(args.block)
    worst = 0.0
    for i in range(int(args.blocks)):
        # boxes centred in the volume, where the brain is
        c = [int(rng.integers(s // 4, 3 * s // 4 - n)) for s in view.shape]
        key = tuple(slice(a, a + n) for a in c)
        t0 = time.perf_counter()
        a = view[key].astype(np.float32)
        dt = time.perf_counter() - t0
        b = np.asarray(ref[key], dtype=np.float32)
        mask = (a > 0) | (b > 0)
        if not mask.any():
            print(f"  block {c}: empty in both"); continue
        mae = float(np.abs(a - b)[mask].mean())
        corr = float(np.corrcoef(a[mask], b[mask])[0, 1]) if mask.sum() > 10 else float("nan")
        cover = float((a > 0).mean()), float((b > 0).mean())
        worst = max(worst, mae)
        print(f"  block {c}: MAE {mae:.2f} (of 255), corr {corr:.4f}, tissue view/ref "
              f"{cover[0]:.3f}/{cover[1]:.3f}, read {dt:.2f} s")
    if args.cluster_affine:
        doc = json.loads(Path(args.cluster_affine).read_text())
        A = np.asarray(doc["affine_zyx_mm"])
        # that file is for the 20 um run grid; rebuild ours for that grid to compare
        g20 = GridSpec((7375, 7465, 7465), 0.02, tuple(-0.02 * (n - 1) / 2 for n in (7375, 7465, 7465)))
        mine = _av.world_to_source_um(g20, [20.0] * 3, [0.0] * 3, T)
        mine_mm = mine.copy(); mine_mm[:3] /= 1000.0  # um -> mm on both sides
        print("max |difference| vs cluster_warmstart affine (run mm -> scan mm):",
              float(np.abs(mine_mm - A).max()))
    return 0 if worst < float(args.tolerance) else 1


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def tuning(p):
        p.add_argument("--iters-80um", type=lambda s: [int(v) for v in s.split(",")], default=[80, 40])
        p.add_argument("--iters-40um", type=lambda s: [int(v) for v in s.split(",")], default=[60, 30])
        p.add_argument("--iters-20um", type=lambda s: [int(v) for v in s.split(",")], default=[40, 20])
        # anneal passes (25, 12) + the plateau window (3) + a few
        p.add_argument("--cap-coarsest", type=int, default=32)
        p.add_argument("--cap-second", type=int, default=18)
        p.add_argument("--cap-80um", type=int, default=2)
        p.add_argument("--cap-40um", type=int, default=1)
        p.add_argument("--cap-20um", type=int, default=1)
        p.add_argument("--gpu-mem-gb", type=float, default=80.0)

    p = sub.add_parser("prepare", help="subject stores, grid record and config per core")
    p.add_argument("--root", required=True)
    p.add_argument("--fixed", required=True, help="S-21-46 OME-Zarr root (NGFF 0.6)")
    p.add_argument("--moving", required=True, help="BigBrain OME-Zarr root")
    p.add_argument("--pair", required=True, help="T_pair_zyx_um.json")
    p.add_argument("--fixed-dataset", default=None, help="run on this dataset of the fixed store (default: finest)")
    p.add_argument("--core", type=int, nargs="+", default=[256, 192])
    p.add_argument("--trust-pyramid", action="store_true",
                   help="link the fixed scan's coarser levels without checking they are 2x means")
    tuning(p)

    p = sub.add_parser("estimate", help="chunk counts and a time projection")
    p.add_argument("config")
    p.add_argument("--seconds-per-chunk", type=float, required=True,
                   help="measured seconds per tissue chunk at the finest level (chunkreg setup prints it)")
    p.add_argument("--seconds-per-empty", type=float, default=1.5)
    p.add_argument("--gpus", type=int, default=8)

    p = sub.add_parser("memcheck", help="one padded chunk on this GPU; no data needed")
    p.add_argument("--core", type=int, default=256)
    tuning(p)

    p = sub.add_parser("check80", help="affine view vs the resampled 80 um volume")
    p.add_argument("--fixed", required=True)
    p.add_argument("--moving", required=True)
    p.add_argument("--pair", required=True)
    p.add_argument("--reference", required=True, help="BigBrain2015_on_S-21-46_linear_80um.ome.zarr/0")
    p.add_argument("--fixed-dataset", default=None)
    p.add_argument("--cluster-affine", default=None, help="bigbrain_affine_runmm_to_scanmm_zyx.json to compare")
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--block", type=int, default=192)
    p.add_argument("--tolerance", type=float, default=6.0, help="max MAE (of 255) to pass")
    p.add_argument("--device", default="auto")

    args = ap.parse_args(argv)
    if args.cmd == "check80":
        from chunkreg import xp

        print(f"device {xp.configure(args.device)}")
    return {"prepare": cmd_prepare, "estimate": cmd_estimate, "memcheck": cmd_memcheck,
            "check80": cmd_check80}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

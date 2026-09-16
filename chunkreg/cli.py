"""Command line entry point.

The ordering of the subcommands is the order a run actually happens in:
``ingest`` once per subject, then ``calibrate`` and ``probe`` to measure the
constants the derived rules need, then ``plan`` to see what the run will cost
before committing to it, then ``template`` or ``pair`` to run it. ``run-task``
is what a scheduler array element invokes and is not normally typed by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="chunkreg",
        description=(
            "Hierarchical chunked co-registration of N massive volumes, with "
            "one fixed chunk geometry at every pyramid level."
        ),
    )
    ap.add_argument("--version", action="version", version=f"chunkreg {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="any format -> multiscale sharded store")
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--spacing", type=float, required=True, help="voxel size in mm")
    p.add_argument("--profile", default="a16")
    p.add_argument("--median", type=float, default=None, help="denoise radius")
    p.add_argument("--dtype", default="uint16")
    p.add_argument("--backend", default="zarr")

    p = sub.add_parser("selftest", help="verify the displacement conventions")
    p.add_argument("config", nargs="?")
    p.add_argument("--engine", default="demons")

    p = sub.add_parser(
        "calibrate", help="measure receptive field, memory and throughput"
    )
    p.add_argument("config")
    p.add_argument("--out", default=None)

    p = sub.add_parser("probe", help="measure the residual at the coarsest level")
    p.add_argument("config")
    p.add_argument("--pairs", type=int, default=3)

    p = sub.add_parser("plan", help="resolve the pyramid, clamps and cost")
    p.add_argument("config")
    p.add_argument("--gpus", type=int, nargs="*", default=[8, 64])

    p = sub.add_parser(
        "features", help="render feature channels as RGB, per subject per level"
    )
    p.add_argument("config")
    p.add_argument("--out", default=None, help="output directory (default: <root>/qc/features)")
    p.add_argument(
        "--at", default=None,
        help="box centre in world mm as Z,Y,X (default: the volume centre)",
    )
    p.add_argument(
        "--size-mm", type=float, default=None,
        help="box width in mm, identical at every level "
             "(default: one chunk core at native resolution)",
    )
    p.add_argument("--levels", default=None, help="comma-separated, default all")
    p.add_argument("--subjects", default=None, help="comma-separated, default all")
    p.add_argument("--plane", default="axial", choices=["axial", "coronal", "sagittal", "all"])
    p.add_argument(
        "--extractor", default=None,
        help="override the profile's features, e.g. mindssc, anatomix, intensity",
    )
    p.add_argument(
        "--warped", action="store_true",
        help="pull each subject through its current field so the anatomy aligns",
    )
    p.add_argument(
        "--basis", default="shared", choices=["shared", "per-level"],
        help="shared: one colour map everywhere, columns comparable. "
             "per-level: each level at its own contrast, columns NOT comparable",
    )
    p.add_argument("--tile-px", type=int, default=192)

    p = sub.add_parser("template", help="build an unbiased group template")
    p.add_argument("config")
    p.add_argument("--from-level", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--engine", default=None)

    p = sub.add_parser("pair", help="register one moving volume to one fixed")
    p.add_argument("config")
    p.add_argument("--fixed", required=True)
    p.add_argument("--moving", required=True)
    p.add_argument("--workers", type=int, default=1)

    p = sub.add_parser("run-task", help="what a scheduler array element invokes")
    p.add_argument("config")
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--iter", type=int, required=True, dest="iteration")
    p.add_argument(
        "--pass", required=True, dest="pass_name", choices=["register", "blend", "update"]
    )
    p.add_argument("--id", type=int, required=True, dest="task_id")
    p.add_argument("--engine", default=None)

    p = sub.add_parser("status", help="progress from task markers")
    p.add_argument("config")

    p = sub.add_parser("apply", help="warp a volume through a field")
    p.add_argument("volume")
    p.add_argument("field")
    p.add_argument("out")
    p.add_argument("--level", type=int, default=0)

    p = sub.add_parser("export", help="store -> nifti or tiff")
    p.add_argument("store")
    p.add_argument("out")

    p = sub.add_parser("clean", help="apply the retention policy")
    p.add_argument("config")
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--iter", type=int, required=True, dest="iteration")

    return ap


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def _load(path):
    from .config import load_config

    return load_config(path)


def _native_grid(cfg):
    from .store import Volume

    first = cfg.subjects[0]
    if Volume.exists(cfg.subject_path(first.id), cfg.backend):
        return Volume.open(cfg.subject_path(first.id), cfg.backend).native_grid
    if cfg.spacing_mm is None:
        raise SystemExit(
            "no ingested subjects found and no spacing_mm in the config, so "
            "the native grid is unknown. Run 'chunkreg ingest' first."
        )
    raise SystemExit(
        f"subject {first.id!r} is not ingested at {cfg.subject_path(first.id)}. "
        f"Run 'chunkreg ingest' first."
    )


def cmd_ingest(args) -> int:
    from .config import get_profile
    from .store import ingest_array
    from .io_formats import read_volume

    data = read_volume(args.src)
    if args.median:
        from scipy import ndimage

        data = ndimage.median_filter(data, size=int(2 * args.median + 1))
    profile = get_profile(args.profile)
    vol = ingest_array(
        data,
        args.dst,
        args.spacing,
        profile,
        dtype=args.dtype,
        backend=args.backend,
        overwrite=True,
    )
    lo, hi = vol.normalisation
    print(
        f"ingested {args.src} -> {args.dst}\n"
        f"  native {vol.native_grid.shape} @ {args.spacing} mm, "
        f"{vol.n_levels} levels, window [{lo:.4g}, {hi:.4g}]"
    )
    return 0


def cmd_selftest(args) -> int:
    from .selftest import run_selftest

    ok = run_selftest(engine=args.engine, verbose=True)
    return 0 if ok else 1


def cmd_plan(args) -> int:
    from .planner import plan

    cfg = _load(args.config)
    print(plan(cfg, _native_grid(cfg)).format(n_gpus=args.gpus))
    return 0


def cmd_features(args) -> int:
    from .featurereport import build_feature_report, write_feature_report

    cfg = _load(args.config)
    centre = None
    if args.at:
        parts = [float(v) for v in args.at.replace(",", " ").split()]
        if len(parts) != 3:
            raise SystemExit(f"--at needs three world coordinates, got {args.at!r}")
        centre = tuple(parts)

    def ints(s):
        return None if not s else [int(v) for v in s.replace(",", " ").split()]

    def strs(s):
        return None if not s else [v for v in s.replace(",", " ").split()]

    planes = ["axial", "coronal", "sagittal"] if args.plane == "all" else [args.plane]
    out_root = Path(args.out or (cfg.root_path / "qc" / "features"))

    first = None
    for plane in planes:
        report = build_feature_report(
            cfg,
            centre_world=centre,
            size_mm=args.size_mm,
            levels=ints(args.levels),
            subjects=strs(args.subjects),
            plane=plane,
            warped=args.warped,
            extractor_name=args.extractor,
            basis_mode=args.basis,
        )
        target = out_root if len(planes) == 1 else out_root / plane
        written = write_feature_report(report, target, tile_px=args.tile_px)
        if first is None:
            first = report
            print(report.summary())
            print()
        print(f"{plane:>8}: {written['sheet']}")
    print(f"\nmetrics and per-tile PNGs in {out_root}")
    return 0


def cmd_probe(args) -> int:
    from .probe import probe

    cfg = _load(args.config)
    report = probe(cfg, pairs=args.pairs)
    print(report.format())
    return 0


def cmd_calibrate(args) -> int:
    from .calibrate import calibrate

    cfg = _load(args.config)
    cal = calibrate(cfg)
    out = Path(args.out or (cfg.root_path / "calibration.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cal.to_json(), indent=2), encoding="utf-8")
    print(cal.format())
    print(f"\nwritten to {out}")
    return 0


def cmd_template(args) -> int:
    from .engines import get_engine
    from .pipelines import build_template
    from .runners import LocalRunner, get_runner

    cfg = _load(args.config)
    engine = get_engine(args.engine) if args.engine else None
    runner = (
        get_runner("slurm", cfg=cfg)
        if cfg.runner == "slurm"
        else LocalRunner(workers=args.workers)
    )
    result = build_template(
        cfg,
        runner=runner,
        engine=engine,
        from_level=args.from_level,
        progress=print,
        config_path=args.config if cfg.runner == "slurm" else None,
    )
    print()
    print(result.summary())
    return 0


def cmd_pair(args) -> int:
    from .pipelines import register_pair
    from .runners import LocalRunner

    cfg = _load(args.config)
    result = register_pair(
        cfg, args.fixed, args.moving, runner=LocalRunner(workers=args.workers)
    )
    print(result.summary())
    return 0


def cmd_run_task(args) -> int:
    """One scheduler array element: re-derive the manifest and run one id."""
    from .engines import get_engine
    from .grid import pyramid
    from .passes import run_blend_task, run_register_task, run_update_task
    from .passes.manifest import Manifest

    cfg = _load(args.config)
    path = cfg.level_dir(args.level) / f"manifest_it{args.iteration}.json"
    if not path.exists():
        raise SystemExit(
            f"no manifest at {path}. The pipeline writes it before dispatching "
            f"a pass; run 'chunkreg template' rather than invoking run-task by hand."
        )
    manifest = Manifest.load(path)

    if args.pass_name == "register":
        engine = get_engine(args.engine) if args.engine else None
        out = run_register_task(cfg, manifest, args.task_id, engine=engine)
    elif args.pass_name == "blend":
        out = run_blend_task(cfg, manifest, args.task_id)
    else:
        out = run_update_task(cfg, manifest, args.task_id)
    print(json.dumps({k: v for k, v in out.items() if k != "records"}, default=str))
    return 0


def cmd_status(args) -> int:
    from .status import status

    print(status(_load(args.config)))
    return 0


def cmd_apply(args) -> int:
    from .store import Field, Volume
    from . import fields as _fields

    vol = Volume.open(args.volume)
    field = Field.open(args.field)
    grid = field.level_grid
    u = field.read_dense((0, 0, 0), grid.shape)
    block = vol.read_padded(args.level, (0, 0, 0), grid.shape)
    out = _fields.warp(block, u, grid.spacing_mm)
    from .store import ingest_array
    from .config import get_profile

    ingest_array(out, args.out, grid.spacing_mm, get_profile("a16"), dtype="float32", overwrite=True)
    print(f"wrote {args.out}")
    return 0


def cmd_export(args) -> int:
    from .io_formats import write_volume
    from .store import Volume

    vol = Volume.open(args.store)
    grid = vol.grid(vol.n_levels - 1)
    write_volume(args.out, vol.read_padded(vol.n_levels - 1, (0, 0, 0), grid.shape), grid)
    print(f"wrote {args.out}")
    return 0


def cmd_clean(args) -> int:
    from . import backend as _backend
    from .passes.manifest import Manifest

    cfg = _load(args.config)
    b = _backend.get_backend(cfg.backend)
    path = cfg.level_dir(args.level) / f"manifest_it{args.iteration}.json"
    removed = 0
    if path.exists():
        manifest = Manifest.load(path)
        for t in manifest.tasks:
            b.remove(cfg.task_path(args.level, args.iteration, t.task_id))
            removed += 1
    for which in ("isum", "wsum"):
        b.remove(cfg.accumulator_path(args.level, args.iteration, which))
        removed += 1
    print(f"removed {removed} transient objects for level {args.level} "
          f"iteration {args.iteration}")
    return 0


_COMMANDS = {
    "ingest": cmd_ingest,
    "selftest": cmd_selftest,
    "calibrate": cmd_calibrate,
    "probe": cmd_probe,
    "plan": cmd_plan,
    "features": cmd_features,
    "template": cmd_template,
    "pair": cmd_pair,
    "run-task": cmd_run_task,
    "status": cmd_status,
    "apply": cmd_apply,
    "export": cmd_export,
    "clean": cmd_clean,
}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.cmd](args)
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as exc:  # noqa: BLE001 - a CLI should not show a traceback
        print(f"chunkreg {args.cmd}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Command line entry point.

A run is two commands against one configuration file::

    chunkreg setup run.json     # ingest, check, measure, plan
    chunkreg run   run.json     # build the template

``setup`` is everything that has to happen before work can be costed: it
ingests each subject that names a source, verifies the displacement
conventions against the engine build, measures the receptive field, memory and
throughput the derived rules depend on, and prints the resolved pyramid and its
cost. ``run`` executes it. Both are idempotent, so a `setup` that already has
its stores does no work and a `run` resumes from whatever finished.

The rest of the commands are utilities rather than steps in that sequence:
``status`` reports progress, ``apply`` and ``export`` move data out, ``pair``
registers two volumes without a cohort, ``selftest`` checks conventions with no
config at all, and ``run-task`` is what a scheduler array element invokes and
is not normally typed by hand.
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

    p = sub.add_parser(
        "setup",
        help="stage one: ingest, check conventions, measure, and plan",
    )
    p.add_argument("config")
    p.add_argument(
        "--reingest", action="store_true",
        help="re-ingest subjects whose store already exists",
    )
    p.add_argument("--no-ingest", action="store_true", help="skip ingest")
    p.add_argument(
        "--no-selftest", action="store_true", help="skip the conventions check"
    )
    p.add_argument(
        "--no-calibrate", action="store_true",
        help="skip measuring receptive field, memory and throughput",
    )
    p.add_argument(
        "--no-probe", action="store_true",
        help="skip measuring how far apart the cohort is",
    )
    p.add_argument("--pairs", type=int, default=3, help="pairs to probe")
    p.add_argument("--gpus", type=int, nargs="*", default=[8, 64],
                   help="GPU counts to cost the run at")
    p.add_argument("--engine", default=None)
    p.add_argument("--out", default=None, help="where to write calibration.json")

    p = sub.add_parser("run", help="stage two: build the unbiased group template")
    p.add_argument("config")
    p.add_argument("--from-level", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--engine", default=None)

    p = sub.add_parser("selftest", help="verify the displacement conventions")
    p.add_argument("--engine", default="demons")

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

    p = sub.add_parser("pair", help="register one moving volume to one fixed")
    p.add_argument("config")
    p.add_argument("--fixed", required=True)
    p.add_argument("--moving", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--engine", default=None)

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
    p.add_argument(
        "--level", type=int, default=None,
        help="volume level to read (default: the one matching the field's grid)",
    )
    p.add_argument("--profile", default="a16", help="profile for the output store")

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
            "the native grid is unknown. Run 'chunkreg setup' first."
        )
    raise SystemExit(
        f"subject {first.id!r} is not ingested at {cfg.subject_path(first.id)}. "
        f"Run 'chunkreg setup' first."
    )


def _resolve(cfg, rel) -> Path:
    """A config-relative path, left alone if it is already absolute."""
    q = Path(rel)
    return q if q.is_absolute() else cfg.root_path / q


def _fmt_stage(st) -> str:
    if st.kind == "moments":
        return "moments"
    return (
        f"{st.kind:<7} scales={list(st.scales)} iters={list(st.iterations)} "
        f"lr={st.lr:g} grad_sigma={st.smooth_grad_sigma:g} "
        f"warp_sigma={st.smooth_warp_sigma:g} cc_kernel={st.cc_kernel} "
        f"loss={st.loss}"
    )


def _ladder(cfg, levels) -> str:
    """The resolved pyramid: what each level index actually means."""
    from .grid import tile

    rows = [
        f"  {'level':>5} {'spacing':>12} {'shape':>22} {'chunks':>8} {'clamp':>12}"
    ]
    for k, g in enumerate(levels):
        n = len(tile(g, cfg.profile))
        clamp = (
            cfg.levels.level0_max_disp_frac * min(g.extent_mm)
            if n == 1
            else cfg.profile.d_max_mm(g.spacing_mm)
        )
        shape = "x".join(str(v) for v in g.shape)
        rows.append(
            f"  {k:>5} {g.spacing_mm * 1000:>9.1f} um {shape:>22} {n:>8} "
            f"{clamp * 1000:>9.1f} um"
        )
    return "\n".join(rows)


def _stage_table(cfg, n_levels: int) -> str:
    """Stages as each level will actually run them, per-level tuning applied."""
    rows = []
    for k in range(n_levels):
        marks = []
        if k in cfg.levels.level_stages:
            marks.append("level_stages")
        if k in cfg.levels.level_params:
            marks.append("level_params")
        tag = f"   <- {' + '.join(marks)}" if marks else ""
        rows.append(f"  level {k} (cap {cfg.levels.cap(k)}){tag}")
        for st in cfg.levels.stages(k):
            rows.append(f"      {_fmt_stage(st)}")
    return "\n".join(rows)


def _report_stray_overrides(cfg, n_levels: int) -> None:
    from .config import unused_level_overrides

    stray = unused_level_overrides(cfg, n_levels)
    if stray:
        print(
            f"  warning: per-level override(s) for level(s) {stray} will not "
            f"run; this pyramid has levels 0..{n_levels - 1}. The depth "
            f"follows from the grid and the chunk core, not the config."
        )


def _ingest_subjects(cfg, reingest: bool = False) -> tuple[list, list]:
    """Ingest every subject that names a source and has no store yet."""
    from .io_formats import is_zarr, open_source, spacing_agrees
    from .store import Volume, ingest_source

    done, kept = [], []
    for spec in cfg.subjects:
        dst = cfg.subject_path(spec.id)
        present = Volume.exists(dst, cfg.backend)
        if present and not reingest:
            kept.append(spec.id)
            continue
        if spec.source is None:
            if present:
                kept.append(spec.id)
                continue
            if is_zarr(dst):
                raise SystemExit(
                    f"subject {spec.id!r}: {dst} is a zarr but not a chunkreg "
                    f"store. Name it as the subject's 'source' and let 'path' "
                    f"default, and setup will convert it into a sharded store."
                )
            raise SystemExit(
                f"subject {spec.id!r} has no store at {dst} and no 'source' to "
                f"ingest from. Add a 'source' to the subject entry."
            )

        src_path = _resolve(cfg, spec.source)
        if src_path.resolve() == Path(dst).resolve():
            raise SystemExit(
                f"subject {spec.id!r}: 'source' and 'path' are both {dst}; the "
                f"store has to be written somewhere else"
            )
        source = open_source(src_path)

        spacing = cfg.spacing_mm
        if spacing is None:
            spacing = source.spacing_mm
        elif source.spacing_mm is not None and not spacing_agrees(
            spacing, source.spacing_mm
        ):
            raise SystemExit(
                f"subject {spec.id!r}: the config says {spacing} mm but "
                f"{source.description} says {source.spacing_mm:g} mm. Fix one; "
                f"the pipeline will not guess which is right."
            )
        if spacing is None:
            raise SystemExit(
                f"subject {spec.id!r}: {source.description} does not state its "
                f"voxel size; set 'spacing_mm' at the top level of the config."
            )

        vol = ingest_source(
            source,
            dst,
            spacing,
            cfg.profile,
            dtype=cfg.ingest.dtype,
            backend=cfg.backend,
            overwrite=True,
            percentiles=cfg.ingest.percentiles,
            median_radius=cfg.ingest.median_radius,
            provenance={"source": str(src_path), "config": cfg.fingerprint()},
        )
        lo, hi = vol.normalisation
        print(
            f"  {spec.id}: {source.description} -> {dst}\n"
            f"      {vol.native_grid.shape} @ {spacing:g} mm, {vol.n_levels} "
            f"levels, {vol.array(vol.n_levels - 1).dtype}, "
            f"window [{lo:.4g}, {hi:.4g}]"
        )
        done.append(spec.id)
    return done, kept


def _check_subjects(cfg) -> None:
    """Every subject must be a store this profile can run on, on one grid."""
    from .store import Volume, check_volume

    grids = {}
    failures = []
    for spec in cfg.subjects:
        dst = cfg.subject_path(spec.id)
        if not Volume.exists(dst, cfg.backend):
            failures.append(f"{spec.id}: no store at {dst}")
            continue
        vol = Volume.open(dst, cfg.backend)
        errors, warnings = check_volume(vol, cfg.profile)
        failures += [f"{spec.id}: {e}" for e in errors]
        for w in warnings:
            print(f"  warning: {spec.id}: {w}")
        g = vol.native_grid
        grids[spec.id] = (g.shape, round(g.spacing_mm, 9))
    shapes = set(grids.values())
    if len(shapes) > 1:
        detail = ", ".join(f"{k} {v[0]} @ {v[1]:g} mm" for k, v in grids.items())
        failures.append(
            f"subjects are on different grids ({detail}). Every subject has to "
            f"share one shape and spacing; pad or resample them before ingest"
        )
    if failures:
        raise SystemExit(
            "subject stores are not ready:\n  "
            + "\n  ".join(failures)
            + "\nRe-run 'chunkreg setup --reingest' after fixing the sources."
        )
    print(f"  {len(grids)} subject store(s) sharded and on one grid")


def cmd_setup(args) -> int:
    """Stage one: ingest, check conventions, measure, and plan."""
    from .grid import pyramid

    cfg = _load(args.config)
    print(
        f"config {args.config}\n"
        f"  root {cfg.root_path}, profile {cfg.profile_name!r} "
        f"({cfg.profile.channels} channels, {cfg.profile.features}), "
        f"{cfg.n_subjects} subject(s), runner {cfg.runner}, backend {cfg.backend}"
    )
    for w in cfg.validate():
        print(f"  warning: {w}")

    print("\ningest")
    if args.no_ingest:
        print("  skipped (--no-ingest)")
    else:
        done, kept = _ingest_subjects(cfg, reingest=args.reingest)
        if not done:
            print(f"  nothing to do; {len(kept)} subject(s) already ingested")
        elif kept:
            print(f"  {len(done)} ingested, {len(kept)} already present")

    print("\nconventions")
    if args.no_selftest:
        print("  skipped (--no-selftest)")
    else:
        from .selftest import run_selftest

        engine_name = args.engine or "demons"
        if not run_selftest(engine=engine_name, verbose=False):
            print(
                "  FAILED: the engine does not honour the displacement "
                "conventions. Run 'chunkreg selftest' for the detail; do not "
                "start a run against this build."
            )
            return 1
        print(f"  passed on engine {engine_name!r}")

    print("\nsubject stores")
    _check_subjects(cfg)

    native = _native_grid(cfg)
    levels = pyramid(native, cfg.profile)

    print("\ncalibration")
    if args.no_calibrate:
        print("  skipped (--no-calibrate); the config's declared constants stand")
    else:
        from .calibrate import calibrate

        cal = calibrate(cfg, engine=args.engine)
        out = Path(args.out or (cfg.root_path / "calibration.json"))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(cal.to_json(), indent=2), encoding="utf-8")
        print(cal.format())
        print(f"  written to {out}")

    print("\ncohort spread")
    if args.no_probe:
        print("  skipped (--no-probe)")
    else:
        from .probe import probe

        print(probe(cfg, pairs=args.pairs, engine=args.engine).format())

    print("\nresolved pyramid")
    print(_ladder(cfg, levels))
    _report_stray_overrides(cfg, len(levels))
    print("\nstages per level")
    print(_stage_table(cfg, len(levels)))

    print()
    from .planner import plan

    print(plan(cfg, native).format(n_gpus=args.gpus))
    print(f"\nsetup complete. Run it with:  chunkreg run {args.config}")
    return 0


def cmd_run(args) -> int:
    """Stage two: build the template."""
    from .engines import get_engine
    from .grid import pyramid
    from .pipelines import build_template
    from .runners import LocalRunner, get_runner

    cfg = _load(args.config)
    _check_subjects(cfg)
    native = _native_grid(cfg)
    levels = pyramid(native, cfg.profile)

    print(
        f"{cfg.n_subjects} subject(s), {len(levels)} levels, profile "
        f"{cfg.profile_name!r}, runner {cfg.runner}"
    )
    print(_ladder(cfg, levels))
    _report_stray_overrides(cfg, len(levels))
    tuned = [k for k in cfg.levels.tuned_levels() if k < len(levels)]
    if tuned:
        print(f"  per-level tuning active on level(s) {tuned}")
    print()

    engine = get_engine(args.engine) if args.engine else None
    runner = (
        get_runner(
            "slurm",
            cfg=cfg,
            poll_seconds=cfg.slurm.poll_seconds,
            max_wait_s=cfg.slurm.max_wait_s,
        )
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


def cmd_selftest(args) -> int:
    from .selftest import run_selftest

    ok = run_selftest(engine=args.engine, verbose=True)
    return 0 if ok else 1


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


def cmd_pair(args) -> int:
    from .pipelines import register_pair
    from .runners import LocalRunner

    from .engines import get_engine

    cfg = _load(args.config)
    result = register_pair(
        cfg,
        args.fixed,
        args.moving,
        runner=LocalRunner(workers=args.workers),
        engine=get_engine(args.engine) if args.engine else None,
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
            f"a pass; run 'chunkreg run' rather than invoking run-task by hand."
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
    """Warp a volume through a field, on the field's own grid."""
    from . import fields as _fields
    from .config import get_profile
    from .store import Field, Volume, ingest_array

    vol = Volume.open(args.volume)
    field = Field.open(args.field)
    grid = field.level_grid

    # The field defines the grid, so the volume has to be read at the level
    # that matches it. Taking the level from a flag let the two disagree, which
    # produced either a shape error or a silently wrong warp.
    level = args.level
    if level is None:
        candidates = [
            k for k in range(vol.n_levels) if vol.grid(k).shape == grid.shape
        ]
        if not candidates:
            raise SystemExit(
                f"no level of {args.volume} is on the field's grid "
                f"{grid.shape} at {grid.spacing_mm} mm; levels are "
                f"{[vol.grid(k).shape for k in range(vol.n_levels)]}"
            )
        level = candidates[0]
    elif vol.grid(level).shape != grid.shape:
        raise SystemExit(
            f"level {level} of {args.volume} is {vol.grid(level).shape} but the "
            f"field is on {grid.shape}; omit --level to match them automatically"
        )

    u = field.read_dense((0, 0, 0), grid.shape)
    block = vol.read_padded(level, (0, 0, 0), grid.shape)
    out = _fields.warp(block, u, grid.spacing_mm)
    ingest_array(
        out,
        args.out,
        grid.spacing_mm,
        get_profile(args.profile),
        dtype="float32",
        overwrite=True,
    )
    print(f"wrote {args.out} from level {level} of {args.volume}")
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
    # The two-stage workflow.
    "setup": cmd_setup,
    "run": cmd_run,
    # Utilities, not steps in that sequence.
    "selftest": cmd_selftest,
    "features": cmd_features,
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

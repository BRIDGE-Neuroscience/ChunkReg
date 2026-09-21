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
        "--jobs", type=int, default=4,
        help="subjects to ingest at once (default 4); 1 ingests them in turn",
    )
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
    p.add_argument(
        "--stop-at", default=None, metavar="LEVEL",
        help="plan as if the run stops at this level: an index such as 3 or a "
             "spacing such as 200um (overrides levels.stop_at)",
    )

    p = sub.add_parser("run", help="stage two: build the unbiased group template")
    p.add_argument("config")
    p.add_argument(
        "--stop-at", default=None, metavar="LEVEL",
        help="finest level to run: an index such as 3 or a spacing such as "
             "200um (overrides levels.stop_at). Running again with a finer "
             "level carries on from here",
    )
    p.add_argument(
        "--from-level", type=int, default=None, metavar="N",
        help="redo the run from level N, discarding progress at N and finer. "
             "Without it a run resumes from wherever it stopped",
    )
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
    p.add_argument(
        "--root", default=None,
        help="where to write this pair's output (default: "
             "<root>/pairs/<fixed>__<moving>)",
    )
    p.add_argument(
        "--stop-at", default=None, metavar="LEVEL",
        help="finest level to run: an index such as 3 or a spacing such as "
             "200um. Running again with a finer level carries on from here",
    )
    p.add_argument(
        "--from-level", type=int, default=None, metavar="N",
        help="redo from level N, discarding progress at N and finer",
    )
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
    p.add_argument(
        "--on", default=None, metavar="STORE",
        help="resample the result onto the lattice the scan behind STORE "
             "arrived on, instead of leaving it on the run grid",
    )
    p.add_argument(
        "--device", default="auto",
        help="where to warp: auto, cuda, cuda:N or cpu (default: a GPU if there is one)",
    )

    p = sub.add_parser("export", help="store -> nifti or tiff")
    p.add_argument("store")
    p.add_argument("out")
    p.add_argument(
        "--level", type=int, default=None,
        help="pyramid level to write, numbered as the run numbers them "
             "(0 coarsest). Default: the store's finest. A template store "
             "holds a single level, so this is for subject stores, where it "
             "is how a coarse level is pulled out to look at beside a "
             "template of the same level",
    )
    p.add_argument(
        "--on", default=None, metavar="STORE",
        help="write on the lattice the scan behind STORE arrived on, rather "
             "than on the run grid. For a registration onto a fixed volume "
             "that is the fixed subject's store, which puts the result back "
             "on the sampling it is expected in, anisotropic voxels included",
    )

    p = sub.add_parser("clean", help="apply the retention policy")
    p.add_argument("config")
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--iter", type=int, required=True, dest="iteration")

    return ap


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def _ingest_subjects(cfg, reingest: bool = False, jobs: int = 4):
    """``chunkreg.ingest.ingest_subjects``, narrating to stdout."""
    from .ingest import ingest_subjects

    return ingest_subjects(cfg, reingest=reingest, jobs=jobs, say=print)


def _check_subjects(cfg) -> None:
    """``chunkreg.ingest.check_subjects``, narrating to stdout."""
    from .ingest import check_subjects

    check_subjects(cfg, say=print)


def _native_grid(cfg):
    from .ingest import native_grid

    return native_grid(cfg)


def _load(path, stop_at=None):
    """Load a config, with a ``--stop-at`` from the command line applied."""
    from dataclasses import replace

    from .config import load_config, parse_level_ref

    cfg = load_config(path)
    if stop_at is not None:
        ref = parse_level_ref(stop_at, "--stop-at")
        cfg = replace(cfg, levels=replace(cfg.levels, stop_at=ref))
    return cfg


def _configure(cfg) -> str:
    """Pick where this process computes, from the config, and say so."""
    from . import xp

    device = xp.configure(cfg.device)
    where = {
        "cpu": "NumPy on the CPU (reference path)",
        "torch-cpu": "PyTorch on the CPU (testing only)",
    }.get(device, f"PyTorch on {device}")
    print(f"  compute: {where}, engine {cfg.engine_for(device)}")
    return device


def _gpu_slots(cfg, device: str) -> list[str]:
    from .runners import gpu_slots

    return gpu_slots(cfg, device, say=print)


def _make_runner(cfg, device: str, engine_name: str, config_path, workers: int = 1):
    from .runners import runner_for

    return runner_for(
        cfg, device, engine_name, config_path, workers=workers, say=print
    )


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
    """The resolved pyramid: what each level index means, and whether it runs."""
    from .grid import tile

    run = set(cfg.levels.run_levels(levels))
    rows = [
        f"  {'level':>5} {'spacing':>12} {'shape':>22} {'chunks':>8} "
        f"{'clamp':>12}  run"
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
            f"{clamp * 1000:>9.1f} um  {'yes' if k in run else 'skip'}"
        )
    return "\n".join(rows)


def _stage_table(cfg, levels) -> str:
    """Stages as each level that runs will use them, per-level tuning applied."""
    run = cfg.levels.run_levels(levels)
    rows = []
    for k in run:
        marks = []
        if k in cfg.levels.level_stages:
            marks.append("level_stages")
        if k in cfg.levels.level_params:
            marks.append("level_params")
        tag = f"   <- {' + '.join(marks)}" if marks else ""
        rows.append(f"  level {k} (cap {cfg.levels.cap(k)}){tag}")
        for st in cfg.levels.stages(k):
            rows.append(f"      {_fmt_stage(st)}")
    skipped = [k for k in range(len(levels)) if k not in set(run)]
    if skipped:
        rows.append(f"  levels skipped by levels.run: {skipped}")
    return "\n".join(rows)


def _report_stray_overrides(cfg, levels) -> None:
    from .config import skipped_level_overrides, unused_level_overrides

    n_levels = len(levels)
    stray = unused_level_overrides(cfg, n_levels)
    if stray:
        print(
            f"  warning: per-level override(s) for level(s) {stray} will not "
            f"run; this pyramid has levels 0..{n_levels - 1}. The depth "
            f"follows from the grid and the chunk core, not the config."
        )
    skipped = [
        k for k in skipped_level_overrides(cfg, cfg.levels.run_levels(levels))
        if k < n_levels
    ]
    if skipped:
        print(
            f"  warning: per-level override(s) for level(s) {skipped} will not "
            f"run, because levels.run leaves those levels out."
        )


def cmd_setup(args) -> int:
    """Stage one: ingest, check conventions, measure, and plan."""
    from .grid import pyramid

    cfg = _load(args.config, args.stop_at)
    print(
        f"config {args.config}\n"
        f"  root {cfg.root_path}, profile {cfg.profile_name!r} "
        f"({cfg.profile.channels} channels, {cfg.profile.features}), "
        f"{cfg.n_subjects} subject(s), runner {cfg.runner}, backend {cfg.backend}"
    )
    for w in cfg.validate():
        print(f"  warning: {w}")
    device = _configure(cfg)
    engine_name = args.engine or cfg.engine_for(device)

    print("\ningest")
    if args.no_ingest:
        print("  skipped (--no-ingest)")
    else:
        done, kept = _ingest_subjects(cfg, reingest=args.reingest, jobs=args.jobs)
        if not done:
            print(f"  nothing to do; {len(kept)} subject(s) already ingested")
        elif kept:
            print(f"  {len(done)} ingested, {len(kept)} already present")

    print("\nconventions")
    if args.no_selftest:
        print("  skipped (--no-selftest)")
    else:
        from .selftest import run_selftest

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

        run_levels = cfg.levels.run_levels(levels)
        cal = calibrate(
            cfg, engine=engine_name, stages=cfg.levels.stages(run_levels[-1])
        )
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

        print(probe(cfg, pairs=args.pairs, engine=engine_name).format())

    print("\nresolved pyramid")
    print(_ladder(cfg, levels))
    _report_stray_overrides(cfg, levels)
    print("\nstages per level")
    print(_stage_table(cfg, levels))

    print()
    from .planner import plan

    print(plan(cfg, native).format(n_gpus=args.gpus))
    print(f"\nsetup complete. Run it with:  chunkreg run {args.config}")
    return 0


def cmd_run(args) -> int:
    """Stage two: build the template."""
    from .grid import pyramid
    from .pipelines import build_template

    cfg = _load(args.config, args.stop_at)
    _check_subjects(cfg)
    native = _native_grid(cfg)
    levels = pyramid(native, cfg.profile)

    print(
        f"{cfg.n_subjects} subject(s), {len(levels)} levels, profile "
        f"{cfg.profile_name!r}, runner {cfg.runner}"
    )
    device = _configure(cfg)
    engine_name = args.engine or cfg.engine_for(device)
    print(_ladder(cfg, levels))
    _report_stray_overrides(cfg, levels)
    run_levels = cfg.levels.run_levels(levels)
    if len(run_levels) < len(levels):
        print(f"  running levels {list(run_levels)} of 0..{len(levels) - 1}")
    tuned = [k for k in cfg.levels.tuned_levels() if k in set(run_levels)]
    if tuned:
        print(f"  per-level tuning active on level(s) {tuned}")
    print()

    runner, engine = _make_runner(cfg, device, engine_name, args.config, args.workers)
    try:
        result = build_template(
            cfg,
            runner=runner,
            engine=engine,
            from_level=args.from_level,
            progress=print,
            config_path=args.config,
        )
    finally:
        if hasattr(runner, "close"):
            runner.close()
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
    _configure(cfg)
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
    """Register one moving volume onto one fixed one."""
    from .config import save_config
    from .grid import pyramid
    from .pipelines import build_template, pair_config

    cfg = _load(args.config, args.stop_at)
    _check_subjects(cfg)
    levels = pyramid(_native_grid(cfg), cfg.profile)

    # Derived and written out before anything is dispatched. A worker process
    # or a batch node rebuilds its task by loading a config file, so the pair
    # has to exist as a file for the run to reach past this process.
    paired = pair_config(cfg, args.fixed, args.moving, root=args.root)
    path = save_config(paired, Path(paired.root) / "config.json")

    print(
        f"registering {args.moving!r} onto {args.fixed!r}, {len(levels)} levels, "
        f"profile {cfg.profile_name!r}, runner {cfg.runner}\n"
        f"  writing to {paired.root} under {path}"
    )
    device = _configure(paired)
    engine_name = args.engine or paired.engine_for(device)
    print(_ladder(paired, levels))
    print()

    runner, engine = _make_runner(paired, device, engine_name, path, args.workers)
    try:
        result = build_template(
            paired,
            runner=runner,
            engine=engine,
            from_level=args.from_level,
            progress=print,
            config_path=str(path),
        )
    finally:
        if hasattr(runner, "close"):
            runner.close()
    print()
    print(result.summary())
    print(f"\nfield: {paired.final_field_path(args.moving)}")
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
    from . import xp

    device = xp.configure(cfg.device)

    if args.pass_name == "register":
        engine = get_engine(args.engine or cfg.engine_for(device))
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
    """Warp a volume through a field, on the field's own grid, block by block."""
    from . import xp
    from .apply import apply_field, matching_level
    from .store import Field, Volume

    device = xp.configure(args.device)
    print(f"  compute: {'NumPy on the CPU' if device == 'cpu' else 'PyTorch on ' + device}")

    vol = Volume.open(args.volume)
    field = Field.open(args.field)
    try:
        level = args.level if args.level is not None else matching_level(vol, field.level_grid)
        apply_field(
            vol, field, args.out, level=level, profile=args.profile,
            target=args.on,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    where = f", on the lattice of {args.on}" if args.on else ""
    print(f"wrote {args.out} from level {level} of {args.volume}{where}")
    return 0


def cmd_export(args) -> int:
    from .apply import export_store

    from .store import Volume

    try:
        vol = Volume.open(args.store)
        level = vol.n_levels - 1 if args.level is None else vol._check_level(args.level)
        grid = vol.grid(level)
        print(
            f"  level {level} of 0..{vol.n_levels - 1}: "
            f"{'x'.join(str(n) for n in grid.shape)} @ {grid.spacing_mm:g} mm"
        )
        export_store(vol, args.out, level=level, target=args.on)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    where = f" on the lattice of {args.on}" if args.on else ""
    print(f"wrote {args.out}{where}")
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
    from .ingest import IngestError

    try:
        return _COMMANDS[args.cmd](args)
    except (SystemExit, KeyboardInterrupt):
        raise
    except IngestError as exc:
        # A cohort that cannot be ingested is a configuration problem with a
        # message written for a person, not a traceback. The library raises so
        # a script can catch it; here it is the end of the run.
        raise SystemExit(str(exc)) from None
    except Exception as exc:  # noqa: BLE001 - a CLI should not show a traceback
        print(f"chunkreg {args.cmd}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

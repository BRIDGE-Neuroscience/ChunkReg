"""The level loop: register, blend, update, test, promote.

One loop serves both entry points. Pairwise registration is the groupwise loop
with one subject and the template held fixed, so there is a single manifest
format, a single task format and a single place where convergence is decided.

The loop exits a level on a measured condition rather than a configured count::

    stop when   d99_residual < residual_frac * D_max(next level)
          and   eps * ||u_bar||_99.9 < spacing(level)

The first half is the coupling between levels: a coarse level keeps working
until what it leaves behind is comfortably inside the halo the next level has
to absorb it with. The second half is template convergence: once the recentring
step is sub-voxel, further passes cannot move the template by anything it can
represent. A level that inherits an already-converged shape exits after one
pass; the first level, which starts from a biased voxelwise mean, uses its cap.

Register and blend are one dispatch, not two. A core's blend reads the chunks
whose padded boxes reach into it and nothing else, so it can start as soon as
those are registered. A runner that says it can (``runs_fused``) is handed both
passes and the dependency between them; the rest keep the barrier, which for a
serial runner costs nothing anyway.

A run resumes rather than restarts. Every finished pass leaves a record and
every seeded level a marker, so starting the same run again replays the
records through the stopping rule, skips the work they cover and carries on
from the first pass that has none. That is also how an early stop is
continued: a later run that allows finer levels finds the last level done and
promotes from it.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .. import xp as _xp
from ..config import RunConfig
from ..engines import get_engine
from ..features import get_extractor
from ..grid import GridSpec, pyramid
from ..passes import blend as _blend
from ..passes import promote as _promote
from .. import qc as _qc
from ..passes import register as _register
from ..passes import update as _update
from ..passes.manifest import Manifest, build_manifest
from ..runners import LocalRunner
from ..runners.base import RunReport
from ..store import Field, Volume
from .. import stats as _stats

__all__ = [
    "PassReport",
    "LevelReport",
    "RunResult",
    "build_template",
    "register_pair",
    "pair_config",
]


@dataclass
class PassReport:
    level: int
    iteration: int
    d99_mm: float
    ubar_mm: float
    step_mm: float
    fold_frac: float
    retries: int
    skipped: int
    seconds: float
    stopped: bool = False
    reason: str = ""


@dataclass
class LevelReport:
    level: int
    grid: GridSpec
    d_max_mm: float | None
    passes: list[PassReport] = field(default_factory=list)

    @property
    def n_passes(self) -> int:
        return len(self.passes)

    @property
    def final(self) -> PassReport:
        return self.passes[-1]


@dataclass
class RunResult:
    template: Volume
    fields: dict[str, Field]
    levels: list[LevelReport] = field(default_factory=list)
    fixed_template: str | None = None
    """The subject held fixed, when there was one. A pairwise run reports no
    template motion, so the column that carries it is left out rather than
    printed as a column of zeros."""

    @property
    def n_passes(self) -> int:
        return sum(lv.n_passes for lv in self.levels)

    def summary(self) -> str:
        step = self.fixed_template is None
        rows = [
            f"{'lvl':>3} {'spacing':>8} {'pass':>5} {'d99 mm':>9} "
            + (f"{'step mm':>9} " if step else "")
            + f"{'folds':>8} {'exit':>28}"
        ]
        for lv in self.levels:
            for p in lv.passes:
                rows.append(
                    f"{lv.level:>3} {lv.grid.spacing_mm:>8.4g} {p.iteration:>5} "
                    f"{p.d99_mm:>9.4f} "
                    + (f"{p.step_mm:>9.4f} " if step else "")
                    + f"{p.fold_frac:>8.2%} "
                    f"{(p.reason if p.stopped else ''):>28}"
                )
        return "\n".join(rows)


def _stop_threshold(cfg: RunConfig, next_grid: GridSpec | None) -> float:
    """How small the residual must get before the next level run can absorb it.

    The next level is the next one that actually runs, which is not always the
    next one in the pyramid: skipping a level means the level after the gap
    has to absorb the residual with its own, smaller clamp.
    """
    if next_grid is None:
        return 0.0  # the last level run has no successor to hand anything to
    return cfg.levels.residual_frac * cfg.profile.d_max_mm(next_grid.spacing_mm)


def build_template(
    cfg: RunConfig,
    runner=None,
    engine=None,
    from_level: int | None = None,
    progress: Callable[[str], None] | None = None,
    config_path: str | None = None,
) -> RunResult:
    """Estimate an unbiased template and one field per subject.

    With ``from_level`` left as ``None`` the run resumes from whatever an
    earlier start of it finished. Naming a level discards the progress at that
    level and every finer one, and redoes them.

    ``config_path`` is only needed for a scheduler runner, which addresses each
    task by config file rather than shipping a callable to the batch node.
    """
    if runner is not None and hasattr(runner, "bind") and config_path is None:
        raise ValueError(
            "a scheduler runner addresses tasks by config file, so "
            "build_template needs config_path=... to pass to 'chunkreg "
            "run-task' on the batch node"
        )
    device = _xp.configure(cfg.device)
    runner = runner or LocalRunner()
    if hasattr(runner, "bind"):
        # Tasks run in other processes, which build their own engine and
        # feature network; the driver only coordinates and promotes.
        extractor = None
    else:
        engine = engine or get_engine(cfg.engine_for(device))
        extractor = get_extractor(cfg.profile.features)
        extractor.setup()
    say = progress or (lambda _msg: None)

    volumes = {s: Volume.open(cfg.subject_path(s), cfg.backend) for s in cfg.subject_ids}
    if not cfg.registered_ids:
        raise ValueError(
            "every subject is the fixed template, so there is nothing to register"
        )
    native = next(iter(volumes.values())).native_grid
    for sid, v in volumes.items():
        if v.native_grid != native:
            raise ValueError(
                f"subject {sid!r} is on grid {v.native_grid.shape} at "
                f"{v.native_grid.spacing_mm} mm, but the cohort is on "
                f"{native.shape} at {native.spacing_mm} mm. Co-registration "
                f"needs every subject resampled onto one grid at ingest; "
                f"'chunkreg setup --reingest' does that."
            )
    levels = pyramid(native, cfg.profile)
    run_levels = cfg.levels.run_levels(levels)
    if from_level is not None:
        if from_level not in run_levels:
            raise ValueError(
                f"from_level {from_level} is not one of the levels this config "
                f"runs: {list(run_levels)}"
            )
        if from_level > 0 and not _is_seeded(cfg, _previous(run_levels, from_level)):
            raise ValueError(
                f"cannot redo from level {from_level}: level "
                f"{_previous(run_levels, from_level)} before it has not run"
            )
        say(f"discarding progress at level {from_level} and finer")
        _forget(cfg, from_level)

    reports: list[LevelReport] = []
    for position, level in enumerate(run_levels):
        grid = levels[level]
        previous = run_levels[position - 1] if position > 0 else None
        following = (
            run_levels[position + 1] if position + 1 < len(run_levels) else None
        )

        if not _is_seeded(cfg, level):
            if previous is None:
                say(
                    "level 0: initial template from "
                    + (
                        f"subject {cfg.template_subject!r}"
                        if cfg.template_subject
                        else "the voxelwise mean"
                    )
                )
                _promote.seed_level_zero(cfg, grid, volumes)
            else:
                say(f"promote level {previous} -> {level}")
                _promote.promote_level(
                    cfg, previous, levels[previous], grid, to_level=level,
                    runner=runner,
                )
            _mark_seeded(cfg, level, previous)

        report = LevelReport(
            level=level,
            grid=grid,
            d_max_mm=None if level == 0 else cfg.profile.d_max_mm(grid.spacing_mm),
        )
        threshold = _stop_threshold(
            cfg, None if following is None else levels[following]
        )
        cap = cfg.levels.cap(level)
        # Once a finer level has been seeded from this one, another pass here
        # would leave that seed describing a solution that no longer exists.
        handed_on = following is not None and _is_seeded(cfg, following)

        for iteration in range(cap):
            pr = _load_pass(cfg, level, iteration)
            replayed = pr is not None
            if not replayed:
                if handed_on:
                    break
                pr, manifest = _run_pass(
                    cfg, runner, engine, extractor, level, iteration, grid, config_path
                )
            stopped, reason = _should_stop(cfg, pr, threshold, grid, iteration, cap)
            pr.stopped, pr.reason = stopped, reason
            report.passes.append(pr)
            say(
                f"level {level} pass {iteration}: "
                + ("done earlier, " if replayed else "")
                + f"d99 {pr.d99_mm:.4f} mm, "
                + ("" if cfg.template_subject else f"step {pr.step_mm:.4f} mm, ")
                + f"folds {pr.fold_frac:.2%}"
                + (f" -> {reason}" if stopped else "")
            )
            if not replayed:
                _record_pass(cfg, pr)
                # A failed sheet is a warning, never a dead run: the run's
                # outputs are the template and fields, and the sheet is
                # only there so they can be judged without pulling them.
                try:
                    png = _qc.render_pass(cfg, level, iteration, grid, manifest.d_max_mm)
                    say(f"  qc sheet {png}")
                except Exception as exc:  # noqa: BLE001 - reported, never fatal
                    say(f"  qc sheet not written: {type(exc).__name__}: {exc}")
                _cleanup(cfg, manifest)
            if stopped:
                break

        reports.append(report)

    last = run_levels[-1]
    if last < len(levels) - 1:
        say(
            f"stopping at level {last} ({levels[last].spacing_mm * 1000:g} um) "
            f"as the config asks; a later run with finer levels carries on from here"
        )
    # The last pass of the last level published a template motion that no
    # promote will ever carry, so the fields on disk still point at where the
    # template was before that move. Fold it in before handing them back.
    say("settling the final fields")
    fields_ = _promote.settle_fields(cfg, last, levels[last], runner=runner)
    return RunResult(
        template=Volume.open(cfg.template_path(last), cfg.backend),
        fields=fields_,
        levels=reports,
        fixed_template=cfg.template_subject,
    )


def _run_pass(
    cfg: RunConfig,
    runner,
    engine,
    extractor,
    level: int,
    iteration: int,
    grid: GridSpec,
    config_path: str | None,
) -> tuple[PassReport, Manifest]:
    """Register, blend and update once at one level."""
    t0 = time.perf_counter()
    manifest = build_manifest(cfg, level, iteration, grid)
    manifest.save(cfg.level_dir(level) / f"manifest_it{iteration}.json")
    # A scheduler runner cannot receive a closure, so it ships the task's
    # address instead and needs to know which pass it is on. The manifest has
    # to be on disk before this: a batch node rebuilds its work from the config
    # and that file.
    if hasattr(runner, "bind"):
        runner.bind(config_path, level, iteration, cfg=cfg)

    # Created before anything is dispatched, so the common path does not rely
    # on the race-tolerance inside the task, and so a blend that starts while
    # register is still running finds them there.
    _blend.ensure_accumulators(cfg, manifest)
    _qc.ensure_residual_stores(cfg, manifest)
    register_ids = range(manifest.n_tasks)
    blend_ids = range(len(manifest.chunks))

    if getattr(runner, "runs_fused", False):
        # A core's blend waits only on the chunks that reach into it, not on
        # the whole register pass, so the two run as one dependency-ordered
        # dispatch. The barrier they replace is where most of a node's GPUs
        # sit idle behind the last few register tasks of every pass.
        reg, bl = runner.run_chain(
            "register",
            register_ids,
            "blend",
            blend_ids,
            _blend.blend_depends(cfg, manifest),
        )
    else:
        reg = runner.run(
            "register",
            lambda tid: _register.run_register_task(
                cfg, manifest, tid, engine=engine, extractor=extractor
            ),
            register_ids,
        )
        reg.raise_for_failures()
        bl = runner.run(
            "blend",
            lambda cid: _blend.run_blend_task(cfg, manifest, cid),
            blend_ids,
        )
    reg.raise_for_failures()
    bl.raise_for_failures()

    # A template held fixed is not estimated, so there is no intensity average
    # to write and no shape bias to recentre away. Skipping the pass is what
    # keeps the fixed volume actually fixed: the update would otherwise
    # overwrite it with the warped moving subject.
    if cfg.template_subject is None:
        _update.ensure_recentre(cfg, manifest)
        up = runner.run(
            "update",
            lambda cid: _update.run_update_task(cfg, manifest, cid),
            range(len(manifest.chunks)),
        )
        up.raise_for_failures()
    else:
        up = RunReport(pass_name="update")

    pr = _collect(
        level,
        iteration,
        reg,
        bl,
        up,
        time.perf_counter() - t0,
        cfg.levels.stop_percentile,
        cfg.levels.shape_update_step,
    )
    return pr, manifest


def _previous(run_levels: Sequence[int], level: int) -> int:
    return run_levels[list(run_levels).index(level) - 1]


# --------------------------------------------------------------------------- #
# Progress on disk
# --------------------------------------------------------------------------- #
def _write_atomic(path: Path, text: str) -> None:
    """Write a small file so a reader sees all of it or none of it.

    A job killed at its time limit can die mid-write, and a torn record read
    back as "this pass finished" would skip work that never completed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _is_seeded(cfg: RunConfig, level: int) -> bool:
    """Has this level been given its starting template and fields?

    A pass record counts too, so a run begun before seed markers existed is
    not seeded a second time over its own progress.
    """
    return (
        cfg.seed_marker_path(level).exists()
        or cfg.pass_record_path(level, 0).exists()
    )


def _mark_seeded(cfg: RunConfig, level: int, from_level: int | None) -> None:
    _write_atomic(
        cfg.seed_marker_path(level),
        json.dumps({"level": level, "from_level": from_level}),
    )


def _load_pass(cfg: RunConfig, level: int, iteration: int) -> PassReport | None:
    """The record a finished pass left, or ``None`` if it has none."""
    path = cfg.pass_record_path(level, iteration)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return PassReport(**{f.name: raw[f.name] for f in fields(PassReport) if f.name in raw})
    except (OSError, ValueError, TypeError):
        return None  # unreadable: treat the pass as not done and redo it


def _forget(cfg: RunConfig, from_level: int) -> None:
    """Discard the progress at ``from_level`` and every finer level.

    Records and seed markers go so the loop redoes those levels, and so do the
    manifests and scratch objects of the passes they describe: a task array
    left from the discarded attempt would otherwise be taken as this
    attempt's finished work.
    """
    from .. import backend as _backend

    b = _backend.get_backend(cfg.backend)
    root = cfg.root_path / "levels"
    if not root.exists():
        return
    for level_dir in root.glob("L*"):
        try:
            level = int(level_dir.name[1:])
        except ValueError:
            continue
        if level < from_level:
            continue
        for path in level_dir.glob("manifest_it*.json"):
            try:
                manifest = Manifest.load(path)
            except (OSError, ValueError, KeyError):
                manifest = None
            if manifest is not None:
                for t in manifest.tasks:
                    b.remove(cfg.task_path(level, manifest.iteration, t.task_id))
                for which in ("isum", "wsum"):
                    b.remove(cfg.accumulator_path(level, manifest.iteration, which))
            path.unlink()
        for path in level_dir.glob("pass_it*.json"):
            path.unlink()
        # Pairwise runs never rewrite the recentring store, so a stale one
        # would be folded into their seeds.
        b.remove(cfg.recentre_path(level))
        marker = cfg.seed_marker_path(level)
        if marker.exists():
            marker.unlink()


def pair_config(
    cfg: RunConfig, fixed: str, moving: str, root: str | Path | None = None
) -> RunConfig:
    """The configuration a pairwise run of two of these subjects runs under.

    Two things change, and both have to, for reasons that are easy to miss.

    The subject list narrows to the two volumes and ``template_subject`` names
    the fixed one. That much is the whole of pairwise registration.

    The root moves to a directory of its own, and the subject paths are made
    absolute so they still point at the cohort's stores from there. A pairwise
    run writes the same files a groupwise run does -- ``levels/L*/template``,
    ``levels/L*/pass_it*.json``, ``fields/<id>`` -- and a run resumes from
    whatever it finds. Sharing a root with the cohort run, or with another
    pair, would make each one replay the other's finished passes and seed from
    the other's template, which is not a conflict any of them can detect.
    """
    from dataclasses import replace

    known = set(cfg.subject_ids)
    for role, sid in (("fixed", fixed), ("moving", moving)):
        if sid not in known:
            raise ValueError(
                f"{role} subject {sid!r} is not in this run's subject list "
                f"{sorted(known)}; both volumes have to be ingested and "
                f"declared before they can be registered"
            )
    if fixed == moving:
        raise ValueError(
            f"fixed and moving are both {fixed!r}; registering a volume to "
            f"itself has no meaning"
        )

    # Absolute, because this config's whole purpose is to be loaded somewhere
    # else: a worker process or a batch node need not start in the directory
    # the driver did, and a relative root would send them somewhere else.
    root = os.path.abspath(root if root is not None else cfg.pair_root(fixed, moving))
    subjects = []
    for spec in cfg.subjects:
        if spec.id not in (fixed, moving):
            continue
        source = spec.source
        if source is not None and not Path(source).is_absolute():
            source = os.path.abspath(cfg.root_path / source)
        subjects.append(
            replace(
                spec,
                path=os.path.abspath(cfg.subject_path(spec.id)),
                source=source,
            )
        )
    return replace(
        cfg,
        root=str(root),
        subjects=tuple(subjects),
        template_subject=fixed,
    )


def register_pair(
    cfg: RunConfig,
    fixed: str,
    moving: str,
    runner=None,
    engine=None,
    root: str | Path | None = None,
    config_path: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
    from_level: int | None = None,
) -> RunResult:
    """Pairwise registration: the same loop with the template held fixed.

    The fixed volume *is* the template at every level, so no template is
    estimated, the unbiasing update pass does not run, and the loop reduces to
    seeded refinement of one subject down the pyramid.

    The derived configuration is written to disk before the run starts, and
    that file is what the run is dispatched by. A multi-GPU worker and a batch
    node each rebuild their task from a config path rather than receiving a
    closure, so a pair that existed only in the driver's memory could not be
    spread over anything: it had to run in one process. Writing it out is what
    makes ``chunkreg pair`` reach the same GPUs ``chunkreg run`` does.
    """
    from ..config import save_config

    paired = pair_config(cfg, fixed, moving, root=root)
    path = save_config(paired, config_path or Path(paired.root) / "config.json")
    return build_template(
        paired,
        runner=runner,
        engine=engine,
        from_level=from_level,
        progress=progress,
        config_path=str(path),
    )


def _collect(level, iteration, reg, bl, up, seconds, q: float, eps: float) -> PassReport:
    records = [r for res in reg.results() for r in (res.get("records") or [])]
    stats = [s for res in bl.results() for s in (res.get("stats") or [])]
    # Percentiles are read off summed histograms, so they are exact over the
    # whole level. Combining per-task percentiles, or taking a maximum across
    # tasks, would turn the statistic into an outlier detector that a level
    # pinned at its clamp by a handful of voxels can never clear.
    residual = _stats.merge(r.get("residual_hist") for r in records)
    steps = _stats.merge(r.get("step_hist") for r in up.results())
    folds = max((s["folds"] for s in stats), default=0.0)
    step = _stats.percentile(steps, q)
    return PassReport(
        level=level,
        iteration=iteration,
        d99_mm=float(_stats.percentile(residual, q)),
        ubar_mm=float(step / max(eps, 1e-9)),
        step_mm=float(step),
        fold_frac=float(folds),
        retries=sum(int(r.get("retries", 0)) for r in records),
        skipped=sum(1 for r in records if r.get("skipped")),
        seconds=float(seconds),
    )


def _should_stop(
    cfg: RunConfig, pr: PassReport, threshold: float, grid: GridSpec, iteration, cap
) -> tuple[bool, str]:
    """Has this level finished?

    Two rules, because the two pipelines converge on different quantities.

    With an estimated template, the binding constraint is the template itself:
    the fields chase a target that is still moving, so the level is done when
    the recentring step goes sub-voxel *and* the residual fits the next halo.

    With a template held fixed the target never moves, the update pass does not
    run, and ``step_mm`` is identically zero -- so that test passes vacuously
    and cannot be the rule. What converges instead is the residual: the part of
    the correspondence each pass could not already explain. The level is done
    once that is smaller than the level can represent, or small enough for the
    next level's halo to absorb. Without this a pairwise run stopped after one
    pass at its finest level, every time.
    """
    fits_next_halo = threshold > 0 and pr.d99_mm < threshold

    if cfg.template_subject is not None:
        if pr.d99_mm < cfg.levels.residual_vox * grid.spacing_mm:
            return True, "residual is sub-voxel"
        if fits_next_halo:
            return True, "residual fits the next halo"
    else:
        converged_template = pr.step_mm < cfg.levels.ubar_vox * grid.spacing_mm
        if threshold <= 0:
            if converged_template:
                return True, "template converged"
        elif fits_next_halo and converged_template:
            return True, "residual fits the next halo"
    if iteration + 1 >= cap:
        return True, "cap reached"
    return False, ""


def _record_pass(cfg: RunConfig, pr: PassReport) -> None:
    """Record that a pass finished, before its transient objects are deleted.

    Retention removes the task arrays as soon as the blend has consumed them,
    so their completion markers cannot answer "did this pass run". This file
    can, and it carries the pass's numbers for anyone reading progress later.
    """
    _write_atomic(
        cfg.pass_record_path(pr.level, pr.iteration), json.dumps(asdict(pr), indent=1)
    )


def _cleanup(cfg: RunConfig, manifest: Manifest) -> None:
    """Delete the transient objects a finished pass no longer needs."""
    from .. import backend as _backend

    b = _backend.get_backend(cfg.backend)
    if cfg.retention.delete_task_arrays:
        for t in manifest.tasks:
            b.remove(cfg.task_path(manifest.level, manifest.iteration, t.task_id))
    if cfg.retention.delete_accumulators:
        for which in ("isum", "wsum"):
            b.remove(cfg.accumulator_path(manifest.level, manifest.iteration, which))

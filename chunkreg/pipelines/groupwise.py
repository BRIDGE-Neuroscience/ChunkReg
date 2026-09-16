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
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from ..config import RunConfig
from ..engines import get_engine
from ..features import get_extractor
from ..grid import GridSpec, pyramid
from ..passes import blend as _blend
from ..passes import promote as _promote
from ..passes import register as _register
from ..passes import update as _update
from ..passes.manifest import Manifest, build_manifest
from ..runners import LocalRunner
from ..runners.base import RunReport
from ..store import Field, Volume
from .. import stats as _stats

__all__ = ["PassReport", "LevelReport", "RunResult", "build_template", "register_pair"]


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

    @property
    def n_passes(self) -> int:
        return sum(lv.n_passes for lv in self.levels)

    def summary(self) -> str:
        rows = [
            f"{'lvl':>3} {'spacing':>8} {'pass':>5} {'d99 mm':>9} "
            f"{'step mm':>9} {'folds':>8} {'exit':>28}"
        ]
        for lv in self.levels:
            for p in lv.passes:
                rows.append(
                    f"{lv.level:>3} {lv.grid.spacing_mm:>8.4g} {p.iteration:>5} "
                    f"{p.d99_mm:>9.4f} {p.step_mm:>9.4f} {p.fold_frac:>8.2%} "
                    f"{(p.reason if p.stopped else ''):>28}"
                )
        return "\n".join(rows)


def _stop_threshold(cfg: RunConfig, levels: Sequence[GridSpec], level: int) -> float:
    """How small the residual must get before the next level can absorb it."""
    if level + 1 >= len(levels):
        return 0.0  # the finest level has no successor to hand anything to
    return cfg.levels.residual_frac * cfg.profile.d_max_mm(levels[level + 1].spacing_mm)


def build_template(
    cfg: RunConfig,
    runner=None,
    engine=None,
    from_level: int = 0,
    progress: Callable[[str], None] | None = None,
    config_path: str | None = None,
) -> RunResult:
    """Estimate an unbiased template and one field per subject.

    ``config_path`` is only needed for a scheduler runner, which addresses each
    task by config file rather than shipping a callable to the batch node.
    """
    if runner is not None and hasattr(runner, "bind") and config_path is None:
        raise ValueError(
            "a scheduler runner addresses tasks by config file, so "
            "build_template needs config_path=... to pass to 'chunkreg "
            "run-task' on the batch node"
        )
    runner = runner or LocalRunner()
    engine = engine or get_engine("demons")
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
                f"needs every subject resampled onto one grid at ingest."
            )
    levels = pyramid(native, cfg.profile)

    if from_level == 0:
        say("level 0: initial template from the voxelwise mean")
        _promote.seed_level_zero(cfg, levels[0], volumes)

    reports: list[LevelReport] = []
    for level in range(from_level, len(levels)):
        grid = levels[level]
        report = LevelReport(
            level=level,
            grid=grid,
            d_max_mm=None if level == 0 else cfg.profile.d_max_mm(grid.spacing_mm),
        )
        threshold = _stop_threshold(cfg, levels, level)
        cap = cfg.levels.cap(level)

        for iteration in range(cap):
            t0 = time.perf_counter()
            manifest = build_manifest(cfg, level, iteration, grid)
            manifest.save(cfg.level_dir(level) / f"manifest_it{iteration}.json")
            # A scheduler runner cannot receive a closure, so it ships the
            # task's address instead and needs to know which pass it is on.
            # The manifest has to be on disk before this: a batch node rebuilds
            # its work from the config and that file.
            if hasattr(runner, "bind"):
                runner.bind(config_path, level, iteration, cfg=cfg)

            reg = runner.run(
                "register",
                lambda tid: _register.run_register_task(
                    cfg, manifest, tid, engine=engine, extractor=extractor
                ),
                range(manifest.n_tasks),
            )
            reg.raise_for_failures()

            # Created once here so the common path does not rely on the
            # race-tolerance inside the task.
            _blend.ensure_accumulators(cfg, manifest)
            bl = runner.run(
                "blend",
                lambda cid: _blend.run_blend_task(cfg, manifest, cid),
                range(len(manifest.chunks)),
            )
            bl.raise_for_failures()

            # A template held fixed is not estimated, so there is no intensity
            # average to write and no shape bias to recentre away. Skipping the
            # pass is what keeps the fixed volume actually fixed: the update
            # would otherwise overwrite it with the warped moving subject.
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
            stopped, reason = _should_stop(cfg, pr, threshold, grid, iteration, cap)
            pr.stopped, pr.reason = stopped, reason
            report.passes.append(pr)
            say(
                f"level {level} pass {iteration}: d99 {pr.d99_mm:.4f} mm, "
                f"step {pr.step_mm:.4f} mm, folds {pr.fold_frac:.2%}"
                + (f" -> {reason}" if stopped else "")
            )
            _record_pass(cfg, pr)
            _cleanup(cfg, manifest)
            if stopped:
                break

        reports.append(report)
        if level + 1 < len(levels):
            say(f"promote level {level} -> {level + 1}")
            _promote.promote_level(cfg, level, grid, levels[level + 1])

    last = len(levels) - 1
    # The last pass of the last level published a template motion that no
    # promote will ever carry, so the fields on disk still point at where the
    # template was before that move. Fold it in before handing them back.
    say("settling the final fields")
    fields = _promote.settle_fields(cfg, last, levels[last])
    return RunResult(
        template=Volume.open(cfg.template_path(last), cfg.backend),
        fields=fields,
        levels=reports,
    )


def register_pair(
    cfg: RunConfig,
    fixed: str,
    moving: str,
    runner=None,
    engine=None,
    config_path: str | None = None,
) -> RunResult:
    """Pairwise registration: the same loop with the template held fixed.

    The fixed volume *is* the template at every level, so no template is
    estimated, the unbiasing update pass does not run, and the loop reduces to
    seeded refinement of one subject down the pyramid.
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

    paired = replace(
        cfg,
        subjects=tuple(s for s in cfg.subjects if s.id in (fixed, moving)),
        template_subject=fixed,
    )
    return build_template(
        paired, runner=runner, engine=engine, config_path=config_path
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
    converged_template = pr.step_mm < cfg.levels.ubar_vox * grid.spacing_mm
    if threshold <= 0:
        if converged_template:
            return True, "template converged"
    elif pr.d99_mm < threshold and converged_template:
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
    import json
    from dataclasses import asdict

    path = cfg.pass_record_path(pr.level, pr.iteration)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(pr), indent=1), encoding="utf-8")


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

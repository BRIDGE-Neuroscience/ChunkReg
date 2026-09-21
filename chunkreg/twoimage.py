"""Registering one moving volume onto one fixed volume, from two file paths.

The rest of the package is built around a cohort: a config file lists the
subjects, ``chunkreg setup`` ingests them, ``chunkreg run`` builds a template.
Pairwise registration was reachable from there, but only from there -- a
caller with two files and a question had to write a configuration, run two
commands and then work out which of the outputs was the answer.

This is the same pipeline with the cohort machinery filled in from the two
paths. It ingests both volumes, holds the fixed one as the template, runs the
level loop, and hands back the field and a way to get the moving volume onto
the fixed one's lattice. Nothing here is a second implementation: every step
is the call the command line makes.

Larger than memory throughout, the same way the cohort path is. Neither volume
is ever held whole: ingest streams each scan onto the run grid a block at a
time, registration works chunk by chunk with a fixed memory footprint per
task, and the delivery streams the warp and the resample together.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import ConfigError, GridRequest, RunConfig, save_config
from .store import Field, Volume

__all__ = ["register", "PairResult", "FIXED", "MOVING"]

FIXED = "fixed"
MOVING = "moving"
"""The two subject ids a two-image run uses.

Fixed names rather than the file stems: the ids appear in every path the run
writes, and a stem can be missing, repeated between the two volumes, or not a
usable identifier at all.
"""


@dataclass
class PairResult:
    """What a two-image registration produced, and what can be done with it."""

    config: RunConfig
    config_path: Path
    field: Field
    """The displacement, on the run grid, in millimetres. It maps fixed
    coordinates to moving ones, which is the direction that resamples the
    moving volume onto the fixed one."""
    fixed: Volume
    moving: Volume
    run: Any
    """The :class:`~chunkreg.pipelines.RunResult` the level loop returned."""

    @property
    def root(self) -> Path:
        return Path(self.config.root)

    def summary(self) -> str:
        return self.run.summary()

    def warp_moving(
        self, out: str | Path, on_fixed_lattice: bool = True, profile=None
    ) -> Volume:
        """Write the moving volume resampled onto the fixed one.

        ``on_fixed_lattice`` delivers it on the sampling the fixed scan
        arrived on, which is where the answer is normally wanted. Turning it
        off leaves the result on the run grid, which is the same thing
        whenever ``grid="fixed"`` chose the run grid from that scan.

        The warp and the resample are one streaming pass, so nothing the size
        of the result is written twice.
        """
        from .apply import apply_field

        return apply_field(
            self.moving,
            self.field,
            out,
            profile=profile or self.config.profile,
            backend=self.config.backend,
            target=self.fixed if on_fixed_lattice else None,
        )

    def export_moving(self, out: str | Path, on_fixed_lattice: bool = True) -> Path:
        """Write the moving volume onto the fixed one, as NIfTI or TIFF.

        Goes through a store, because the warp is defined on the run grid and
        a file is not chunked: the store is the thing that can be written a
        block at a time and then read back a plane at a time.

        That store is kept, under ``<root>/<name>.zarr``, rather than written
        into ``scratch`` and deleted. It is the same result in the format that
        scales, the export is the lossy step, and a caller who wants it again
        at a different size or in a different format should not have to
        re-run the warp. It is also not scratch: retention deletes what is
        under there once a pass has consumed it.
        """
        from .apply import export_store

        out = Path(out)
        name = out.name.split(".")[0] or "moving_in_fixed"
        store = self.root / f"{name}.zarr"
        self.warp_moving(store, on_fixed_lattice=on_fixed_lattice)
        export_store(store, out, backend=self.config.backend)
        return out


def _grid_block(grid, fixed_id: str) -> dict:
    """The ``grid`` block of the config a two-image registration builds.

    ``"fixed"`` is the default and the reason this function exists: the answer
    is expected on the fixed volume's lattice, and a moving scan sampled more
    finely would otherwise pull the whole run to a resolution neither volume
    resolves. The cohort settings stay reachable for the cases that want them,
    and a mapping goes straight through as the block itself.
    """
    if isinstance(grid, dict):
        return dict(grid)
    if grid == "fixed":
        request = GridRequest(reference=fixed_id)
    elif isinstance(grid, str):
        if grid not in ("finest", "coarsest"):
            raise ConfigError(
                f"grid must be 'fixed', 'finest', 'coarsest', a spacing in "
                f"millimetres, or a grid block, got {grid!r}"
            )
        request = GridRequest(spacing_mm=grid)
    else:
        request = GridRequest(spacing_mm=float(grid))
    return {
        k: v
        for k, v in (
            ("spacing_mm", request.spacing_mm),
            ("shape", None if request.shape is None else list(request.shape)),
            ("align", request.align),
            ("reference", request.reference),
        )
        if v is not None
    }


def register(
    fixed: str | Path,
    moving: str | Path,
    root: str | Path,
    grid: str | float = "fixed",
    profile: str = "a16",
    spacing_mm: float | None = None,
    fixed_spacing_mm=None,
    moving_spacing_mm=None,
    stop_at: str | int | float | None = None,
    device: str = "auto",
    engine: str = "auto",
    gpus: int | str = "all",
    workers: int = 1,
    runner=None,
    reingest: bool = False,
    from_level: int | None = None,
    progress: Callable[[str], None] | None = None,
    **overrides: Any,
) -> PairResult:
    """Register ``moving`` onto ``fixed``, both larger than memory.

    ``fixed`` and ``moving`` are scan files or stores in any format ingest
    accepts: OME-Zarr, plain zarr, NIfTI, or a TIFF stack. ``root`` is where
    everything is written -- the two ingested stores, the levels, and the
    field.

    By default the fixed volume's own sampling is the run grid, so the result
    comes back on the lattice it is expected on. ``grid`` takes the cohort
    settings instead: ``"finest"``, ``"coarsest"``, a spacing in millimetres,
    or a whole ``grid`` block as a mapping.

    Safe to call again. Ingest skips a volume already converted and the level
    loop resumes from the last pass that finished, so a run stopped at a time
    limit is continued by calling this with the same arguments.

    ``overrides`` reach any other top-level configuration key, for the
    settings a two-image run does not need a named argument for::

        register(..., levels={"caps": [4, 3, 2]}, features={"mind": False})

    Returns a :class:`PairResult`; ``warp_moving`` and ``export_moving`` on it
    are how the moving volume is delivered on the fixed one.
    """
    from .ingest import check_subjects, ingest_subjects
    from .pipelines import build_template
    from .runners import runner_for
    from . import xp as _xp

    say = progress or (lambda _msg: None)
    root = Path(root)

    raw: dict[str, Any] = dict(overrides)
    raw.update(
        root=str(root),
        profile=profile,
        subjects=[
            {"id": FIXED, "source": os.path.abspath(fixed)},
            {"id": MOVING, "source": os.path.abspath(moving)},
        ],
        template_subject=FIXED,
        device=device,
        engine=engine,
        gpus=gpus,
    )
    for sid, value in ((FIXED, fixed_spacing_mm), (MOVING, moving_spacing_mm)):
        if value is not None:
            for entry in raw["subjects"]:
                if entry["id"] == sid:
                    entry["spacing_mm"] = value
    if spacing_mm is not None:
        raw["spacing_mm"] = spacing_mm

    raw["grid"] = _grid_block(grid, FIXED)
    if stop_at is not None:
        levels = dict(raw.get("levels") or {})
        levels["stop_at"] = stop_at
        raw["levels"] = levels

    # Built by loading, not by constructing: a RunConfig assembled field by
    # field skips the validation and the derivation that load_config does, and
    # this config is one a worker will later load anyway.
    from .config import load_config

    cfg = load_config(raw)
    config_path = save_config(cfg, root / "config.json")
    say(f"configuration written to {config_path}")

    done, kept = ingest_subjects(cfg, reingest=reingest, jobs=2, say=say)
    if done:
        say(f"  ingested {len(done)}, {len(kept)} already present")
    check_subjects(cfg, say=say)

    device_name = _xp.configure(cfg.device)
    engine_name = cfg.engine_for(device_name)
    built = None
    if runner is None:
        runner, built = runner_for(
            cfg, device_name, engine_name, config_path, workers=workers, say=say
        )
        close = True
    else:
        close = False
    try:
        result = build_template(
            cfg,
            runner=runner,
            engine=built,
            from_level=from_level,
            progress=say,
            config_path=str(config_path),
        )
    finally:
        if close and hasattr(runner, "close"):
            runner.close()

    return PairResult(
        config=cfg,
        config_path=config_path,
        field=result.fields[MOVING],
        fixed=Volume.open(cfg.subject_path(FIXED), cfg.backend),
        moving=Volume.open(cfg.subject_path(MOVING), cfg.backend),
        run=result,
    )

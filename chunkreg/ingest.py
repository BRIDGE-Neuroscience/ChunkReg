"""Putting the scans named by a config onto disk as stores the pipeline can run.

Ingest was reachable only through ``chunkreg setup``, which left a script able
to build a template but not to prepare the volumes one is built from. Nothing
here is new: it is the work ``setup`` has always done, moved to where the
Python interface can call it too. :func:`chunkreg.register` is the caller that
needed it.

Two things are decided here and nowhere else. The run grid follows from the
whole cohort at once (see :mod:`chunkreg.cohort`), because every pass reads
subject level *k* and template level *k* at the same voxel indices, so one
scan cannot be placed without knowing the rest. And whether a scan is copied
or read where it lies follows from whether the ingest would change a voxel: a
zarr already exactly on the run grid is opened read-only and only its coarser
levels are written.

Reporting goes through a ``say`` callable rather than ``print`` so that the
command line can narrate an ingest while a script stays quiet. Failures are
:class:`IngestError`, which the command line turns back into a clean exit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .config import ConfigError, RunConfig

__all__ = [
    "IngestError",
    "ingest_subjects",
    "check_subjects",
    "native_grid",
    "same_grid",
    "fmt_grid",
    "fmt_voxel",
    "scan_voxel",
]


class IngestError(ConfigError):
    """A cohort that cannot be ingested or run as configured."""


def _resolve(cfg, rel) -> Path:
    """A config-relative path, left alone if it is already absolute."""
    q = Path(rel)
    return q if q.is_absolute() else cfg.root_path / q


def native_grid(cfg):
    from .store import Volume

    first = cfg.subjects[0]
    if Volume.exists(cfg.subject_path(first.id), cfg.backend):
        return Volume.open(cfg.subject_path(first.id), cfg.backend).native_grid
    if cfg.spacing_mm is None:
        raise IngestError(
            "no ingested subjects found and no spacing_mm in the config, so "
            "the native grid is unknown. Run 'chunkreg setup' first."
        )
    raise IngestError(
        f"subject {first.id!r} is not ingested at {cfg.subject_path(first.id)}. "
        f"Run 'chunkreg setup' first."
    )


def fmt_voxel(v) -> str:
    v = tuple(float(x) for x in v)
    if max(v) <= min(v) * (1 + 1e-6):
        return f"{v[0]:g} mm"
    return " x ".join(f"{x:g}" for x in v) + " mm (z, y, x)"


def scan_voxel(
    cfg, spec, source, say: Callable[[str], None] | None = None
) -> tuple[float, float, float]:
    """A scan's voxel size, from the subject entry, the file or the config.

    A subject's own ``spacing_mm`` is taken over the file, since the point of
    setting it is a file that is wrong or silent. The top-level value is a
    cohort-wide default, so it has to agree with any file that states a size.
    """
    from .io_formats import spacing_agrees

    say = say or (lambda _msg: None)
    stated = source.voxel_mm
    if spec.spacing_mm is not None:
        if stated is not None and not spacing_agrees(spec.spacing_mm, stated):
            say(
                f"  note: {spec.id}: using the subject's spacing_mm "
                f"{fmt_voxel(spec.spacing_mm)} over the file's {fmt_voxel(stated)}"
            )
        return spec.spacing_mm
    if cfg.spacing_mm is not None:
        if stated is not None and not spacing_agrees(cfg.spacing_mm, stated):
            raise IngestError(
                f"subject {spec.id!r}: the config says {cfg.spacing_mm:g} mm but "
                f"{source.description} says {fmt_voxel(stated)}. Fix one, or set "
                f"spacing_mm on the subject to overrule the file; the pipeline "
                f"will not guess which is right."
            )
        return (cfg.spacing_mm,) * 3
    if stated is not None:
        return stated
    if not isinstance(cfg.grid.spacing_mm, str):
        # Configs written before the grid block chose the run resolution used
        # grid.spacing_mm to mean the sources' voxel size.
        return (cfg.grid.spacing_mm,) * 3
    raise IngestError(
        f"subject {spec.id!r}: {source.description} does not state its voxel "
        f"size; set 'spacing_mm' on the subject or at the top level of the config."
    )


def same_grid(a, b) -> bool:
    import math

    return (
        a.shape == b.shape
        and math.isclose(a.spacing_mm, b.spacing_mm, rel_tol=1e-9)
        and all(
            math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-9 * a.spacing_mm)
            for x, y in zip(a.origin_mm, b.origin_mm)
        )
    )


def fmt_grid(g) -> str:
    return f"{'x'.join(str(n) for n in g.shape)} @ {g.spacing_mm:g} mm"


def ingest_subjects(
    cfg: RunConfig,
    reingest: bool = False,
    jobs: int = 4,
    say: Callable[[str], None] | None = None,
) -> tuple[list[str], list[str]]:
    """Ingest every subject that names a source and has no store yet.

    Scans may differ in shape and voxel size. The run grid is resolved from the
    whole cohort (see :mod:`chunkreg.cohort`) and each scan is resampled onto
    it, so every store ends up on one grid.
    """
    say = say or (lambda _msg: None)

    from . import backend as _backend
    from .cohort import Placement, ResampledSource, placement_notes, resolve_run_grid
    from .io_formats import is_chunkreg_store, is_zarr, open_source
    from .store import Volume, ingest_source, link_source

    todo, kept = [], []
    for spec in cfg.subjects:
        dst = cfg.subject_path(spec.id)
        present = Volume.exists(dst, cfg.backend)
        if present and not (reingest and spec.source is not None):
            kept.append(spec.id)
            continue
        if spec.source is None:
            if is_zarr(dst):
                raise IngestError(
                    f"subject {spec.id!r}: {dst} is a zarr but not a chunkreg "
                    f"store. Name it as the subject's 'source' and let 'path' "
                    f"default; setup then reads it in place, or copies it if it "
                    f"is not on the run grid."
                )
            raise IngestError(
                f"subject {spec.id!r} has no store at {dst} and no 'source' to "
                f"ingest from. Add a 'source' to the subject entry."
            )
        todo.append(spec)
    if not todo:
        return [], kept

    # The run grid depends on every scan, so the ones already ingested count
    # too. Their stores record the scan they came from, which saves opening
    # sources that may have been moved since.
    scans, sources = {}, {}
    for spec in cfg.subjects:
        if spec in todo:
            src_path = _resolve(cfg, spec.source)
            dst = cfg.subject_path(spec.id)
            if src_path.resolve() == Path(dst).resolve():
                raise IngestError(
                    f"subject {spec.id!r}: 'source' and 'path' are both {dst}; "
                    f"the store has to be written somewhere else"
                )
            source = open_source(src_path)
            sources[spec.id] = (src_path, source)
            scans[spec.id] = (source.shape, scan_voxel(cfg, spec, source, say))
            continue
        vol = Volume.open(cfg.subject_path(spec.id), cfg.backend)
        recorded = (vol.meta.get("provenance") or {}).get("placement")
        if recorded:
            pl = Placement.from_json(recorded)
            scans[spec.id] = (pl.shape, pl.voxel_mm)
        else:
            g = vol.native_grid
            scans[spec.id] = (g.shape, (g.spacing_mm,) * 3)

    try:
        grid, placements = resolve_run_grid(
            scans,
            cfg.grid.spacing_mm,
            cfg.grid.shape,
            cfg.grid.align,
            reference=cfg.grid.reference,
        )
    except ValueError as exc:
        raise IngestError(f"cannot choose a run grid: {exc}") from None
    moved = {
        sid: Volume.open(cfg.subject_path(sid), cfg.backend).native_grid
        for sid in kept
    }
    moved = {sid: g for sid, g in moved.items() if not same_grid(g, grid)}
    if moved:
        detail = ", ".join(f"{sid} {fmt_grid(g)}" for sid, g in moved.items())
        raise IngestError(
            f"with these scans and grid settings the run grid is "
            f"{fmt_grid(grid)}, but existing stores are on another grid "
            f"({detail}). The grid follows from every scan, so adding a larger "
            f"or finer scan, or changing the grid block, moves it. Run "
            f"'chunkreg setup --reingest' to put every scan on the new grid."
        )

    say(
        f"  run grid {fmt_grid(grid)} "
        + (
            f"(from subject {cfg.grid.reference!r}, align {cfg.grid.align})"
            if cfg.grid.reference
            else f"(grid.spacing_mm {cfg.grid.spacing_mm}, align {cfg.grid.align})"
        )
    )
    _backend.write_json(
        cfg.grid_record_path,
        {
            "shape": list(grid.shape),
            "spacing_mm": grid.spacing_mm,
            "origin_mm": list(grid.origin_mm),
            "align": cfg.grid.align,
            "placements": {sid: p.to_json() for sid, p in placements.items()},
        },
        cfg.backend,
    )

    # Ingest is one long read of each scan off shared storage, and the scans
    # are independent objects written to separate stores, so the subjects
    # overlap rather than queue. Threads, not processes: the time goes in zarr
    # decompression and compression, which release the GIL, and a thread keeps
    # whatever device this process configured instead of building its own.
    def ingest_one(spec) -> list[str]:
        dst = cfg.subject_path(spec.id)
        src_path, source = sources[spec.id]
        placement = placements[spec.id]
        how, notes = placement_notes(grid, placement)
        view = ResampledSource(source, placement, grid)
        provenance = {
            "source": str(src_path),
            "config": cfg.fingerprint(),
            "placement": placement.to_json(),
        }
        # A zarr already exactly on the run grid is read where it is: copying
        # it would only duplicate the largest level. Anything that changes the
        # voxels on the way in (a median, another dtype) needs the copy.
        linkable = (
            cfg.ingest.link
            and view.is_identity
            and is_zarr(src_path)
            and not is_chunkreg_store(src_path)
            and cfg.ingest.median_radius is None
            and cfg.ingest.dtype in ("auto", str(source.dtype))
        )
        if linkable:
            from .io_formats import open_zarr_in_place

            arr = open_zarr_in_place(src_path)
            layout = f"chunks {tuple(getattr(arr, 'chunks', ()) or ())}"
            shards = getattr(arr, "shards", None)
            layout += f", shards {tuple(shards)}" if shards else ", not sharded"
            how = (
                f"read in place, full resolution not copied ({layout}; set "
                f"ingest.link false to copy it into {cfg.profile.core}-voxel shards)"
            )
            vol = link_source(
                src_path,
                dst,
                grid.spacing_mm,
                cfg.profile,
                origin_mm=grid.origin_mm,
                backend=cfg.backend,
                overwrite=True,
                percentiles=cfg.ingest.percentiles,
                provenance=provenance,
            )
        else:
            vol = ingest_source(
                view,
                dst,
                grid.spacing_mm,
                cfg.profile,
                origin_mm=grid.origin_mm,
                dtype=cfg.ingest.dtype,
                backend=cfg.backend,
                overwrite=True,
                percentiles=cfg.ingest.percentiles,
                median_radius=cfg.ingest.median_radius,
                provenance=provenance,
            )
        lo, hi = vol.normalisation
        lines = [
            f"  {spec.id}: {source.description} -> {dst}\n"
            f"      scan {'x'.join(str(n) for n in source.shape)} @ "
            f"{fmt_voxel(placement.voxel_mm)}, {how}\n"
            f"      {vol.native_grid.shape} @ {grid.spacing_mm:g} mm, "
            f"{vol.n_levels} levels, {vol.array(vol.n_levels - 1).dtype}, "
            f"window [{lo:.4g}, {hi:.4g}]"
        ]
        lines += [f"      warning: {note}" for note in notes]
        return lines

    def guarded(spec) -> list[str]:
        try:
            return ingest_one(spec)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 - re-raised with the subject named
            raise RuntimeError(f"ingesting subject {spec.id!r}: {exc}") from exc

    n_jobs = max(1, min(int(jobs), len(todo)))
    if n_jobs > 1:
        say(f"  {len(todo)} subject(s), {n_jobs} at a time")
    if n_jobs == 1:
        output = {spec.id: guarded(spec) for spec in todo}
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futures = {spec.id: ex.submit(guarded, spec) for spec in todo}
        # Reported in config order however they finished, so the log reads the
        # same whether or not ingest ran concurrently.
        output = {sid: fut.result() for sid, fut in futures.items()}

    for spec in todo:
        for line in output[spec.id]:
            say(line)
    return [spec.id for spec in todo], kept


def check_subjects(cfg: RunConfig, say: Callable[[str], None] | None = None) -> None:
    """Every subject must be a store this profile can run on, on one grid."""
    from .store import Volume, check_volume

    say = say or (lambda _msg: None)
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
            say(f"  warning: {spec.id}: {w}")
        grids[spec.id] = vol.native_grid
    first = next(iter(grids.values()), None)
    if first is not None and not all(same_grid(g, first) for g in grids.values()):
        detail = ", ".join(f"{k} {fmt_grid(g)}" for k, g in grids.items())
        failures.append(
            f"subjects are on different grids ({detail}). Every store has to be "
            f"on the one run grid; 'chunkreg setup --reingest' resamples every "
            f"scan onto it"
        )
    if failures:
        raise IngestError(
            "subject stores are not ready:\n  "
            + "\n  ".join(failures)
            + "\nRe-run 'chunkreg setup --reingest' after fixing the sources."
        )
    say(f"  {len(grids)} subject store(s) sharded and on one grid")

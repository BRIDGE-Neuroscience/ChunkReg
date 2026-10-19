"""Putting the scans named by a config onto disk as stores the pipeline can run.

Ingest was reachable only through ``chunkreg setup``, which left a script able
to build a template but not to prepare the volumes one is built from. Nothing
here is new: it is the work ``setup`` has always done, moved to where the
Python interface can call it too. :func:`chunkreg.register` is the caller that
needed it.

Two things are decided here and nowhere else. The run grid follows from the
whole cohort at once (see :mod:`chunkreg.cohort`), because every pass reads
subject level *l* and template level *l* at the same voxel indices, so one
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


def _has_pyramid(path) -> bool:
    """Is this zarr an OME multiscale group with more than one dataset?"""
    import json

    doc = Path(path) / "zarr.json"
    if not doc.exists():
        return False
    try:
        attrs = json.loads(doc.read_text(encoding="utf-8")).get("attributes") or {}
    except (OSError, ValueError):
        return False
    ome = attrs.get("ome", attrs)
    ms = ome.get("multiscales") if isinstance(ome, dict) else None
    return bool(ms) and len(ms[0].get("datasets") or []) > 1


def _affine_matrix(cfg, value):
    """A warm-start affine as a 4x4 array, from the config's list or a JSON file.

    A file may hold the matrix under any key; the first 4x4 or 3x4 numeric
    list found is taken (``T_pair_bigbrain_to_s2146_zyx_um``, ``affine_um``...).
    """
    import json

    import numpy as np

    if isinstance(value, str):
        path = _resolve(cfg, value)
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise IngestError(f"cannot read the warm-start affine {path}: {exc}") from None
        found = None
        candidates = [doc] if isinstance(doc, list) else list(doc.values())
        for cand in candidates:
            try:
                a = np.asarray(cand, dtype=np.float64)
            except (TypeError, ValueError):
                continue
            if a.shape in ((4, 4), (3, 4)):
                found = a
                break
        if found is None:
            raise IngestError(f"{path} holds no 4x4 or 3x4 matrix")
        value = found
    T = np.asarray(value, dtype=np.float64)
    if T.shape == (3, 4):
        T = np.vstack([T, [0.0, 0.0, 0.0, 1.0]])
    return T


def _physical_frame(cfg, fixed_id: str, grid, sources, say):
    """The fixed scan's own physical frame: voxel ``i`` at ``scale * i + translation`` um.

    From its OME-Zarr metadata when the source is one; a plain array or a
    NIfTI has no stated origin, so its physical frame is taken as voxel index
    times voxel size. The run grid is the fixed scan's own sampling (it is the
    grid reference), so this is also the map from run voxels to physical um.
    """
    from .store import Volume

    src = sources.get(fixed_id)
    src_path = src[0] if src else None
    if src_path is None:
        spec = next(s for s in cfg.subjects if s.id == fixed_id)
        if spec.source is not None:
            src_path = _resolve(cfg, spec.source)
    if src_path is not None and (Path(src_path) / "zarr.json").exists():
        try:
            from .affine_view import ome_datasets

            datasets = ome_datasets(src_path)
            finest = min(datasets, key=lambda d: max(d["scale_um"]))
            if tuple(finest["shape"]) == tuple(grid.shape):
                return finest["scale_um"], finest["translation_um"]
            say(
                f"  note: {fixed_id}: the run grid {grid.shape} is not the finest dataset "
                f"of {src_path} ({finest['shape']}); its physical frame is taken as "
                f"voxel x spacing"
            )
        except (OSError, ValueError, KeyError):
            pass
    um = grid.spacing_mm * 1000.0
    return [um] * 3, [0.0, 0.0, 0.0]


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
    # sources that may have been moved since. A scan read through a warm-start
    # affine lives in its own frame and is sampled onto whatever grid the rest
    # decide, so it takes no part in choosing it.
    scans, sources = {}, {}
    through_affine = {s.id for s in cfg.subjects if (s.warm_start or {}).get("affine_um")}
    for spec in cfg.subjects:
        if spec in todo:
            src_path = _resolve(cfg, spec.source)
            dst = cfg.subject_path(spec.id)
            if src_path.resolve() == Path(dst).resolve():
                raise IngestError(
                    f"subject {spec.id!r}: 'source' and 'path' are both {dst}; "
                    f"the store has to be written somewhere else"
                )
            if spec.id in through_affine:
                sources[spec.id] = (src_path, None)
                continue
            source = open_source(src_path)
            sources[spec.id] = (src_path, source)
            scans[spec.id] = (source.shape, scan_voxel(cfg, spec, source, say))
            continue
        if spec.id in through_affine:
            continue
        vol = Volume.open(cfg.subject_path(spec.id), cfg.backend)
        recorded = (vol.meta.get("provenance") or {}).get("placement")
        if recorded:
            pl = Placement.from_json(recorded)
            scans[spec.id] = (pl.shape, pl.voxel_mm)
        else:
            g = vol.native_grid
            scans[spec.id] = (g.shape, (g.spacing_mm,) * 3)

    if not scans:
        raise IngestError(
            "every subject is read through a warm-start affine; one scan has to "
            "define the run grid (name it with grid.reference)"
        )
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
    for sid in through_affine:
        placements[sid] = Placement.from_grid(grid)
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
    def ingest_through_affine(spec) -> list[str]:
        """A store that samples the scan through its warm-start affine."""
        from . import affine_view as _av
        from .store import link_affine_source

        dst = cfg.subject_path(spec.id)
        src_path, _ = sources[spec.id]
        ws = spec.warm_start
        fixed_id = cfg.grid.reference or cfg.template_subject
        T = _affine_matrix(cfg, ws["affine_um"])
        scale_um, trans_um = _physical_frame(cfg, fixed_id, grid, sources, say)
        world_to_src = _av.world_to_source_um(grid, scale_um, trans_um, T)
        bg = ws.get("background")
        spec_dict = _av.build_spec(
            src_path, world_to_src, background=None if bg is None else tuple(bg)
        )
        vol = link_affine_source(
            spec_dict,
            dst,
            grid,
            cfg.profile,
            backend=cfg.backend,
            overwrite=True,
            percentiles=cfg.ingest.percentiles,
            provenance={
                "source": str(src_path),
                "config": cfg.fingerprint(),
                "warm_start": {k: v for k, v in ws.items() if k != "field"},
                "placement": Placement.from_grid(grid).to_json(),
            },
        )
        lo, hi = vol.normalisation
        used = [vol.array(k).dataset["path"] for k in range(vol.n_levels)]
        return [
            f"  {spec.id}: {src_path} -> {dst}\n"
            f"      read in its own frame through the warm-start affine onto the run "
            f"grid ({'x'.join(str(n) for n in grid.shape)} @ {grid.spacing_mm:g} mm), "
            f"nothing resampled on disk\n"
            f"      {vol.n_levels} levels from source datasets {used}, window [{lo:.4g}, {hi:.4g}]"
        ]

    def ingest_one(spec) -> list[str]:
        if spec.id in through_affine:
            return ingest_through_affine(spec)
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
            from .store import link_pyramid

            arr = open_zarr_in_place(src_path)
            layout = f"chunks {tuple(getattr(arr, 'chunks', ()) or ())}"
            shards = getattr(arr, "shards", None)
            layout += f", shards {tuple(shards)}" if shards else ", not sharded"
            vol = None
            if _has_pyramid(src_path):
                # The scan carries its own pyramid: every level whose shape is
                # the run's is read where it lies, and only the rest is built.
                try:
                    vol = link_pyramid(
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
                except ValueError as exc:
                    say(f"  note: {spec.id}: pyramid of {src_path} not linked ({exc})")
                    vol = None
            if vol is not None:
                linked = [k for k in range(vol.n_levels) if vol.is_linked(k)]
                how = (
                    f"read in place at levels {linked} of 0..{vol.n_levels - 1} from "
                    f"the scan's own pyramid ({layout})"
                )
                skipped = (vol.meta.get("provenance") or {}).get("pyramid_levels_not_linked")
                if skipped:
                    notes = list(notes) + [
                        f"level {k} of the scan's pyramid not used: {why}; built here instead"
                        for k, why in sorted(skipped.items())
                    ]
            else:
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
            f"{vol.n_levels} levels, {vol.array(0).dtype}, "
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

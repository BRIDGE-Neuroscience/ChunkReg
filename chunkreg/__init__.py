"""chunkreg: hierarchical chunked co-registration of N massive volumes.

A resolution pyramid with one fixed chunk geometry at every level. The coarsest
level is by construction the level at which a whole volume fits inside a single
chunk, so there is no separate whole-volume code path. Every volumetric object
is a chunked array on disk; similarity is computed on anatomix feature channels
registered by FireANTs.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .grid import (  # noqa: F401
    Chunk,
    GridSpec,
    Profile,
    chunks_touching,
    pyramid,
    pyramid_depth,
    shard_grid,
    tile,
    window,
)

# The rest of the public API, imported on first use. Naming them here is what
# makes `chunkreg.build_template(...)` work from a script the way the CLI's
# `chunkreg run` does; importing them eagerly would pull the passes, engines
# and feature extractors into every `import chunkreg`, including the ones that
# only want a GridSpec.
_LAZY = {
    "load_config": ("chunkreg.config", "load_config"),
    "RunConfig": ("chunkreg.config", "RunConfig"),
    "get_profile": ("chunkreg.config", "get_profile"),
    "plan": ("chunkreg.planner", "plan"),
    "Plan": ("chunkreg.planner", "Plan"),
    "calibrate": ("chunkreg.calibrate", "calibrate"),
    "probe": ("chunkreg.probe", "probe"),
    "selftest": ("chunkreg.selftest", "run_selftest"),
    "build_template": ("chunkreg.pipelines", "build_template"),
    "register_pair": ("chunkreg.pipelines", "register_pair"),
    "pair_config": ("chunkreg.pipelines", "pair_config"),
    "register": ("chunkreg.twoimage", "register"),
    "PairResult": ("chunkreg.twoimage", "PairResult"),
    "RunResult": ("chunkreg.pipelines", "RunResult"),
    "apply_field": ("chunkreg.apply", "apply_field"),
    "export_store": ("chunkreg.apply", "export_store"),
    "resample_to_scan": ("chunkreg.apply", "resample_to_scan"),
    "scan_view": ("chunkreg.apply", "scan_view"),
    "save_config": ("chunkreg.config", "save_config"),
    "ingest_subjects": ("chunkreg.ingest", "ingest_subjects"),
    "Placement": ("chunkreg.cohort", "Placement"),
    "Volume": ("chunkreg.store", "Volume"),
    "Field": ("chunkreg.store", "Field"),
    "get_runner": ("chunkreg.runners", "get_runner"),
}


def __getattr__(name: str):
    try:
        module, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'chunkreg' has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(module), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "__version__",
    "GridSpec",
    "Profile",
    "Chunk",
    "pyramid",
    "pyramid_depth",
    "tile",
    "window",
    "chunks_touching",
    "shard_grid",
    *sorted(_LAZY),
]

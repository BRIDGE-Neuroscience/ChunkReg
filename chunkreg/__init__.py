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
]

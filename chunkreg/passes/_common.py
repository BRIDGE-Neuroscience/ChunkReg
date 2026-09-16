"""Helpers shared by the passes."""

from __future__ import annotations

import numpy as np

from .. import fields as _fields
from ..grid import GridSpec
from ..store import Volume


def warped_subject_block(
    volume: Volume,
    level: int,
    origin,
    shape,
    disp_mm: np.ndarray,
    spacing_mm: float,
    normalise: bool = True,
) -> np.ndarray:
    """Read a subject through a displacement field, with a sized margin.

    The margin is derived from the field actually present rather than from a
    configured bound, because the field at a seeded level carries the whole
    accumulated displacement from every coarser level and can be many times the
    halo. Reading only the requested box and warping in place would silently
    fade the result to zero wherever the field points outside it.
    """
    margin = _fields.required_margin(disp_mm, spacing_mm)
    src_origin = tuple(int(o) - margin for o in origin)
    src_shape = tuple(int(s) + 2 * margin for s in shape)
    block = volume.read_padded(level, src_origin, src_shape, normalise=normalise)
    return _fields.warp_offset(block, (margin, margin, margin), disp_mm, spacing_mm)


def tissue_fraction(patch: np.ndarray, threshold: float = 0.0) -> float:
    """Fraction of a normalised patch above a background threshold."""
    a = np.asarray(patch, dtype=np.float32)
    if a.size == 0:
        return 0.0
    thr = threshold if threshold > 0 else max(0.02, 0.05 * float(a.max()))
    return float(np.mean(a > thr))


def core_view(block: np.ndarray, chunk, lead: int = 0) -> np.ndarray:
    """Slice a padded-box array down to the chunk's core."""
    off = chunk.core_offset_in_pad
    key = (slice(None),) * lead + tuple(
        slice(o, o + s) for o, s in zip(off, chunk.core_shape)
    )
    return block[key]

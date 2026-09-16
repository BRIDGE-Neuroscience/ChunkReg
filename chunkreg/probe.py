"""Measure how far apart the cohort actually is.

The pyramid exists because a chunk cannot see a deformation longer than its own
width. Whether a given cohort needs every level, or could be handled at fewer,
is a property of the data and takes minutes to measure at the coarsest level.

The statistic is the residual displacement after registering at level 0, where
the whole volume is a single chunk and nothing is clipped by a halo. Compared
against each level's chunk width it says which levels are doing real work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import fields as _fields
from . import stats as _stats
from .config import RunConfig
from .engines import get_engine
from .features import get_extractor
from .grid import GridSpec, pyramid
from .store import Volume

__all__ = ["ProbeReport", "probe"]


@dataclass
class ProbeReport:
    pairs: list[tuple[str, str]]
    d50_mm: float
    d99_mm: float
    levels: list[tuple[int, float, float]] = field(default_factory=list)
    """(level, chunk width L in mm, clamp in mm)."""

    def format(self) -> str:
        out = [
            f"probe on {len(self.pairs)} pair(s) at the coarsest level",
            f"  median residual {self.d50_mm:.3f} mm, "
            f"99th percentile {self.d99_mm:.3f} mm",
            "",
            f"{'lvl':>3} {'L (mm)':>9} {'d99/L':>8} {'clamp mm':>10}  verdict",
        ]
        for level, width, clamp in self.levels:
            ratio = self.d99_mm / width if width else 0.0
            if ratio < 0.05:
                verdict = "chunks see the whole deformation"
            elif ratio < 0.25:
                verdict = "chunks see most of it"
            else:
                verdict = "too large for a chunk; this level needs its parent"
            out.append(
                f"{level:>3} {width:>9.1f} {ratio:>8.2f} {clamp:>10.3f}  {verdict}"
            )
        out.append("")
        out.append(
            "  A level whose chunk width is comfortably larger than d99 can "
            "resolve the deformation on its own; one below it is relying on "
            "the level above to have done so."
        )
        return "\n".join(out)


def probe(cfg: RunConfig, pairs: int = 3, engine: str | None = None) -> ProbeReport:
    """Register a few pairs at the coarsest level and report the residual."""
    ids = list(cfg.subject_ids)
    if len(ids) < 2:
        raise ValueError("probe needs at least two subjects to compare")

    vols = {s: Volume.open(cfg.subject_path(s), cfg.backend) for s in ids}
    native = vols[ids[0]].native_grid
    grid = pyramid(native, cfg.profile)[0]

    from . import xp

    eng = get_engine(engine or cfg.engine_for(xp.device_name()))
    ex = get_extractor(cfg.profile.features)
    ex.setup()
    stages = list(cfg.levels.stages(0))

    fixed_id = ids[0]
    fixed_img = vols[fixed_id].read_padded(0, (0, 0, 0), grid.shape, normalise=True)
    fixed = ex(fixed_img, grid.spacing_mm)

    hist = _stats.empty()
    used: list[tuple[str, str]] = []
    for moving_id in ids[1 : 1 + pairs]:
        moving_img = vols[moving_id].read_padded(
            0, (0, 0, 0), grid.shape, normalise=True
        )
        res = eng.register(fixed, ex(moving_img, grid.spacing_mm), grid.spacing_mm, stages)
        hist += _stats.histogram(_fields.magnitude(res.disp_mm))
        used.append((fixed_id, moving_id))

    levels = [
        (k, cfg.profile.core * g.spacing_mm, cfg.profile.d_max_mm(g.spacing_mm))
        for k, g in enumerate(pyramid(native, cfg.profile))
    ]
    return ProbeReport(
        pairs=used,
        d50_mm=_stats.percentile(hist, 50.0),
        d99_mm=_stats.percentile(hist, 99.0),
        levels=levels,
    )

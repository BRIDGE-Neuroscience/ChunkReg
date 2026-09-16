"""Build a feature sheet: the same anatomy, every level, several subjects.

Answers the question the design assumes and never checks: are the learned
features carrying real structure at this scale, or is the extractor being paid
for sixteen channels of nothing?

Two things make the comparison honest.

**The box is fixed in world millimetres, not in voxels.** Every level samples
the same physical cube, so the anatomy is identical down the row and only the
resolution changes. Fixing it in voxels instead would show a different amount
of anatomy at every level and the columns would not be comparable.

**Features are extracted with a margin and then cropped.** An extractor with a
receptive field reads beyond any box it is given, so the border of a
naively-extracted tile is the network responding to zero padding. The margin is
taken from the extractor's own ``r_f``, and a minimum context size keeps a
coarse level, where the box may be only a few voxels across, from collapsing a
downsampling network to nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from . import featureviz as _viz
from .config import RunConfig
from .features import get_extractor
from .grid import GridSpec, pyramid
from .store import Field, Volume

__all__ = ["FeatureReport", "build_feature_report", "sample_box"]

MIN_CONTEXT_VOX = 64
"""Smallest cube an extractor is run on. A four-level downsampling network fed
a sixteen-voxel cube has nothing left at its coarsest stage, so a small box at
a coarse level is grown to this and cropped back afterwards."""


@dataclass
class FeatureReport:
    tiles: list
    subjects: list
    levels: list
    basis: _viz.RGBBasis
    centre_world: tuple
    size_mm: float
    extractor: str
    warped: bool
    basis_mode: str = "shared"
    bases: dict = None

    def to_json(self) -> dict:
        return {
            "extractor": self.extractor,
            "channels": self.basis.n_channels,
            "basis_mode": self.basis_mode,
            "pooled_explained_variance": round(self.basis.explained, 4),
            "centre_world_mm": [round(float(v), 4) for v in self.centre_world],
            "box_mm": round(self.size_mm, 4),
            "warped_by_current_fields": self.warped,
            "subjects": list(self.subjects),
            "levels": list(self.levels),
            "tiles": [
                {
                    "subject": t.subject,
                    "level": t.level,
                    "spacing_mm": round(t.spacing_mm, 6),
                    "box_mm": round(t.extent_mm, 4),
                    **{
                        k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in t.metrics.items()
                    },
                }
                for t in self.tiles
            ],
        }

    def summary(self) -> str:
        """A table ranking where the features are actually carrying structure."""
        rows = [
            f"extractor {self.extractor}, {self.basis.n_channels} channels, "
            f"box {self.size_mm:.2f} mm at "
            f"({', '.join(f'{v:.1f}' for v in self.centre_world)}) mm"
            + ("  [subjects warped by their current fields]" if self.warped else "")
        ]
        if self.basis_mode == "shared":
            rows.append(
                f"one shared projection, carrying {self.basis.explained:.0%} of "
                f"pooled variance. colours are comparable across every panel."
            )
        else:
            rows.append(
                "a separate projection per level, fitted across subjects. "
                "colours are comparable DOWN a column but NOT across columns; "
                "each level is shown at its own best contrast."
            )
        rows.append("")
        head = (
            f"{'lvl':>3} {'spacing':>9} {'box vox':>8} {'contrast':>10} "
            f"{'rank':>6} {'shown':>7}"
        )
        rows.append(head)
        rows.append("-" * len(head))
        for level in self.levels:
            at = [t for t in self.tiles if t.level == level]
            if not at:
                continue
            contrast = float(np.mean([t.metrics["contrast"] for t in at]))
            rank = float(np.mean([t.metrics["rank"] for t in at]))
            shown = [t.metrics.get("explained") for t in at]
            shown = [s for s in shown if s is not None]
            vox = int(round(self.size_mm / at[0].spacing_mm))
            rows.append(
                f"{level:>3} {at[0].spacing_mm:>9.4g} {vox:>8} {contrast:>10.4f} "
                f"{rank:>6.1f} {(np.mean(shown) if shown else 0):>6.0%}"
            )
        rows.append("")
        rows.append(
            "contrast near zero means the extractor found no structure at that "
            "scale. rank well below the channel count means the description has "
            "collapsed. shown well below 100% means the picture is hiding most "
            "of what the registration sees."
        )
        return "\n".join(rows)


def sample_box(
    volume: Volume,
    level: int,
    grid: GridSpec,
    centre_world,
    size_vox: int,
    extractor,
    field: Field | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one world-aligned cube and its features, edges trimmed.

    Returns the normalised intensity cube and its ``(C, Z, Y, X)`` features,
    both cropped back to ``size_vox`` after extraction so no voxel on screen was
    computed against padding.
    """
    centre_vox = np.rint(grid.voxel(np.asarray(centre_world, dtype=np.float64)))
    origin = (centre_vox - size_vox // 2).astype(np.int64)

    margin = int(getattr(extractor, "r_f", 0)) + 2
    need = max(size_vox + 2 * margin, MIN_CONTEXT_VOX)
    pad = (need - size_vox) // 2
    read_origin = origin - pad
    read_shape = (int(size_vox + 2 * pad),) * 3

    if field is None:
        block = volume.read_padded(level, read_origin, read_shape, normalise=True)
    else:
        from .passes._common import warped_subject_block

        u = field.read_dense(read_origin, read_shape)
        block = warped_subject_block(
            volume, level, read_origin, read_shape, u, grid.spacing_mm, normalise=True
        )

    feats = extractor(block, grid.spacing_mm)
    sl = (slice(pad, pad + size_vox),) * 3
    # The report renders images on the host, so only the cropped cube and its
    # features come back.
    from . import xp

    return xp.get(block[sl]), xp.get(feats[(slice(None),) + sl])


def build_feature_report(
    cfg: RunConfig,
    centre_world: Sequence[float] | None = None,
    size_mm: float | None = None,
    levels: Sequence[int] | None = None,
    subjects: Sequence[str] | None = None,
    plane: str = "axial",
    warped: bool = False,
    extractor_name: str | None = None,
    basis_mode: str = "shared",
    min_box_vox: int = 8,
) -> FeatureReport:
    """Sample every (subject, level) and fit one shared colour projection."""
    subject_ids = list(subjects) if subjects else list(cfg.subject_ids)
    if not subject_ids:
        raise ValueError("no subjects to report on")

    vols = {s: Volume.open(cfg.subject_path(s), cfg.backend) for s in subject_ids}
    first = vols[subject_ids[0]]
    native = first.native_grid

    # The levels come from the store, not from the profile. A volume ingested
    # under a different profile has its own pyramid depth, and indexing it by a
    # depth derived from the current config reads the wrong array: level 0 of a
    # three-level store is the coarsest, and asking for a native-resolution box
    # there lands outside it and returns zeros.
    grids = first.grids()
    for sid, v in vols.items():
        if v.n_levels != first.n_levels:
            raise ValueError(
                f"subject {sid!r} has {v.n_levels} levels but {subject_ids[0]!r} "
                f"has {first.n_levels}; they were ingested under different "
                f"profiles and cannot be compared level by level"
            )

    level_ids = list(levels) if levels else list(range(len(grids)))
    for k in level_ids:
        if not 0 <= k < len(grids):
            raise ValueError(
                f"level {k} out of range; this store has {len(grids)} levels "
                f"(0 is the coarsest, {len(grids) - 1} is native)"
            )

    if centre_world is None:
        centre_world = tuple(native.world(np.asarray(native.shape, float) / 2.0 - 0.5))
    centre_world = tuple(float(v) for v in centre_world)

    if size_mm is None:
        # One chunk core at native resolution: the box the finest level's
        # registration actually operates on.
        size_mm = cfg.profile.core * native.spacing_mm

    name = extractor_name or cfg.profile.features
    extractor = get_extractor(name)
    extractor.setup()

    raw: list[tuple[str, int, np.ndarray, np.ndarray, float]] = []
    for level in level_ids:
        grid = grids[level]
        size_vox = max(int(round(size_mm / grid.spacing_mm)), min_box_vox)
        for sid in subject_ids:
            f = None
            if warped:
                path = cfg.field_path(level, sid)
                if Field.exists(path, cfg.backend):
                    f = Field.open(path, cfg.backend)
            block, feats = sample_box(
                vols[sid], level, grid, centre_world, size_vox, extractor, f
            )
            raw.append((sid, level, block, feats, grid.spacing_mm))

    if basis_mode not in ("shared", "per-level"):
        raise ValueError(
            f"unknown basis mode {basis_mode!r}; use 'shared' or 'per-level'"
        )

    if basis_mode == "shared":
        # One projection over every tile, which is what makes the colours mean
        # the same thing in every panel and is the only mode in which the
        # columns can honestly be compared.
        shared = _viz.fit_rgb_basis([f for _, _, _, f, _ in raw])
        bases = {k: shared for k in level_ids}
    else:
        # Each level at its own best contrast. Use this to check whether a
        # level that renders dead under the shared basis has structure in its
        # own subspace; do not use it to compare levels.
        bases = {
            k: _viz.fit_rgb_basis([f for _, lv, _, f, _ in raw if lv == k])
            for k in level_ids
        }
    basis = bases[level_ids[0]] if basis_mode == "per-level" else shared

    tiles = []
    for sid, level, block, feats, spacing in raw:
        b = bases[level]
        metrics = _viz.feature_metrics(feats, b, tissue=block > 0.05)
        rgb_vol = b.to_rgb(feats)
        tiles.append(
            _viz.Tile(
                subject=sid,
                level=level,
                spacing_mm=spacing,
                extent_mm=size_mm,
                intensity=_viz.orthoslice(block, plane),
                rgb=_viz.orthoslice(rgb_vol, plane),
                metrics=metrics,
            )
        )

    return FeatureReport(
        tiles=tiles,
        subjects=subject_ids,
        levels=level_ids,
        basis=basis,
        centre_world=centre_world,
        size_mm=float(size_mm),
        extractor=name,
        warped=bool(warped),
        basis_mode=basis_mode,
        bases=bases,
    )


def write_feature_report(report: FeatureReport, out_dir, tile_px: int = 192) -> dict:
    """Write the contact sheet, per-tile PNGs and the metrics file."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if report.basis_mode == "shared":
        colour = f"shared projection, {report.basis.explained:.0%} of variance"
    else:
        colour = "per-level projection, columns NOT comparable"
    title = (
        f"chunkreg features: {report.extractor} "
        f"({report.basis.n_channels} ch)  box {report.size_mm:.2f} mm  {colour}"
    )
    sheet = _viz.contact_sheet(
        report.tiles,
        report.subjects,
        report.levels,
        tile_px=tile_px,
        title=title,
        basis_mode=report.basis_mode,
    )
    sheet_path = out / "feature_sheet.png"
    _viz.save_png(sheet_path, sheet)

    for t in report.tiles:
        _viz.save_png(out / f"L{t.level}_{t.subject}_features.png", t.rgb)
        _viz.save_png(out / f"L{t.level}_{t.subject}_intensity.png", t.intensity)

    metrics_path = out / "feature_metrics.json"
    metrics_path.write_text(json.dumps(report.to_json(), indent=2), encoding="utf-8")

    return {
        "sheet": str(sheet_path),
        "metrics": str(metrics_path),
        "tiles": len(report.tiles),
        "dir": str(out),
    }

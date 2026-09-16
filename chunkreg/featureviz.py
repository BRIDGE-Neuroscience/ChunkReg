"""Render feature channels as RGB, so feature quality can be judged per scale.

The design assumes learned features are worth their cost at every pyramid
level and never checks it. This is the check. It renders the same anatomy, at
every level, for several subjects, as both intensity and features, on one
sheet.

The load-bearing decision is that **one projection is shared by every tile**.
Feature channels are mapped to RGB by projecting onto the three leading
components of their covariance, and that basis, its mean, and the intensity
range are all fitted once on a pooled sample from every subject and every
level. Fit per tile instead and each panel gets its own colour meaning, which
makes the sheet look informative while comparing nothing. The cost of sharing
is that a level whose features occupy a different subspace renders dark rather
than colourful, and that is the honest signal, not a defect.

Three numbers accompany each tile because colour alone cannot answer the
question:

``explained``
    How much of that tile's feature variance the three displayed components
    actually capture. At 0.9 the picture is the features; at 0.3 most of what
    the registration sees is not on screen.
``contrast``
    Mean spatial gradient of the normalised features. Near zero means the
    extractor found no structure at this scale, whatever the colours suggest.
``rank``
    Participation ratio of the channel covariance: how many channels the
    descriptor is really using. A sixteen-channel extractor with a rank near
    one has collapsed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "RGBBasis",
    "Tile",
    "fit_rgb_basis",
    "feature_metrics",
    "orthoslice",
    "contact_sheet",
    "save_png",
]

PLANES = {"axial": 0, "coronal": 1, "sagittal": 2}


# --------------------------------------------------------------------------- #
# The shared projection
# --------------------------------------------------------------------------- #
@dataclass
class RGBBasis:
    """A fixed feature-to-colour map, shared by every tile on a sheet."""

    basis: np.ndarray
    """(3, C) the three leading components of the pooled feature covariance."""
    mean: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    explained: float
    """Fraction of pooled variance the three components carry."""
    n_channels: int
    spectrum: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def to_rgb(self, feats: np.ndarray) -> np.ndarray:
        """Map ``(C, ...)`` features to ``(..., 3)`` in [0, 1]."""
        f = np.asarray(feats, dtype=np.float32)
        if f.shape[0] != self.n_channels:
            raise ValueError(
                f"basis was fitted for {self.n_channels} channels but got "
                f"{f.shape[0]}; a basis cannot be reused across extractors"
            )
        flat = f.reshape(f.shape[0], -1) - self.mean[:, None]
        proj = self.basis @ flat
        span = np.maximum(self.hi - self.lo, 1e-8)[:, None]
        rgb = np.clip((proj - self.lo[:, None]) / span, 0.0, 1.0)
        return rgb.reshape((3,) + f.shape[1:]).transpose(
            *range(1, f.ndim), 0
        )


def fit_rgb_basis(
    samples: Iterable[np.ndarray],
    max_vox: int = 200_000,
    seed: int = 0,
    percentiles: tuple[float, float] = (1.0, 99.0),
) -> RGBBasis:
    """Fit one feature-to-colour projection over a pooled sample.

    ``samples`` is an iterable of ``(C, ...)`` feature arrays, typically one per
    (subject, level). Every one contributes to the basis, so the resulting
    colours are comparable across all of them.
    """
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    n_channels = None
    for f in samples:
        a = np.asarray(f, dtype=np.float32)
        if n_channels is None:
            n_channels = a.shape[0]
        elif a.shape[0] != n_channels:
            raise ValueError(
                f"samples disagree on channel count: {n_channels} then {a.shape[0]}"
            )
        flat = a.reshape(a.shape[0], -1).T
        take = min(len(flat), max(1, max_vox // 8))
        idx = rng.choice(len(flat), size=take, replace=False)
        rows.append(flat[idx])
    if not rows:
        raise ValueError("fit_rgb_basis needs at least one sample")

    x = np.concatenate(rows, axis=0).astype(np.float64)
    mean = x.mean(axis=0)
    centred = x - mean
    if n_channels == 1:
        # A single channel has no basis to find; show it on all three guns so a
        # grey tile reads as intensity rather than as a failed projection.
        basis = np.ones((3, 1), dtype=np.float32)
        s = np.array([np.sqrt((centred**2).sum())])
        explained = 1.0
    else:
        _, s, vt = np.linalg.svd(centred, full_matrices=False)
        basis = vt[:3].astype(np.float32)
        if basis.shape[0] < 3:
            basis = np.vstack(
                [basis, np.zeros((3 - basis.shape[0], n_channels), np.float32)]
            )
        total = float((s**2).sum()) or 1.0
        explained = float((s[:3] ** 2).sum() / total)
        # Component signs are arbitrary. Fix them so a rerun gives the same
        # colours: make the largest-magnitude loading positive.
        for k in range(3):
            row = basis[k]
            if row.size and row[np.argmax(np.abs(row))] < 0:
                basis[k] = -row

    proj = basis @ centred.T
    lo = np.percentile(proj, percentiles[0], axis=1).astype(np.float32)
    hi = np.percentile(proj, percentiles[1], axis=1).astype(np.float32)
    return RGBBasis(
        basis=basis,
        mean=mean.astype(np.float32),
        lo=lo,
        hi=hi,
        explained=explained,
        n_channels=int(n_channels),
        spectrum=np.asarray(s, dtype=np.float64),
    )


# --------------------------------------------------------------------------- #
# Per-tile metrics
# --------------------------------------------------------------------------- #
def feature_metrics(
    feats: np.ndarray, basis: RGBBasis | None = None, tissue: np.ndarray | None = None
) -> dict:
    """Quantities that say whether a tile's colours mean anything."""
    f = np.asarray(feats, dtype=np.float32)
    flat = f.reshape(f.shape[0], -1)

    centred = flat - flat.mean(axis=1, keepdims=True)
    cov = (centred @ centred.T) / max(flat.shape[1] - 1, 1)
    eig = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    total = float(eig.sum())

    # Participation ratio: the number of channels actually carrying variance.
    rank = float((total**2) / max(float((eig**2).sum()), 1e-20)) if total > 0 else 0.0

    grads = np.gradient(f, axis=tuple(range(1, f.ndim)))
    contrast = float(
        np.mean(np.sqrt(sum(np.square(g) for g in grads) + 1e-20))
    )

    explained = None
    if basis is not None and total > 0:
        proj = basis.basis @ centred
        explained = float(np.var(proj, axis=1).sum() / (total + 1e-20))
        explained = min(explained, 1.0)

    out = {
        "contrast": contrast,
        "rank": rank,
        "channels": int(f.shape[0]),
        "explained": explained,
    }
    if tissue is not None:
        out["tissue_fraction"] = float(np.mean(np.asarray(tissue) > 0))
    return out


# --------------------------------------------------------------------------- #
# Slicing and sheets
# --------------------------------------------------------------------------- #
def orthoslice(vol: np.ndarray, plane: str = "axial", lead: int = 0) -> np.ndarray:
    """Take the mid-slice of a volume along one plane."""
    if plane not in PLANES:
        raise ValueError(f"unknown plane {plane!r}; use {sorted(PLANES)}")
    axis = PLANES[plane] + lead
    mid = vol.shape[axis] // 2
    return np.take(vol, mid, axis=axis)


@dataclass
class Tile:
    """One cell of the sheet: a subject at a level."""

    subject: str
    level: int
    spacing_mm: float
    extent_mm: float
    intensity: np.ndarray
    rgb: np.ndarray
    metrics: dict = field(default_factory=dict)


def _to_uint8(a: np.ndarray) -> np.ndarray:
    return (np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(
        np.uint8
    )


def _resize_nearest(img: np.ndarray, size: int) -> np.ndarray:
    """Scale to a square tile with nearest neighbour.

    Nearest deliberately, not smooth: a coarse level should look blocky next to
    a fine one. Interpolating would hide the very difference the sheet exists
    to show.
    """
    h, w = img.shape[:2]
    zy = (np.arange(size) * h // size).clip(0, h - 1)
    zx = (np.arange(size) * w // size).clip(0, w - 1)
    return img[zy][:, zx]


def contact_sheet(
    tiles: Sequence[Tile],
    subjects: Sequence[str],
    levels: Sequence[int],
    tile_px: int = 192,
    title: str = "",
    basis_mode: str = "shared",
):
    """Lay out tiles as rows of subjects by columns of levels.

    Each cell shows intensity above and features below, so the pair can be read
    together: colour that does not follow the anatomy above it is the extractor
    responding to something other than structure.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "rendering a contact sheet needs Pillow: pip install 'chunkreg[viz]'"
        ) from exc

    by_key = {(t.subject, t.level): t for t in tiles}
    pad, label_h, header_h = 6, 34, 30 if title else 0
    left = 76
    cell_w = tile_px + pad
    cell_h = 2 * tile_px + label_h + pad

    width = left + len(levels) * cell_w + pad
    height = header_h + 26 + len(subjects) * cell_h + pad
    sheet = Image.new("RGB", (width, height), (18, 20, 24))
    draw = ImageDraw.Draw(sheet)

    if title:
        draw.text((pad, pad), title, fill=(235, 238, 242))

    for ci, level in enumerate(levels):
        x = left + ci * cell_w
        any_tile = next((by_key[(s, level)] for s in subjects if (s, level) in by_key), None)
        head = f"L{level}"
        if any_tile is not None:
            head += f"  {any_tile.spacing_mm:g}mm"
        draw.text((x + 2, header_h + 4), head, fill=(190, 200, 212))

    for ri, subject in enumerate(subjects):
        y = header_h + 26 + ri * cell_h
        draw.text((pad, y + tile_px // 2), subject, fill=(190, 200, 212))
        for ci, level in enumerate(levels):
            t = by_key.get((subject, level))
            x = left + ci * cell_w
            if t is None:
                draw.rectangle(
                    [x, y, x + tile_px, y + 2 * tile_px], outline=(60, 66, 74)
                )
                continue
            grey = _resize_nearest(_to_uint8(t.intensity), tile_px)
            sheet.paste(Image.fromarray(np.dstack([grey] * 3)), (x, y))
            rgb = _resize_nearest(_to_uint8(t.rgb), tile_px)
            sheet.paste(Image.fromarray(rgb), (x, y + tile_px))
            m = t.metrics
            line = (
                f"c {m.get('contrast', 0):.3f}  r {m.get('rank', 0):.1f}"
                f"/{m.get('channels', 0)}"
            )
            if m.get("explained") is not None:
                line += f"  e {m['explained']:.2f}"
            draw.text((x + 2, y + 2 * tile_px + 3), line, fill=(150, 160, 172))
            draw.text(
                (x + 2, y + 2 * tile_px + 16),
                f"{t.extent_mm:.1f}mm box",
                fill=(120, 130, 142),
            )

    # The legend must name the mode. A per-level sheet labelled "shared" invites
    # exactly the cross-column comparison that mode cannot support.
    projection = (
        "one shared projection, columns comparable"
        if basis_mode == "shared"
        else "per-level projection, columns NOT comparable"
    )
    legend = (
        f"top: intensity   bottom: features as RGB ({projection})   "
        "c=contrast  r=effective rank/channels  e=variance shown"
    )
    draw.text((pad, height - 14), legend, fill=(120, 130, 142))
    return sheet


def save_png(path, image) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if hasattr(image, "save"):
        image.save(str(path))
        return
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("saving a PNG needs Pillow: pip install 'chunkreg[viz]'") from exc
    a = np.asarray(image)
    if a.ndim == 2:
        a = np.dstack([_to_uint8(a)] * 3)
    elif a.dtype != np.uint8:
        a = _to_uint8(a)
    Image.fromarray(a).save(str(path))

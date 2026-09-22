"""Where a level is failing, written out every pass.

Every number a pass reports is a percentile over the whole level. ``d99`` says
one voxel in a hundred is worse than some distance; it cannot say whether
those voxels are in the cortex, the cerebellum or the skull boundary, and
every tuning decision made against it alone has been made blind to that.

Two things fix it, and both are produced by the run itself rather than pulled
afterwards:

**A residual map per subject per level.** The register pass already computes
the residual -- the clamped displacement a pass could not already explain --
and only kept its histogram. Its magnitude over each chunk's core now goes
into a single-level store on the displacement lattice, one per subject,
overwritten every pass so it always shows the latest. A chunk that emitted
its seed for lack of tissue writes zero, which is true: nothing was there to
register. A chunk that exhausted the fold ladder writes NaN, because zero
would read as "converged" where the truth is "no acceptable solution".

**One PNG per pass.** The template's three mid-slices above the residual's,
the residual on a fixed scale of zero to the level's clamp so passes are
comparable by eye. Small enough to open over SSH; the point is not to have to
pull a NIfTI to know whether a run is worth continuing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import fields as _fields
from . import xp as _xp
from .config import RunConfig
from .grid import GridSpec
from .passes._common import core_view
from .passes.manifest import Manifest
from .store import TaskEntry, Volume

__all__ = [
    "residual_path",
    "slices_path",
    "ensure_residual_stores",
    "write_residual_core",
    "render_pass",
]

PANEL_PX = 360
"""Longest side of one panel. Six panels a pass, so a few hundred kB."""


def residual_path(cfg: RunConfig, level: int, subject: str) -> Path:
    return cfg.level_dir(level) / "qc" / f"residual_{subject}.zarr"


def slices_path(cfg: RunConfig, level: int, iteration: int) -> Path:
    return cfg.level_dir(level) / "qc" / f"it{iteration}.png"


# --------------------------------------------------------------------------- #
# The residual map
# --------------------------------------------------------------------------- #
def _lattice_grid(cfg: RunConfig, grid: GridSpec) -> GridSpec:
    return grid.coarsened(cfg.profile.lattice_factor)


def ensure_residual_stores(cfg: RunConfig, manifest: Manifest) -> None:
    """Create the per-subject residual stores for a level, once.

    Like the accumulators: writing is owner-computes, creating is shared
    state, so the pipeline makes them before dispatching and a task that
    finds one missing tolerates losing the race.
    """
    lattice = _lattice_grid(cfg, manifest.grid)
    for subject in manifest.subjects:
        path = residual_path(cfg, manifest.level, subject)
        if Volume.exists(path, cfg.backend):
            continue
        try:
            Volume.create(
                path,
                lattice,
                cfg.profile,
                dtype="float32",
                backend=cfg.backend,
                overwrite=False,
                n_levels=1,
                provenance={"config": cfg.fingerprint(), "level": manifest.level,
                            "kind": "residual_mm"},
            )
        except FileExistsError:
            pass


def write_residual_core(
    cfg: RunConfig,
    manifest: Manifest,
    entry: TaskEntry,
    residual: Any,
    reason: str | None,
) -> None:
    """Write ``|residual|`` over one chunk's core to its subject's store.

    ``residual`` is the clamped displacement over the padded box, or ``None``
    for a chunk that emitted its seed; ``reason`` then says why, and decides
    between zero (nothing to register) and NaN (nothing acceptable found).
    """
    chunk = manifest.chunk(entry.chunk_id)
    f = int(cfg.profile.lattice_factor)
    lat_origin = tuple(int(o) // f for o in chunk.core_origin)
    lat_shape = tuple(-(-int(s) // f) for s in chunk.core_shape)

    if residual is None:
        fill = np.nan if reason == "folds" else 0.0
        block = np.full(lat_shape, fill, dtype=np.float32)
    else:
        core = core_view(residual, chunk, lead=1)
        mag = _fields.magnitude(core)
        block = _xp.get(_fields.pool_field(mag[None], f, target=lat_shape)[0])

    path = residual_path(cfg, manifest.level, entry.subject)
    if not Volume.exists(path, cfg.backend):
        ensure_residual_stores(cfg, manifest)
    Volume.open(path, cfg.backend).write_block(0, lat_origin, np.asarray(block, np.float32))


# --------------------------------------------------------------------------- #
# The slice sheet
# --------------------------------------------------------------------------- #
def _mid_slices(arr) -> list[np.ndarray]:
    """Axial, coronal and sagittal slices through the centre, as 2D arrays."""
    z, y, x = (int(n) // 2 for n in arr.shape[-3:])
    return [
        np.asarray(arr[z, :, :], dtype=np.float32),
        np.asarray(arr[:, y, :], dtype=np.float32),
        np.asarray(arr[:, :, x], dtype=np.float32),
    ]


def _grey(a: np.ndarray) -> np.ndarray:
    """Robustly windowed greyscale, ``(H, W, 3)`` uint8."""
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros(a.shape + (3,), dtype=np.uint8)
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if not hi > lo:
        lo, hi = float(finite.min()), float(finite.max()) + 1e-6
    g = np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
    g8 = (g * 255).astype(np.uint8)
    return np.dstack([g8, g8, g8])


_RAMP = np.array(
    [[0, 0, 0], [24, 40, 160], [0, 190, 220], [250, 220, 40], [255, 255, 255]],
    dtype=np.float32,
)
_NAN_RGB = np.array([255, 0, 255], dtype=np.uint8)


def _heat(a: np.ndarray, vmax: float) -> np.ndarray:
    """``0 .. vmax`` on a dark-to-light ramp; NaN in magenta, ``(H, W, 3)``."""
    t = np.clip(np.nan_to_num(a, nan=0.0) / max(float(vmax), 1e-9), 0.0, 1.0)
    pos = t * (len(_RAMP) - 1)
    i = np.clip(np.floor(pos).astype(np.int64), 0, len(_RAMP) - 2)
    w = (pos - i)[..., None]
    rgb = _RAMP[i] * (1 - w) + _RAMP[i + 1] * w
    out = rgb.astype(np.uint8)
    out[~np.isfinite(a)] = _NAN_RGB
    return out


def _fit(rgb: np.ndarray, box: int):
    from PIL import Image

    im = Image.fromarray(rgb)
    h, w = rgb.shape[:2]
    s = box / max(h, w, 1)
    return im.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.BILINEAR)


def render_pass(
    cfg: RunConfig,
    level: int,
    iteration: int,
    grid: GridSpec,
    d_max_mm: float | None,
    subjects=None,
) -> Path:
    """Write the pass's slice sheet and return its path.

    Template above, residual below; axial, coronal, sagittal across. The
    residual panel is the maximum over subjects, so a region any subject
    could not register shows, and its scale is fixed at the level's clamp so
    the same colour means the same millimetres from pass to pass. The panel
    label carries the actual 99th percentile of what is shown.
    """
    from PIL import Image, ImageDraw

    template = Volume.open(cfg.template_path(level), cfg.backend)
    t_slices = _mid_slices(template.array(0))

    subjects = list(subjects) if subjects is not None else list(cfg.registered_ids)
    r_slices: list[np.ndarray] | None = None
    for subject in subjects:
        path = residual_path(cfg, level, subject)
        if not Volume.exists(path, cfg.backend):
            continue
        s = _mid_slices(Volume.open(path, cfg.backend).array(0))
        r_slices = s if r_slices is None else [np.maximum(a, b) for a, b in zip(r_slices, s)]

    vmax = float(d_max_mm) if d_max_mm else 1.0
    if r_slices is not None:
        finite = np.concatenate([s[np.isfinite(s)] for s in r_slices]) if any(
            np.isfinite(s).any() for s in r_slices) else np.zeros(1)
        p99 = float(np.percentile(finite, 99.0)) if finite.size else 0.0
        if not d_max_mm:
            vmax = max(p99, 1e-6)
    else:
        p99 = float("nan")

    panels = [_fit(_grey(s), PANEL_PX) for s in t_slices]
    if r_slices is not None:
        panels += [_fit(_heat(s, vmax), PANEL_PX) for s in r_slices]
    else:
        panels += [Image.new("RGB", (PANEL_PX, PANEL_PX), (40, 40, 40))] * 3

    margin, gutter, label_h = 8, 6, 16
    col_w = max(p.width for p in panels)
    row_h = [max(p.height for p in panels[:3]), max(p.height for p in panels[3:])]
    width = margin * 2 + col_w * 3 + gutter * 2
    height = margin * 2 + sum(row_h) + label_h * 2 + gutter
    sheet = Image.new("RGB", (width, height), (18, 20, 24))
    draw = ImageDraw.Draw(sheet)

    spacing_um = grid.spacing_mm * 1000
    labels = [
        f"L{level} pass {iteration}  template @ {spacing_um:g} um  "
        f"(axial | coronal | sagittal)",
        (
            f"residual |u| mm, max over {len(subjects)} subject(s), "
            f"scale 0-{vmax:.2f}, p99 {p99:.2f}; magenta = fold ladder exhausted"
            if r_slices is not None
            else "residual: not written this pass"
        ),
    ]
    y = margin
    for row in range(2):
        draw.text((margin, y), labels[row], fill=(210, 210, 210))
        y += label_h
        x = margin
        for col in range(3):
            p = panels[row * 3 + col]
            sheet.paste(p, (x, y))
            x += col_w + gutter
        y += row_h[row] + gutter

    out = slices_path(cfg, level, iteration)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(str(out))
    return out

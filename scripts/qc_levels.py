"""One QC figure per finished level of a run.

Per level, three mid-planes (axial, coronal, sagittal) in five rows: the fixed
volume (template), the moving volume warped through the level's final field,
an overlay (fixed magenta, moving green: aligned tissue reads grey), the
displacement magnitude in mm, and the Jacobian determinant of the warp (folds
in magenta). The title carries the passes run, the last d99, the Dice of the
tissue masks on those planes and the fold fraction. Written to
``levels/L<k>/qc/level_<spacing>um.png`` with a JSON beside it.

    python scripts/qc_levels.py RUN_ROOT/config.json                 # every finished level
    python scripts/qc_levels.py RUN_ROOT/config.json --levels 0,1 --device cpu --out DIR

Reads the stores only; runs on a login node without a GPU (slower), or on one.
A level counts as finished when it has a template, a field and a pass record.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from chunkreg import fields as _fields
from chunkreg import xp as _xp
from chunkreg.config import load_config
from chunkreg.passes._common import warped_subject_block
from chunkreg.qc import PANEL_PX, _fit, _grey, _heat
from chunkreg.store import Field, Volume


def finished_levels(cfg) -> list[int]:
    out = []
    for p in sorted((cfg.root_path / "levels").glob("L*")):
        if not p.name[1:].isdigit():
            continue
        k = int(p.name[1:])
        if (
            Volume.exists(cfg.template_path(k), cfg.backend)
            and all(Field.exists(cfg.field_path(k, s), cfg.backend) for s in cfg.registered_ids)
            and cfg.pass_record_path(k, 0).exists()
        ):
            out.append(k)
    return out


def passes(cfg, level: int) -> list[dict]:
    recs = []
    for p in sorted(cfg.level_dir(level).glob("pass_it*.json"), key=lambda s: int(s.stem[7:])):
        try:
            recs.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            pass
    return recs


def _plane_box(shape, axis: int, index: int, thickness: int = 1):
    origin = [0, 0, 0]
    size = [int(n) for n in shape]
    origin[axis] = int(index)
    size[axis] = thickness
    return tuple(origin), tuple(size)


def level_figure(cfg, level: int, subject: str, out_dir: Path) -> tuple[Path, dict]:
    from PIL import Image, ImageDraw

    tmpl = Volume.open(cfg.template_path(level), cfg.backend)
    grid = tmpl.grid(0)
    spacing = grid.spacing_mm
    vol = Volume.open(cfg.subject_path(subject), cfg.backend)
    field = Field.open(cfg.field_path(level, subject), cfg.backend)
    um = spacing * 1000

    fixed, moving, umag, jac = [], [], [], []
    dice_num = dice_den = 0.0
    folds_num = folds_den = 0
    t0 = time.perf_counter()
    for axis in range(3):
        mid = grid.shape[axis] // 2
        # the warp needs its neighbours for the Jacobian: three planes, keep the middle
        o3, s3 = _plane_box(grid.shape, axis, max(mid - 1, 0), 3)
        u3 = field.read_dense(o3, s3)
        o1, s1 = _plane_box(grid.shape, axis, mid, 1)
        u1 = field.read_dense(o1, s1)
        f = _xp.get(tmpl.read_padded(0, o1, s1, normalise=True))
        m = _xp.get(warped_subject_block(vol, level, o1, s1, u1, spacing, normalise=True))
        f2, m2 = np.take(f, 0, axis=axis), np.take(m, 0, axis=axis)
        fixed.append(f2)
        moving.append(m2)
        umag.append(np.take(_xp.get(_fields.magnitude(u1)), 0, axis=axis))
        j3 = _xp.get(_fields.jacobian_determinant(u3, spacing))
        jac.append(np.take(j3, min(1, j3.shape[axis] - 1), axis=axis))
        fm, mm = f2 > 0.02, m2 > 0.02
        dice_num += 2.0 * float((fm & mm).sum())
        dice_den += float(fm.sum() + mm.sum())
        folds_num += int((jac[-1] <= 0).sum())
        folds_den += int(jac[-1].size)
    dice = dice_num / dice_den if dice_den else float("nan")
    fold_frac = folds_num / max(folds_den, 1)

    recs = passes(cfg, level)
    d99 = recs[-1].get("d99_mm") if recs else None
    umax = max(float(np.percentile(u, 99.5)) for u in umag) or 1.0

    def overlay(f, m):
        g = lambda a: (np.clip(a, 0, 1) * 255).astype(np.uint8)
        return np.dstack([g(f), g(m), g(f)])  # fixed magenta, moving green

    def jac_rgb(j):
        lj = np.log2(np.clip(j, 1e-3, 1e3))
        t = np.clip((lj + 1.0) / 2.0, 0, 1)  # 0.5x .. 2x -> 0 .. 1, 1x in the middle
        rgb = np.dstack([t, 1 - np.abs(t - 0.5) * 2, 1 - t]) * 255
        rgb = rgb.astype(np.uint8)
        rgb[j <= 0] = (255, 0, 255)
        return rgb

    rows = [
        ("fixed (template)", [_grey(s) for s in fixed]),
        (f"moving '{subject}' warped through the L{level} field", [_grey(s) for s in moving]),
        ("overlay: fixed magenta, moving green (aligned = grey)", [overlay(f, m) for f, m in zip(fixed, moving)]),
        (f"|u| mm, 0 to {umax:.2f}", [_heat(s, umax) for s in umag]),
        ("Jacobian determinant, blue 0.5x .. green 1x .. red 2x, magenta = fold", [jac_rgb(s) for s in jac]),
    ]
    panels = [[_fit(img, PANEL_PX) for img in imgs] for _, imgs in rows]
    margin, gutter, label_h = 8, 6, 16
    col_w = max(p.width for ps in panels for p in ps)
    row_h = [max(p.height for p in ps) for ps in panels]
    width = margin * 2 + col_w * 3 + gutter * 2
    height = margin * 2 + sum(row_h) + label_h * (len(rows) + 1) + gutter * len(rows)
    sheet = Image.new("RGB", (width, height), (18, 20, 24))
    draw = ImageDraw.Draw(sheet)
    title = (
        f"L{level}  {um:g} um  {'x'.join(map(str, grid.shape))}   passes {len(recs)}"
        + (f"   last d99 {d99:.3f} mm" if d99 is not None else "")
        + f"   Dice(mid-planes) {dice:.3f}   folds {fold_frac:.2%}"
    )
    y = margin
    draw.text((margin, y), title, fill=(230, 230, 230))
    y += label_h
    for (label, _), ps, h in zip(rows, panels, row_h):
        draw.text((margin, y), label + "   (axial | coronal | sagittal)", fill=(200, 200, 200))
        y += label_h
        x = margin
        for p in ps:
            sheet.paste(p, (x, y))
            x += col_w + gutter
        y += h + gutter
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"level_{um:g}um.png"
    sheet.save(str(png))
    stats = {
        "level": level, "spacing_um": um, "shape": list(grid.shape), "passes": len(recs),
        "last_d99_mm": d99, "dice_mid_planes": dice, "fold_fraction_mid_planes": fold_frac,
        "u_p99.5_mm": umax, "seconds": time.perf_counter() - t0,
        "pass_d99_mm": [r.get("d99_mm") for r in recs],
    }
    (out_dir / f"level_{um:g}um.json").write_text(json.dumps(stats, indent=1))
    return png, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config")
    ap.add_argument("--levels", default=None, help="comma-separated level indices (default: every finished level)")
    ap.add_argument("--subject", default=None, help="moving subject id (default: the first registered one)")
    ap.add_argument("--out", default=None, help="directory for the figures (default: each level's qc/ folder)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force", action="store_true", help="redo a level whose figure exists")
    args = ap.parse_args()
    cfg = load_config(args.config)
    print(f"device {_xp.configure(args.device)}")
    subject = args.subject or cfg.registered_ids[0]
    levels = finished_levels(cfg) if args.levels is None else [int(v) for v in args.levels.split(",")]
    if not levels:
        print("no finished level yet")
        return 0
    for k in levels:
        out_dir = Path(args.out) if args.out else cfg.level_dir(k) / "qc"
        um = Volume.open(cfg.template_path(k), cfg.backend).grid(0).spacing_mm * 1000
        if not args.force and (out_dir / f"level_{um:g}um.png").exists():
            print(f"L{k}: figure exists, skipping (--force to redo)")
            continue
        png, st = level_figure(cfg, k, subject, out_dir)
        print(f"L{k} {st['spacing_um']:g} um: passes {st['passes']}, d99 {st['last_d99_mm']}, "
              f"Dice {st['dice_mid_planes']:.3f}, folds {st['fold_fraction_mid_planes']:.2%} -> {png} "
              f"({st['seconds']:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

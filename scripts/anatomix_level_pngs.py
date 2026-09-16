#!/usr/bin/env python
"""Anatomix feature maps from one chunk at every pyramid level of one subject.

For each level of a subject's store, take the registration chunk (core plus
halo, exactly the box the register pass reads) whose core contains one chosen
anatomical point, run the run's feature extractor on it, and write PNGs through
that point:

    L<k>_<um>um_intensity.png          axial | coronal | sagittal, chunk core
                                       outlined in yellow, the point in red
    L<k>_<um>um_features_rgb.png       the same planes, the feature channels
                                       projected to RGB by a PCA of this chunk
    L<k>_<um>um_channels_<plane>.png   every channel on its own, one tile each
    overview.png                       one row per level: intensity, then RGB
    summary.json                       per level: grid, chunk, timings, stats

Extraction and the RGB projection run on the configured device (the GPU for a
``"device": "cuda"`` config). Only 2D planes and a few numbers leave it.

The point defaults to the tissue centroid of the coarsest level, so every
level shows the same anatomy even when the scan is off centre. The subject has
to be ingested already (``chunkreg setup``).

    python scripts/anatomix_level_pngs.py CONFIG [--subject ID] [--out DIR]
        [--at Z,Y,X] [--levels 0,1,2] [--features anatomix] [--tile-px 256]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

PLANES = ("axial", "coronal", "sagittal")  # normal to z, y and x
TISSUE = 0.05  # normalised intensity above which a voxel counts as tissue


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", help="run config (JSON or YAML)")
    ap.add_argument("--subject", default=None, help="subject id (default: the first)")
    ap.add_argument(
        "--out", default=None,
        help="output folder (default: <root>/qc/anatomix_levels/<subject>)",
    )
    ap.add_argument(
        "--at", default=None,
        help="point in world mm as Z,Y,X (default: tissue centroid at level 0)",
    )
    ap.add_argument("--levels", default=None, help="comma-separated levels (default: all)")
    ap.add_argument(
        "--features", default=None,
        help="extractor to use (default: the config profile's, e.g. anatomix)",
    )
    ap.add_argument("--device", default=None, help="override the config's device")
    ap.add_argument("--tile-px", type=int, default=256, help="plane height in overview.png")
    ap.add_argument("--samples", type=int, default=200_000, help="voxels for the PCA fit")
    ap.add_argument(
        "--repeat", type=int, default=2,
        help="extractions per chunk; the last is timed, so the first can "
             "absorb one-off GPU kernel selection (default 2)",
    )
    return ap.parse_args(argv)


# --------------------------------------------------------------------------- #
# Device-side helpers (all inputs are torch tensors on the compute device)
# --------------------------------------------------------------------------- #
def as_tensor(a, torch, device):
    if isinstance(a, torch.Tensor):
        return a.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)


def quantiles(x, qs, torch):
    """Per-row quantiles of a (R, N) tensor, by order statistic."""
    n = x.shape[1]
    out = []
    for q in qs:
        k = min(max(int(round(q * (n - 1))) + 1, 1), n)
        out.append(torch.kthvalue(x, k, dim=1).values)
    return out


def fit_projection(feats, block, n_samples, torch, seed=0):
    """PCA of the chunk's feature vectors, fitted on the device.

    Samples tissue voxels when there are enough of them, so the colours spend
    their range on anatomy rather than on the air around it. Returns the RGB
    basis, the mean, per-component display ranges, per-channel display ranges
    and the share of variance the three components carry.
    """
    c = feats.shape[0]
    flat = feats.reshape(c, -1)
    tissue = torch.nonzero(block.reshape(-1) > TISSUE).squeeze(1)
    pool = tissue if tissue.numel() >= 1000 else None
    total = flat.shape[1] if pool is None else pool.numel()
    gen = torch.Generator(device=feats.device).manual_seed(seed)
    pick = torch.randint(total, (min(n_samples, total),), generator=gen, device=feats.device)
    if pool is not None:
        pick = pool[pick]
    x = flat[:, pick].double()
    mean = x.mean(dim=1, keepdim=True)
    xc = x - mean
    cov = xc @ xc.T / max(x.shape[1] - 1, 1)
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order].clamp_min(0), evecs[:, order]
    k = min(3, c)
    basis = evecs[:, :k].T.contiguous()
    # Fix each component's sign so the same anatomy gets the same colour when
    # the script is rerun: the largest loading is made positive.
    peak = basis.abs().argmax(dim=1)
    basis = basis * torch.sign(basis[torch.arange(k, device=basis.device), peak])[:, None]
    proj = basis @ xc
    lo, hi = quantiles(proj, (0.01, 0.99), torch)
    ch_lo, ch_hi = quantiles(x, (0.01, 0.99), torch)
    explained = float(evals[:k].sum() / evals.sum().clamp_min(1e-30))
    return {
        "basis": basis.float(),
        "mean": mean.float(),
        "lo": lo.float(),
        "hi": hi.float(),
        "ch_lo": ch_lo.float(),
        "ch_hi": ch_hi.float(),
        "explained": explained,
        "k": k,
    }


def to_uint8(x, torch):
    return (x.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)


def plane_of(t, plane: str, local, lead: int):
    """The slice of ``t`` through ``local`` normal to one axis."""
    axis = PLANES.index(plane)
    index = [slice(None)] * t.ndim
    index[lead + axis] = int(local[axis])
    return t[tuple(index)]


def rgb_plane(feat_plane, fit, torch):
    """(C, H, W) features -> (H, W, 3) uint8 on the host."""
    c, h, w = feat_plane.shape
    proj = fit["basis"] @ (feat_plane.reshape(c, -1) - fit["mean"])
    span = (fit["hi"] - fit["lo"]).clamp_min(1e-12)
    rgb = ((proj - fit["lo"][:, None]) / span[:, None]).reshape(fit["k"], h, w)
    if fit["k"] < 3:
        rgb = torch.cat([rgb, rgb[-1:].expand(3 - fit["k"], h, w)], dim=0)
    return np.ascontiguousarray(to_uint8(rgb, torch).permute(1, 2, 0).cpu().numpy())


def channel_grid(feat_plane, fit, torch, gap: int = 2):
    """Every channel of a (C, H, W) plane as a grey tile, laid out in a grid."""
    c, h, w = feat_plane.shape
    span = (fit["ch_hi"] - fit["ch_lo"]).clamp_min(1e-12)
    grey = to_uint8((feat_plane - fit["ch_lo"][:, None, None]) / span[:, None, None], torch)
    cols = int(math.ceil(math.sqrt(c)))
    rows = int(math.ceil(c / cols))
    sheet = torch.zeros(
        (rows * (h + gap) - gap, cols * (w + gap) - gap),
        dtype=torch.uint8,
        device=feat_plane.device,
    )
    for i in range(c):
        r, q = divmod(i, cols)
        sheet[r * (h + gap) : r * (h + gap) + h, q * (w + gap) : q * (w + gap) + w] = grey[i]
    g = sheet.cpu().numpy()
    return np.dstack([g, g, g])


def intensity_plane(block_plane, plane, core_lo, core_hi, point, torch):
    """Grey plane with the chunk core outlined and the point marked."""
    g = to_uint8(block_plane, torch).cpu().numpy()
    img = np.dstack([g, g, g])
    axes = [d for d in range(3) if d != PLANES.index(plane)]
    r0, r1 = core_lo[axes[0]], core_hi[axes[0]] - 1
    c0, c1 = core_lo[axes[1]], core_hi[axes[1]] - 1
    yellow = np.array([255, 220, 0], np.uint8)
    img[r0, c0 : c1 + 1] = yellow
    img[r1, c0 : c1 + 1] = yellow
    img[r0 : r1 + 1, c0] = yellow
    img[r0 : r1 + 1, c1] = yellow
    pr, pc = point[axes[0]], point[axes[1]]
    img[max(pr - 1, 0) : pr + 2, max(pc - 1, 0) : pc + 2] = np.array([255, 0, 0], np.uint8)
    return img


# --------------------------------------------------------------------------- #
# Host-side layout (arranging finished uint8 images, no computation)
# --------------------------------------------------------------------------- #
def strip(images, gap: int = 4):
    height = max(im.shape[0] for im in images)
    parts = []
    for i, im in enumerate(images):
        pad = np.zeros((height - im.shape[0], im.shape[1], 3), np.uint8)
        parts.append(np.vstack([im, pad]))
        if i + 1 < len(images):
            parts.append(np.zeros((height, gap, 3), np.uint8))
    return np.hstack(parts)


def to_height(img, height: int):
    """Nearest-neighbour rescale to a height, keeping the aspect ratio."""
    h, w = img.shape[:2]
    width = max(1, int(round(w * height / h)))
    rows = (np.arange(height) * h // height).clip(0, h - 1)
    cols = (np.arange(width) * w // width).clip(0, w - 1)
    return img[rows][:, cols]


def label(img, text):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return img
    pil = Image.fromarray(img)
    ImageDraw.Draw(pil).text((4, 2), text, fill=(255, 255, 255))
    return np.asarray(pil)


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = parse_args(argv)
    import torch

    from chunkreg import xp
    from chunkreg.config import load_config
    from chunkreg.featureviz import save_png
    from chunkreg.features import get_extractor
    from chunkreg.grid import tile
    from chunkreg.store import Volume

    cfg = load_config(args.config)
    device = xp.configure(args.device or cfg.device)
    dev = xp.torch_device()
    on_gpu = xp.on_gpu()

    subject = args.subject or cfg.subject_ids[0]
    path = cfg.subject_path(subject)
    if not Volume.exists(path, cfg.backend):
        sys.exit(
            f"no store for subject {subject!r} at {path}; run "
            f"'chunkreg setup {args.config}' first"
        )
    vol = Volume.open(path, cfg.backend)
    if vol.normalisation is None:
        sys.exit(f"{path} has no intensity window; re-ingest it with 'chunkreg setup'")
    grids = vol.grids()
    levels = (
        [int(v) for v in args.levels.replace(",", " ").split()]
        if args.levels
        else list(range(len(grids)))
    )
    bad = [k for k in levels if not 0 <= k < len(grids)]
    if bad:
        sys.exit(f"levels {bad} do not exist; this store has levels 0..{len(grids) - 1}")

    features = args.features or cfg.profile.features
    out = Path(args.out) if args.out else cfg.root_path / "qc" / "anatomix_levels" / subject
    out.mkdir(parents=True, exist_ok=True)

    print(f"subject {subject}: {path}")
    print(f"device {device}, features {features}, {len(grids)} levels, writing {out}")
    t0 = time.perf_counter()
    extractor = get_extractor(features)
    extractor.setup()
    print(f"extractor ready in {time.perf_counter() - t0:.1f} s")

    # The point: given, or the tissue centroid of the coarsest level.
    if args.at:
        centre = np.asarray([float(v) for v in args.at.replace(",", " ").split()])
        if centre.size != 3:
            sys.exit(f"--at needs three world coordinates, got {args.at!r}")
    else:
        g0 = grids[0]
        top = as_tensor(vol.read_padded(0, (0, 0, 0), g0.shape, normalise=True), torch, dev)
        w = (top > TISSUE).to(torch.float32)
        if float(w.sum()) < 1:
            w = torch.ones_like(top)
        idx = [torch.arange(n, device=dev, dtype=torch.float32) for n in g0.shape]
        vox0 = [
            float((w.sum(dim=tuple(a for a in range(3) if a != d)) * idx[d]).sum() / w.sum())
            for d in range(3)
        ]
        centre = g0.world(np.asarray(vox0))
        del top, w
    centre = tuple(float(v) for v in centre)
    print(f"point {', '.join(f'{v:.2f}' for v in centre)} mm")

    rows = []
    summary = {
        "subject": subject,
        "store": str(path),
        "config": str(args.config),
        "device": device,
        "features": features,
        "point_mm": list(centre),
        "levels": [],
    }
    header = f"{'lvl':>3} {'spacing':>9} {'chunk':>14} {'pad shape':>16} {'read s':>7} {'feat s':>7} {'rgb var':>8} {'peak GB':>8}"
    print(header)
    print("-" * len(header))

    for k in levels:
        grid = grids[k]
        um = grid.spacing_mm * 1000.0
        vox = np.clip(np.rint(grid.voxel(np.asarray(centre))).astype(np.int64), 0,
                      np.asarray(grid.shape) - 1)
        chunk = next(
            c for c in tile(grid, cfg.profile)
            if all(c.core_origin[d] <= vox[d] < c.core_origin[d] + c.core_shape[d]
                   for d in range(3))
        )
        if on_gpu:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        t_read = time.perf_counter()
        block = as_tensor(vol.read_chunk(k, chunk, normalise=True), torch, dev)
        if on_gpu:
            torch.cuda.synchronize()
        t_feat = time.perf_counter()
        times = []
        for _ in range(max(1, args.repeat)):
            ts = time.perf_counter()
            feats = as_tensor(extractor(block, grid.spacing_mm), torch, dev)
            if on_gpu:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - ts)

        fit = fit_projection(feats, block, args.samples, torch)
        local = [int(vox[d] - chunk.pad_origin[d]) for d in range(3)]
        core_lo = [int(o) for o in chunk.core_offset_in_pad]
        core_hi = [core_lo[d] + int(chunk.core_shape[d]) for d in range(3)]

        tag = f"L{k}_{um:g}um"
        grey, colour = [], []
        for plane in PLANES:
            gi = intensity_plane(
                plane_of(block, plane, local, 0), plane, core_lo, core_hi, local, torch
            )
            fp = plane_of(feats, plane, local, 1)
            ci = rgb_plane(fp, fit, torch)
            save_png(out / f"{tag}_channels_{plane}.png", channel_grid(fp, fit, torch))
            grey.append(gi)
            colour.append(ci)
        save_png(out / f"{tag}_intensity.png", strip(grey))
        save_png(out / f"{tag}_features_rgb.png", strip(colour))

        row = strip([to_height(im, args.tile_px) for im in grey + colour])
        rows.append(label(row, f"L{k}  {um:g} um  chunk {chunk.index}  {features}"))

        peak = torch.cuda.max_memory_allocated() / 1024**3 if on_gpu else 0.0
        info = {
            "level": k,
            "spacing_um": um,
            "grid_shape": list(grid.shape),
            "chunk_index": list(chunk.index),
            "core_origin": list(chunk.core_origin),
            "pad_origin": list(chunk.pad_origin),
            "pad_shape": list(chunk.pad_shape),
            "point_voxel": [int(v) for v in vox],
            "feature_shape": list(feats.shape),
            "read_seconds": round(t_feat - t_read, 3),
            "feature_seconds": round(times[-1], 3),
            "feature_seconds_first": round(times[0], 3),
            "tissue_fraction": round(float((block > TISSUE).float().mean()), 4),
            "rgb_explained": round(fit["explained"], 4),
            "channel_std": [round(float(v), 5) for v in feats.reshape(feats.shape[0], -1).std(dim=1)],
            "peak_gpu_gb": round(peak, 2),
        }
        summary["levels"].append(info)
        print(
            f"{k:>3} {um:>6g} um {str(chunk.index):>14} {str(tuple(chunk.pad_shape)):>16} "
            f"{info['read_seconds']:>7.2f} {info['feature_seconds']:>7.2f} "
            f"{fit['explained']:>7.0%} {peak:>8.1f}"
        )
        del block, feats, fit
        xp.release()

    width = max(r.shape[1] for r in rows)
    sheet = np.vstack(
        [np.hstack([r, np.zeros((r.shape[0], width - r.shape[1], 3), np.uint8)]) for r in rows]
    )
    save_png(out / "overview.png", sheet)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {len(list(out.glob('*.png')))} PNGs and summary.json to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

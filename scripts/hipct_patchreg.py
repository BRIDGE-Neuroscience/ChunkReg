#!/usr/bin/env python
"""
hipct_patchreg.py -- GPU patchwise diffeomorphic registration to an unbiased
average template for HiP-CT brain volumes, in ONE file.

The 50 um build and the 20 um refinement are the *same* process
(``build_template``) with slightly different parameters. The fine stage simply
seeds itself from the coarse solution (upsampled template + upsampled warps)
and then runs the identical loop with a tighter halo and fewer iterations.

CALLS
-----
Every tunable is a CLI flag (run `coarse --help` / `fine --help` for the full
list). The defaults suit the COARSE (50 um) build; for FINE (20 um) pass the
tighter values shown below. Flag defaults come from the DEF_* constants further
down, each annotated with its recommended [fine] value.

    # 0. verify the FireANTs warp convention on THIS install (do this first!)
    python hipct_patchreg.py selftest

    # 1. 50 um unbiased template  (subjects must already be on the common grid)
    python hipct_patchreg.py coarse \
        --subjects s01=/data/grid50/s01.zarr s02=/data/grid50/s02.zarr ... \
        --shape 2560,2560,1920 --spacing 0.05 \
        --core-mm 3.0 --halo-mm 1.0 --scales 4,2,1 --iterations 100,60,30 \
        --max-disp-mm 0.9 --post-blend-sigma-mm 0.05 --template-iters 4 \
        --work-dir ./work_50um --num-gpus 8
    # (those reg/patch/template flags are also the defaults, so for a coarse run
    #  you can drop them and just pass --subjects/--shape/--spacing/--work-dir.)

    # 2. 20 um seeded refinement  (same scans, resampled onto the 20 um grid)
    python hipct_patchreg.py fine \
        --subjects s01=/data/grid20/s01.zarr s02=/data/grid20/s02.zarr ... \
        --shape 2560,2560,1920 --spacing 0.05 --fine-spacing 0.02 \
        --core-mm 2.0 --halo-mm 0.5 --scales 2,1 --iterations 60,40 \
        --max-disp-mm 0.2 --post-blend-sigma-mm 0.02 --template-iters 2 \
        --seed-dir ./work_50um --seed-template-iters 4 \
        --work-dir ./work_20um --num-gpus 8

Anything not passed on the command line falls back to the DEF_* defaults (or, for
subjects/grid, the CONFIG block below), so a coarse run can be as short as
`python hipct_patchreg.py coarse --work-dir ./work_50um` once CONFIG is filled in.

IMPORTANT -- patchwise vs. full-context
---------------------------------------
The FireANTs gigavoxel paper (Jena et al., FFDP, ICLR 2026, arXiv 2509.25044)
reports that naive patchwise registration *degrades* at high resolution because
patches lose anatomical context. Their recommended alternative is full-context
distributed registration (GridParallel + Ring Sampler) with halo exchange across
GPUs. If a volume fits across the node's combined memory, prefer that. This
script takes the patchwise route on purpose: it depends only on the stable
single-GPU FireANTs API, is embarrassingly parallel, scales to arbitrarily large
volumes on a fixed GPU count (the 20 um case), and mitigates context loss via
generous halos, multiscale-within-patch, Sobolev smoothing, and partition-of-
unity blending. It is an informed tradeoff, not a free lunch.

CONVENTION (used everywhere)
----------------------------
Every volume is resampled onto a common, isotropic, axis-aligned grid first, so
world = origin + voxel * spacing (diagonal affine, no rotation). A displacement
field u is stored in MILLIMETRES on that grid; to pull image M (already on the
grid) through u, sample at voxel coords  x + u / spacing  (see ``apply_warp``).
Because u is physical, upsampling 50->20 um leaves the vectors unchanged and only
densifies the lattice -- which is exactly what makes the fine-stage seeding exact.

Two version-sensitive FireANTs spots are marked  # >>> API  below; the
``selftest`` subcommand pins them down with a known 5-voxel shift.

Deps: numpy, scipy, torch, zarr; SimpleITK + nibabel (image construction); and
fireants (the registration engine; build its fused_ops for the fast kernels).
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Iterable, Iterator, Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp


# =========================================================================== #
# CONFIG  -- defaults used when a CLI flag is omitted. Edit freely.
# =========================================================================== #
SUBJECTS_DEFAULT: dict[str, str] = {
    # subject_id -> path to a volume ALREADY resampled onto the common grid (zarr).
    # Use your own GPU tiled resampler with the 500 um MNI affine so every
    # subject shares world space with the template.
    "s01": "/data/grid/s01.zarr",
    "s02": "/data/grid/s02.zarr",
    "s03": "/data/grid/s03.zarr",
    "s04": "/data/grid/s04.zarr",
    "s05": "/data/grid/s05.zarr",
}
GRID_SHAPE_DEFAULT = (2560, 2560, 1920)     # (X, Y, Z) of the COARSE grid
GRID_ORIGIN_DEFAULT = (0.0, 0.0, 0.0)       # world coord (mm) of voxel (0,0,0)

# --------------------------------------------------------------------------- #
# Default values for every tunable. EACH IS A CLI FLAG (see --help); the values
# here are just the defaults if you omit the flag, and they're set for the
# COARSE (50 um) build. Recommended FINE (20 um) values are noted as [fine: ...].
# The two stages are otherwise the same process -- only these numbers differ.
# --------------------------------------------------------------------------- #
DEF_SPACING_MM      = 0.05      # coarse grid voxel size (mm); fine grid uses --fine-spacing
DEF_FINE_SPACING_MM = 0.02      # fine grid voxel size (mm), derived as the same world box

# patch tiling
DEF_CORE_MM      = 5.0          # [fine: 2.0]  cube of anatomy each patch "owns" (mm)
DEF_HALO_MM      = 1.0          # [fine: 0.5]  context margin per side; MUST exceed the
                                #              largest expected displacement + LNCC radius
# multiscale registration schedule (within each patch)
DEF_SCALES       = "4,2,1"      # [fine: 2,1]  in-patch downsample factors, coarse->fine.
                                #              Coarse needs the aggressive top level for
                                #              big deformation + global context; fine only
                                #              refines a small residual, so drop it.
DEF_ITERATIONS   = "100,60,30"  # [fine: 60,40]  iterations per scale (len must match --scales)
DEF_MAX_DISP_MM  = 0.9          # [fine: 0.2]  hard clamp on |displacement| per patch (mm).
                                #              Keep it < halo_mm - cc_kernel*spacing/2 so a
                                #              moved feature keeps valid LNCC context.
                                #              Set to a negative value to disable the clamp.
DEF_POST_BLEND   = 0.05         # [fine: 0.02] light Gaussian smoothing of the BLENDED field
                                #              (mm) to erase any residual inter-patch mismatch;
                                #              0 to disable.
# unbiased-template construction
DEF_TEMPLATE_ITERS = 4          # [fine: 2]    outer iterations. Coarse builds the shape from
                                #              scratch (3-5); fine inherits it, so 1-2 is plenty.
DEF_SHAPE_UPDATE   = 0.25       # step for the mean-shape recenter (the "unbiased" update)
# shared registration recipe (identical at both scales unless you override)
DEF_TRANSFORM    = "greedy"     # "greedy" (diffeomorphic greedy) or "syn"
DEF_LOSS         = "cc"         # "cc" == fused LNCC (recommended); also "mi", "mse"
DEF_CC_KERNEL    = 7            # LNCC window
DEF_LR           = 0.5          # Adam learning rate
DEF_SMOOTH_GRAD  = 1.0          # Sobolev sigma on the update (velocity)
DEF_SMOOTH_WARP  = 0.5          # Sobolev sigma on the total displacement
# =========================================================================== #


# --------------------------------------------------------------------------- #
# Config objects
# --------------------------------------------------------------------------- #
@dataclass
class GridSpec:
    shape: tuple[int, int, int]
    spacing_mm: float
    origin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def extent_mm(self) -> tuple[float, float, float]:
        return tuple(s * self.spacing_mm for s in self.shape)

    def rescaled(self, new_spacing_mm: float) -> "GridSpec":
        new_shape = tuple(int(round(e / new_spacing_mm)) for e in self.extent_mm)
        return GridSpec(new_shape, new_spacing_mm, self.origin_mm)


@dataclass
class PatchConfig:
    core_mm: tuple[float, float, float] = (3.0, 3.0, 3.0)
    halo_mm: float = 1.0
    taper_mm: Optional[float] = None
    window: Literal["hann", "trapezoid"] = "hann"

    def core_vox(self, s):  return tuple(max(1, int(round(c / s))) for c in self.core_mm)
    def halo_vox(self, s):  return max(1, int(round(self.halo_mm / s)))
    def taper_vox(self, s):
        t = self.taper_mm if self.taper_mm is not None else self.halo_mm
        return max(1, int(round(t / s)))


@dataclass
class RegConfig:
    transform: Literal["greedy", "syn"] = "greedy"
    loss_type: Literal["cc", "mi", "mse"] = "cc"
    cc_kernel: int = 7
    scales: Sequence[int] = (4, 2, 1)
    iterations: Sequence[int] = (100, 60, 30)
    optimizer: str = "Adam"
    optimizer_lr: float = 0.5
    smooth_grad_sigma: float = 1.0
    smooth_warp_sigma: float = 0.5
    max_disp_mm: Optional[float] = None
    post_blend_sigma_mm: float = 0.0


@dataclass
class TemplateConfig:
    n_iterations: int = 4
    shape_update_step: float = 0.25
    unbias: bool = True


# --------------------------------------------------------------------------- #
# Patch tiling with halo + partition-of-unity (PoU) windows
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Patch:
    core_origin: tuple[int, int, int]
    core_shape: tuple[int, int, int]
    pad_origin: tuple[int, int, int]
    pad_shape: tuple[int, int, int]

    @property
    def core_offset_in_pad(self) -> tuple[int, int, int]:
        return tuple(c - p for c, p in zip(self.core_origin, self.pad_origin))


def build_patches(grid: GridSpec, pc: PatchConfig) -> list[Patch]:
    shape = np.asarray(grid.shape, int)
    core = np.asarray(pc.core_vox(grid.spacing_mm), int)
    halo = pc.halo_vox(grid.spacing_mm)
    starts = [np.arange(0, shape[d], core[d]) for d in range(3)]
    out = []
    for cx, cy, cz in product(*starts):
        co = np.array([cx, cy, cz], int)
        cs = np.minimum(core, shape - co)
        po = np.clip(co - halo, 0, None)
        ph = np.minimum(co + cs + halo, shape) - po
        out.append(Patch(tuple(co), tuple(cs), tuple(po), tuple(ph)))
    return out


def _axis_window(pad_len, core_lo, core_len, taper, at_low, at_high, kind):
    w = np.ones(pad_len, np.float64)
    core_hi = core_lo + core_len

    def ramp(n):
        t = (np.arange(n) + 0.5) / n
        return 0.5 - 0.5 * np.cos(np.pi * t) if kind == "hann" else t

    if not at_low:
        n = min(taper, core_lo)
        if n > 0:
            w[core_lo - n:core_lo] = ramp(n); w[:core_lo - n] = 0.0
        else:
            w[:core_lo] = 0.0
    if not at_high:
        tail = pad_len - core_hi
        n = min(taper, tail)
        if n > 0:
            w[core_hi:core_hi + n] = ramp(n)[::-1]; w[core_hi + n:] = 0.0
        else:
            w[core_hi:] = 0.0
    return w


def blend_window(p: Patch, grid: GridSpec, pc: PatchConfig) -> np.ndarray:
    taper = pc.taper_vox(grid.spacing_mm)
    off = p.core_offset_in_pad
    axes = []
    for d in range(3):
        at_low = p.pad_origin[d] == 0 and p.core_origin[d] == 0
        at_high = (p.pad_origin[d] + p.pad_shape[d]) >= grid.shape[d] and \
                  (p.core_origin[d] + p.core_shape[d]) >= grid.shape[d]
        axes.append(_axis_window(p.pad_shape[d], off[d], p.core_shape[d],
                                 taper, at_low, at_high, pc.window))
    wx, wy, wz = axes
    w = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
    return np.maximum(w.astype(np.float32), 1e-6)


# --------------------------------------------------------------------------- #
# Warp-field ops (GPU, mm-on-grid convention)
# --------------------------------------------------------------------------- #
def _normalised_grid(coords_vox: torch.Tensor, shape) -> torch.Tensor:
    X, Y, Z = shape
    x, y, z = coords_vox.unbind(-1)
    gx = 2.0 * x / max(X - 1, 1) - 1.0
    gy = 2.0 * y / max(Y - 1, 1) - 1.0
    gz = 2.0 * z / max(Z - 1, 1) - 1.0
    return torch.stack([gz, gy, gx], dim=-1)        # grid_sample wants (z,y,x)


def identity_voxel_grid(shape, device) -> torch.Tensor:
    X, Y, Z = shape
    ax = torch.arange(X, device=device, dtype=torch.float32)
    ay = torch.arange(Y, device=device, dtype=torch.float32)
    az = torch.arange(Z, device=device, dtype=torch.float32)
    gx, gy, gz = torch.meshgrid(ax, ay, az, indexing="ij")
    return torch.stack([gx, gy, gz], dim=0)


def apply_warp(moving, u_mm, spacing_mm, mode="bilinear"):
    shape = tuple(moving.shape)
    sample = (identity_voxel_grid(shape, moving.device) + u_mm / spacing_mm).permute(1, 2, 3, 0)
    grid = _normalised_grid(sample, shape)[None]
    return F.grid_sample(moving[None, None], grid, mode=mode,
                         padding_mode="zeros", align_corners=True)[0, 0]


def resample_field_to_grid(u_mm, src: GridSpec, dst: GridSpec):
    dev = u_mm.device
    ident = identity_voxel_grid(dst.shape, dev)
    world = ident * dst.spacing_mm + torch.tensor(dst.origin_mm, device=dev).view(3, 1, 1, 1)
    src_org = torch.tensor(src.origin_mm, device=dev).view(3, 1, 1, 1)
    src_vox = ((world - src_org) / src.spacing_mm).permute(1, 2, 3, 0)
    grid = _normalised_grid(src_vox, src.shape)[None]
    return F.grid_sample(u_mm[None], grid, mode="bilinear",
                         padding_mode="border", align_corners=True)[0]


def compose(u_outer, u_inner, spacing_mm):
    shape = tuple(u_inner.shape[1:])
    ident = identity_voxel_grid(shape, u_inner.device)
    sample = (ident + u_outer / spacing_mm).permute(1, 2, 3, 0)
    grid = _normalised_grid(sample, shape)[None]
    u_inner_at = F.grid_sample(u_inner[None], grid, mode="bilinear",
                               padding_mode="border", align_corners=True)[0]
    return u_outer + u_inner_at


def to_numpy(u_mm) -> np.ndarray:
    return u_mm.detach().to("cpu", torch.float32).numpy()


# --------------------------------------------------------------------------- #
# On-disk accumulator on the template grid (zarr-backed; never all in RAM)
# --------------------------------------------------------------------------- #
@dataclass
class GridArray:
    data: object                       # (C, X, Y, Z) zarr.Array or ndarray
    grid: GridSpec

    @classmethod
    def zeros(cls, grid: GridSpec, channels: int, path: Optional[str] = None,
              chunk: int = 128) -> "GridArray":
        shape = (channels, *grid.shape)
        if path is None:
            return cls(np.zeros(shape, np.float32), grid)
        import zarr
        z = zarr.open(path, mode="w", shape=shape, dtype="f4",
                      chunks=(channels, chunk, chunk, chunk))
        return cls(z, grid)

    def add_block(self, origin, block: np.ndarray) -> None:
        o = np.asarray(origin, int); s = np.asarray(block.shape[1:], int)
        lo = np.clip(o, 0, None); hi = np.minimum(o + s, np.asarray(self.grid.shape, int))
        if np.any(hi <= lo):
            return
        b_lo = lo - o; b_hi = b_lo + (hi - lo)
        sub = block[:, b_lo[0]:b_hi[0], b_lo[1]:b_hi[1], b_lo[2]:b_hi[2]]
        cur = self.data[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        self.data[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = np.asarray(cur) + sub

    def read_block(self, origin, shape) -> np.ndarray:
        o = np.asarray(origin, int); s = np.asarray(shape, int)
        lo = np.clip(o, 0, None); hi = np.minimum(o + s, np.asarray(self.grid.shape, int))
        out = np.zeros((self.data.shape[0], *s), np.float32)
        if np.any(hi <= lo):
            return out
        sub = np.asarray(self.data[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        d = lo - o
        out[:, d[0]:d[0]+sub.shape[1], d[1]:d[1]+sub.shape[2], d[2]:d[2]+sub.shape[3]] = sub
        return out


# --------------------------------------------------------------------------- #
# FireANTs backend  (the only library-version-sensitive code)
# --------------------------------------------------------------------------- #
try:
    from fireants.io.image import Image, BatchedImages
    from fireants.registration.greedy import GreedyRegistration
    try:
        from fireants.registration.syn import SyNRegistration
    except Exception:
        SyNRegistration = None
    _HAVE_FIREANTS = True
except Exception:
    _HAVE_FIREANTS = False


def _require_fireants():
    if not _HAVE_FIREANTS:
        raise ImportError(
            "FireANTs is not importable. Install it (github.com/rohitrango/fireants) "
            "and build fused_ops for the fast LNCC / grid-sampler kernels.")


def _image_from_array(arr: np.ndarray, spacing_mm: float, device: str):
    """# >>> API  Build a FireANTs Image from a (X,Y,Z) array via SimpleITK.

    Patches live on an isotropic axis-aligned grid, so spacing is (s,s,s) and
    direction is identity. If your build exposes ``Image.from_numpy(...)`` you
    can swap that in here.
    """
    _require_fireants()
    import SimpleITK as sitk
    itk = sitk.GetImageFromArray(np.ascontiguousarray(arr.transpose(2, 1, 0)))  # ITK z,y,x
    itk.SetSpacing((float(spacing_mm),) * 3)
    if "device" in Image.__init__.__code__.co_varnames:
        return Image(itk, device=device)
    return Image(itk)


def _displacement_mm_from_reg(reg, shape, spacing_mm, device) -> torch.Tensor:
    """# >>> API  Recover the dense displacement (3,X,Y,Z) in mm on the patch grid.

    Asks FireANTs for the warped sampling coordinates phi, converts to voxel
    coords, subtracts identity, scales by spacing. Auto-detects normalised
    ([-1,1]) vs voxel coordinates and (z,y,x) vs (x,y,z) axis order; the
    ``selftest`` subcommand confirms sign/scale/axis on your install.
    """
    coords = None
    for getter in ("get_warped_coordinates", "get_warp_field", "get_sample_grid"):
        fn = getattr(reg, getter, None)
        if callable(fn):
            try:
                coords = fn()
            except TypeError:
                coords = fn(reg.fixed_images, reg.moving_images)
            break
    if coords is None and hasattr(reg, "warp"):
        coords = reg.warp
    if coords is None:
        raise RuntimeError("Could not obtain a warp from this FireANTs reg object; "
                           "inspect dir(reg) and adapt _displacement_mm_from_reg (# >>> API).")

    g = (coords if torch.is_tensor(coords) else torch.as_tensor(coords)).to(device).float()
    if g.dim() == 5:
        g = g[0]
    X, Y, Z = shape
    gx, gy, gz = g[..., 0], g[..., 1], g[..., 2]         # this build returns (x,y,z) order
    if float(g.abs().max()) <= 1.5:                      # normalised
        vx = (gx + 1) * 0.5 * max(X - 1, 1)
        vy = (gy + 1) * 0.5 * max(Y - 1, 1)
        vz = (gz + 1) * 0.5 * max(Z - 1, 1)
    else:                                                # voxel
        vx, vy, vz = gx, gy, gz
    sample = torch.stack([vx, vy, vz], dim=0)
    if sample.shape[1:] == (Z, Y, X):
        sample = sample.permute(0, 3, 2, 1)
    return (sample - identity_voxel_grid(shape, device)) * spacing_mm


def register_patch(fixed: np.ndarray, moving: np.ndarray, spacing_mm: float,
                   cfg: RegConfig, device: str = "cuda",
                   init_warp_mm: Optional[np.ndarray] = None) -> np.ndarray:
    """Register one patch; return total displacement (3,X,Y,Z) in mm.

    ``init_warp_mm`` (the upsampled coarse warp, for the fine stage) pre-warps
    ``moving`` so FireANTs solves only the residual, which is then composed
    back in so the moving image is interpolated once downstream.
    """
    _require_fireants()
    shape = tuple(fixed.shape)

    moving_eff = moving
    init_t = None
    if init_warp_mm is not None:
        init_t = torch.as_tensor(init_warp_mm, dtype=torch.float32, device=device)
        moving_eff = to_numpy(apply_warp(
            torch.as_tensor(moving, dtype=torch.float32, device=device), init_t, spacing_mm))

    fimg = BatchedImages([_image_from_array(fixed, spacing_mm, device)])
    mimg = BatchedImages([_image_from_array(moving_eff, spacing_mm, device)])
    common = dict(scales=list(cfg.scales), iterations=list(cfg.iterations),
                  fixed_images=fimg, moving_images=mimg,
                  optimizer=cfg.optimizer, optimizer_lr=cfg.optimizer_lr,
                  smooth_grad_sigma=cfg.smooth_grad_sigma,
                  smooth_warp_sigma=cfg.smooth_warp_sigma, loss_type=cfg.loss_type)
    if cfg.loss_type == "cc":
        common["cc_kernel_size"] = cfg.cc_kernel

    if cfg.transform == "syn":
        if SyNRegistration is None:
            raise ImportError("SyNRegistration not available in this FireANTs build.")
        reg = SyNRegistration(**common)
    else:
        reg = GreedyRegistration(**common)
    reg.optimize()

    u_res = _displacement_mm_from_reg(reg, shape, spacing_mm, device)
    if cfg.max_disp_mm is not None:
        mag = u_res.norm(dim=0, keepdim=True).clamp_min(1e-8)
        u_res = u_res * (mag.clamp_max(cfg.max_disp_mm) / mag)
    total = u_res if init_t is None else compose(u_res, init_t, spacing_mm)
    return to_numpy(total)


def selftest(device: str = "cuda") -> bool:
    """Register a blob to a known 5-voxel shift and check the recovered warp."""
    _require_fireants()
    rng = np.random.default_rng(0)
    base = rng.random((64, 64, 64)).astype(np.float32)
    k = torch.ones(1, 1, 5, 5, 5, device=device) / 125
    base = F.conv3d(torch.as_tensor(base, device=device)[None, None], k, padding=2)[0, 0].cpu().numpy()
    shift = (5, 0, 0)
    moving = np.roll(base, shift=shift, axis=(0, 1, 2)).astype(np.float32)
    cfg = RegConfig(scales=(2, 1), iterations=(150, 80), smooth_warp_sigma=0.3)
    u = register_patch(base, moving, 0.05, cfg, device=device)
    med = np.median(u[:, 16:48, 16:48, 16:48].reshape(3, -1), axis=1)
    exp = np.array([shift[0] * 0.05, shift[1] * 0.05, shift[2] * 0.05])
    print(f"recovered median displacement (mm): {med}")
    print(f"expected (mm):                       {exp}")
    err = np.abs(med - exp)
    ok = err[0] < 0.05 and np.all(err[1:] < 0.03)
    print("SELFTEST", "PASSED" if ok else
          "FAILED -- adjust axis order / sign in _displacement_mm_from_reg (# >>> API)")
    return bool(ok)


# --------------------------------------------------------------------------- #
# Multi-GPU pool (one independent patch registration per GPU, work queue)
# --------------------------------------------------------------------------- #
@dataclass
class _Sentinel:
    pass


def _worker_loop(rank, gpu_ids, worker_fn, init_args, task_q, result_q):
    dev = f"cuda:{gpu_ids[rank]}" if torch.cuda.is_available() else "cpu"
    if dev.startswith("cuda"):
        torch.cuda.set_device(dev)
    state = worker_fn.setup(dev, *init_args) if hasattr(worker_fn, "setup") else None
    while True:
        try:
            task = task_q.get(timeout=1.0)
        except queue.Empty:
            continue
        if isinstance(task, _Sentinel):
            break
        key, payload = task
        try:
            result_q.put(("ok", key, worker_fn(payload, dev, state)))
        except Exception as e:
            import traceback
            result_q.put(("err", key, f"{e}\n{traceback.format_exc()}"))


class GpuPool:
    def __init__(self, worker_fn, num_gpus=8, gpu_ids=None, init_args=()):
        self.worker_fn = worker_fn
        self.num_gpus = num_gpus
        self.gpu_ids = gpu_ids or list(range(num_gpus))
        self.init_args = init_args

    def run(self, tasks: Iterable[tuple[Any, Any]]) -> Iterator[tuple[Any, Any]]:
        ctx = mp.get_context("spawn")
        task_q = ctx.Queue(maxsize=self.num_gpus * 4)
        result_q = ctx.Queue()
        procs = [ctx.Process(target=_worker_loop,
                             args=(r, self.gpu_ids, self.worker_fn, self.init_args,
                                   task_q, result_q))
                 for r in range(self.num_gpus)]
        for p in procs:
            p.start()
        task_list = list(tasks)
        n = len(task_list)
        feed = iter(task_list)
        done = 0
        try:
            for _ in range(self.num_gpus * 2):
                task_q.put(next(feed))
        except StopIteration:
            pass
        try:
            while done < n:
                status, key, out = result_q.get()
                done += 1
                if status == "err":
                    raise RuntimeError(f"worker failed on {key}:\n{out}")
                try:
                    task_q.put(next(feed))
                except StopIteration:
                    pass
                yield key, out
        finally:
            for _ in procs:
                task_q.put(_Sentinel())
            for p in procs:
                p.join(timeout=30)
                if p.is_alive():
                    p.terminate()


class _PatchWorker:
    """GpuPool worker: read fixed/moving (+ optional seed) patches, register.

    Top-level so torch's ``spawn`` can re-import this script and find it.
    """
    @staticmethod
    def setup(device, template_path, subject_paths, grid_dict, reg_cfg, seed_paths):
        import zarr
        return dict(
            device=device, grid=GridSpec(**grid_dict), reg_cfg=reg_cfg,
            template=zarr.open(template_path, mode="r"),
            subjects={k: zarr.open(v, mode="r") for k, v in subject_paths.items()},
            seeds={k: zarr.open(v, mode="r") for k, v in (seed_paths or {}).items()})

    def __call__(self, payload, device, state):
        sid, patch = payload["sid"], payload["patch"]
        po, ph = patch.pad_origin, patch.pad_shape
        sl = (slice(po[0], po[0]+ph[0]), slice(po[1], po[1]+ph[1]), slice(po[2], po[2]+ph[2]))
        fixed = np.asarray(state["template"][sl], np.float32)
        if (fixed > 0).mean() < 0.02:                        # skip near-empty (air) patches
            return np.zeros((3, *fixed.shape), np.float32)    # identity warp: no tissue to move
        moving = np.asarray(state["subjects"][sid][sl], np.float32)
        init = None
        if sid in state["seeds"]:
            init = np.asarray(state["seeds"][sid][(slice(None),) + sl], np.float32)
        return register_patch(fixed, moving, state["grid"].spacing_mm,
                              state["reg_cfg"], device=device, init_warp_mm=init)


# --------------------------------------------------------------------------- #
# Blockwise helpers (bounded displacement -> bounded read margin)
# --------------------------------------------------------------------------- #
def _read_padded(z, origin, shape) -> np.ndarray:
    o = np.asarray(origin, int); s = np.asarray(shape, int); full = np.asarray(z.shape, int)
    lo = np.clip(o, 0, None); hi = np.minimum(o + s, full)
    out = np.zeros(tuple(s), np.float32)
    if np.any(hi <= lo):
        return out
    sub = np.asarray(z[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], np.float32)
    d = lo - o
    out[d[0]:d[0]+sub.shape[0], d[1]:d[1]+sub.shape[1], d[2]:d[2]+sub.shape[2]] = sub
    return out


def warp_subject_blockwise(subject_zarr, warp: GridArray, grid: GridSpec, out_zarr,
                           max_disp_mm: float, device: str, block: int = 192) -> None:
    """Write ``subject ∘ warp`` (subject sampled in template space) to out_zarr."""
    margin = int(np.ceil(max_disp_mm / grid.spacing_mm)) + 2
    X, Y, Z = grid.shape
    for x0 in range(0, X, block):
        for y0 in range(0, Y, block):
            for z0 in range(0, Z, block):
                bs = (min(block, X-x0), min(block, Y-y0), min(block, Z-z0))
                u = torch.as_tensor(warp.read_block((x0, y0, z0), bs), device=device)
                ro = (x0-margin, y0-margin, z0-margin)
                rs = (bs[0]+2*margin, bs[1]+2*margin, bs[2]+2*margin)
                m = torch.as_tensor(_read_padded(subject_zarr, ro, rs), device=device)
                ident = identity_voxel_grid(bs, device) + margin
                sample = (ident + u / grid.spacing_mm).permute(1, 2, 3, 0)
                grid_n = _normalised_grid(sample, rs)[None]
                warped = F.grid_sample(m[None, None], grid_n, mode="bilinear",
                                       padding_mode="zeros", align_corners=True)[0, 0]
                out_zarr[x0:x0+bs[0], y0:y0+bs[1], z0:z0+bs[2]] = warped.cpu().numpy()


def _src_bbox(o_dst, bs_dst, gsrc: GridSpec, gdst: GridSpec, pad: int = 2):
    o = np.asarray(o_dst); bs = np.asarray(bs_dst)
    org_s = np.asarray(gsrc.origin_mm); org_d = np.asarray(gdst.origin_mm)
    world_lo = org_d + o * gdst.spacing_mm
    world_hi = org_d + (o + bs) * gdst.spacing_mm
    s_lo = np.floor((world_lo - org_s) / gsrc.spacing_mm).astype(int) - pad
    s_hi = np.ceil((world_hi - org_s) / gsrc.spacing_mm).astype(int) + pad
    return s_lo, s_hi


def _read_padded3(z, origin, shape):
    return _read_padded(z, origin, shape)


def _read_padded_field(z, origin, shape):
    o = np.asarray(origin, int); s = np.asarray(shape, int); full = np.asarray(z.shape[1:], int)
    lo = np.clip(o, 0, None); hi = np.minimum(o + s, full)
    out = np.zeros((3, *s), np.float32)
    if np.any(hi <= lo):
        return out
    sub = np.asarray(z[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], np.float32)
    d = lo - o
    out[:, d[0]:d[0]+sub.shape[1], d[1]:d[1]+sub.shape[2], d[2]:d[2]+sub.shape[3]] = sub
    return out


def resample_scalar_blockwise(src_zarr, gsrc, gdst, out_zarr, device, block=192):
    X, Y, Z = gdst.shape
    org_s = torch.tensor(gsrc.origin_mm, device=device).view(3, 1, 1, 1)
    org_d = torch.tensor(gdst.origin_mm, device=device).view(3, 1, 1, 1)
    for x0 in range(0, X, block):
        for y0 in range(0, Y, block):
            for z0 in range(0, Z, block):
                bs = (min(block, X-x0), min(block, Y-y0), min(block, Z-z0))
                s_lo, s_hi = _src_bbox((x0, y0, z0), bs, gsrc, gdst)
                sub = _read_padded3(src_zarr, s_lo, s_hi - s_lo)
                sub_t = torch.as_tensor(sub, device=device)
                ident = identity_voxel_grid(bs, device)
                world = (ident + torch.tensor([x0, y0, z0], device=device).view(3, 1, 1, 1)) \
                    * gdst.spacing_mm + org_d
                src_vox = ((world - org_s) / gsrc.spacing_mm
                           - torch.as_tensor(s_lo, device=device).view(3, 1, 1, 1)).permute(1, 2, 3, 0)
                grid = _normalised_grid(src_vox, tuple(sub.shape))[None]
                out = F.grid_sample(sub_t[None, None], grid, mode="bilinear",
                                    padding_mode="border", align_corners=True)[0, 0]
                out_zarr[x0:x0+bs[0], y0:y0+bs[1], z0:z0+bs[2]] = out.cpu().numpy()


def resample_field_blockwise(src_zarr, gsrc, gdst, out_zarr, device, block=192):
    X, Y, Z = gdst.shape
    org_s = torch.tensor(gsrc.origin_mm, device=device).view(3, 1, 1, 1)
    org_d = torch.tensor(gdst.origin_mm, device=device).view(3, 1, 1, 1)
    for x0 in range(0, X, block):
        for y0 in range(0, Y, block):
            for z0 in range(0, Z, block):
                bs = (min(block, X-x0), min(block, Y-y0), min(block, Z-z0))
                s_lo, s_hi = _src_bbox((x0, y0, z0), bs, gsrc, gdst)
                sub = _read_padded_field(src_zarr, s_lo, s_hi - s_lo)
                sub_t = torch.as_tensor(sub, device=device)
                ident = identity_voxel_grid(bs, device)
                world = (ident + torch.tensor([x0, y0, z0], device=device).view(3, 1, 1, 1)) \
                    * gdst.spacing_mm + org_d
                src_vox = ((world - org_s) / gsrc.spacing_mm
                           - torch.as_tensor(s_lo, device=device).view(3, 1, 1, 1)).permute(1, 2, 3, 0)
                grid = _normalised_grid(src_vox, tuple(sub.shape[1:]))[None]
                out = F.grid_sample(sub_t[None], grid, mode="bilinear",
                                    padding_mode="border", align_corners=True)[0]
                out_zarr[:, x0:x0+bs[0], y0:y0+bs[1], z0:z0+bs[2]] = out.cpu().numpy()


# --------------------------------------------------------------------------- #
# Block helpers used by the template loop
# --------------------------------------------------------------------------- #
def _p(d, name): return os.path.join(d, name)


def _accumulate(acc: GridArray, src_zarr, block: int = 256):
    X, Y, Z = acc.grid.shape
    for x0 in range(0, X, block):
        for y0 in range(0, Y, block):
            for z0 in range(0, Z, block):
                bs = (min(block, X-x0), min(block, Y-y0), min(block, Z-z0))
                s = np.asarray(src_zarr[x0:x0+bs[0], y0:y0+bs[1], z0:z0+bs[2]])[None]
                acc.add_block((x0, y0, z0), s.astype(np.float32))


def _scaled(src: GridArray, factor: float, out_path: str, block: int = 256):
    import zarr
    C = src.data.shape[0]; X, Y, Z = src.grid.shape
    z = zarr.open(out_path, mode="w", shape=src.data.shape, dtype="f4",
                  chunks=(C, 128, 128, 128))
    for x0 in range(0, X, block):                  # blockwise: never load the whole field
        for y0 in range(0, Y, block):
            for z0 in range(0, Z, block):
                xs = slice(x0, min(x0+block, X))
                ys = slice(y0, min(y0+block, Y))
                zs = slice(z0, min(z0+block, Z))
                z[:, xs, ys, zs] = np.asarray(src.data[:, xs, ys, zs], np.float32) * factor
    return z


def _smooth_field_inplace(field_zarr, sigma_vox: float, field_path: str, block: int = 256):
    """Gaussian-smooth each channel of a (3,X,Y,Z) field in place, blockwise.

    Reads an overlapping halo (~4 sigma) per block and writes through a temp
    store, so every halo reads the ORIGINAL (unsmoothed) field -- no dependence
    on block order, and peak host RAM is one padded block, not the whole
    259 GiB/channel volume. Temp store is removed afterwards.
    """
    from scipy.ndimage import gaussian_filter
    import zarr, shutil
    if sigma_vox <= 0:
        return
    C = field_zarr.shape[0]
    X, Y, Z = field_zarr.shape[1:]
    halo = int(np.ceil(4 * sigma_vox)) + 1
    tmp_path = field_path + ".smoothtmp"
    tmp = zarr.open(tmp_path, mode="w", shape=field_zarr.shape, dtype="f4",
                    chunks=getattr(field_zarr, "chunks", (C, 128, 128, 128)))
    for c in range(C):
        for x0 in range(0, X, block):
            for y0 in range(0, Y, block):
                for z0 in range(0, Z, block):
                    bx = min(block, X-x0); by = min(block, Y-y0); bz = min(block, Z-z0)
                    rx0 = max(0, x0-halo); ry0 = max(0, y0-halo); rz0 = max(0, z0-halo)
                    rx1 = min(X, x0+bx+halo); ry1 = min(Y, y0+by+halo); rz1 = min(Z, z0+bz+halo)
                    sub = np.asarray(field_zarr[c, rx0:rx1, ry0:ry1, rz0:rz1], np.float32)
                    sm = gaussian_filter(sub, sigma_vox)
                    tmp[c, x0:x0+bx, y0:y0+by, z0:z0+bz] = \
                        sm[x0-rx0:x0-rx0+bx, y0-ry0:y0-ry0+by, z0-rz0:z0-rz0+bz]
    for c in range(C):                              # copy temp back into the field, blockwise
        for x0 in range(0, X, block):
            for y0 in range(0, Y, block):
                for z0 in range(0, Z, block):
                    xs = slice(x0, min(x0+block, X))
                    ys = slice(y0, min(y0+block, Y))
                    zs = slice(z0, min(z0+block, Z))
                    field_zarr[c, xs, ys, zs] = tmp[c, xs, ys, zs]
    shutil.rmtree(tmp_path, ignore_errors=True)


# --------------------------------------------------------------------------- #
# The unified engine: one template iteration, and the iteration loop
# --------------------------------------------------------------------------- #
def run_iteration(template_path, subject_paths, grid, patch_cfg, reg_cfg,
                  tmpl_cfg, work_dir, iteration, num_gpus, seed_paths=None) -> dict:
    import zarr
    os.makedirs(work_dir, exist_ok=True)
    subjects = list(subject_paths)
    patches = build_patches(grid, patch_cfg)

    num = {sid: GridArray.zeros(grid, 3, _p(work_dir, f"it{iteration}_num_{sid}.zarr"))
           for sid in subjects}
    den = GridArray.zeros(grid, 1, _p(work_dir, f"it{iteration}_den.zarr"))
    windows = {i: blend_window(p, grid, patch_cfg) for i, p in enumerate(patches)}
    den_done = set()

    pool = GpuPool(_PatchWorker(), num_gpus=num_gpus,
                   init_args=(template_path, subject_paths, grid.__dict__, reg_cfg, seed_paths))
    tasks = [((sid, i), {"sid": sid, "patch": patches[i]})
             for sid in subjects for i in range(len(patches))]

    for (sid, i), u_block in pool.run(tasks):
        p = patches[i]; w = windows[i]
        num[sid].add_block(p.pad_origin, u_block * w[None])
        if i not in den_done:
            den.add_block(p.pad_origin, w[None]); den_done.add(i)

    warp_paths = {}
    X, Y, Z = grid.shape
    BLK = 256                                      # peak host RAM ~= one block, never a whole volume
    for sid in subjects:
        wpath = _p(work_dir, f"it{iteration}_warp_{sid}.zarr")
        wz = zarr.open(wpath, mode="w", shape=(3, *grid.shape), dtype="f4",
                       chunks=(3, 128, 128, 128))
        for x0 in range(0, X, BLK):
            for y0 in range(0, Y, BLK):
                for z0 in range(0, Z, BLK):
                    xs = slice(x0, min(x0+BLK, X))
                    ys = slice(y0, min(y0+BLK, Y))
                    zs = slice(z0, min(z0+BLK, Z))
                    d = np.maximum(np.asarray(den.data[0, xs, ys, zs], np.float32), 1e-6)
                    nb = np.asarray(num[sid].data[:, xs, ys, zs], np.float32)
                    wz[:, xs, ys, zs] = nb / d[None]
        if reg_cfg.post_blend_sigma_mm > 0:
            _smooth_field_inplace(wz, reg_cfg.post_blend_sigma_mm / grid.spacing_mm, wpath)
        warp_paths[sid] = wpath

    device = "cuda" if torch.cuda.is_available() else "cpu"
    isum = GridArray.zeros(grid, 1, _p(work_dir, f"it{iteration}_isum.zarr"))
    wsum = GridArray.zeros(grid, 3, _p(work_dir, f"it{iteration}_wsum.zarr"))
    max_disp = reg_cfg.max_disp_mm or patch_cfg.halo_mm
    for sid in subjects:
        warped = _p(work_dir, f"it{iteration}_warped_{sid}.zarr")
        wz = zarr.open(warped, mode="w", shape=grid.shape, dtype="f4", chunks=(128, 128, 128))
        warp_arr = GridArray(zarr.open(warp_paths[sid], mode="r"), grid)
        warp_subject_blockwise(zarr.open(subject_paths[sid], mode="r"),
                               warp_arr, grid, wz, max_disp_mm=max_disp, device=device)
        _accumulate(isum, wz)
        for x0 in range(0, X, BLK):                 # wsum += warp_arr, blockwise
            for y0 in range(0, Y, BLK):
                for z0 in range(0, Z, BLK):
                    bs = (min(BLK, X-x0), min(BLK, Y-y0), min(BLK, Z-z0))
                    blk_arr = np.asarray(
                        warp_arr.data[:, x0:x0+bs[0], y0:y0+bs[1], z0:z0+bs[2]], np.float32)
                    wsum.add_block((x0, y0, z0), blk_arr)

    n = len(subjects)
    new_tmpl = _p(work_dir, f"template_it{iteration + 1}.zarr")
    tz = zarr.open(new_tmpl, mode="w", shape=grid.shape, dtype="f4", chunks=(128, 128, 128))

    # intensity average isum/n, written blockwise (never materialise the whole volume)
    iavg_z = zarr.open(_p(work_dir, f"it{iteration}_iavg.zarr"), mode="w",
                       shape=grid.shape, dtype="f4", chunks=(128, 128, 128))
    for x0 in range(0, X, BLK):
        for y0 in range(0, Y, BLK):
            for z0 in range(0, Z, BLK):
                xs = slice(x0, min(x0+BLK, X))
                ys = slice(y0, min(y0+BLK, Y))
                zs = slice(z0, min(z0+BLK, Z))
                iavg_z[xs, ys, zs] = np.asarray(isum.data[0, xs, ys, zs], np.float32) / n

    if tmpl_cfg.unbias:
        mean_warp = GridArray(_scaled(wsum, -tmpl_cfg.shape_update_step / n,
                                      _p(work_dir, f"it{iteration}_meanwarp.zarr")), grid)
        warp_subject_blockwise(iavg_z, mean_warp, grid, tz, max_disp_mm=max_disp, device=device)
    else:
        for x0 in range(0, X, BLK):                 # copy iavg_z -> tz blockwise
            for y0 in range(0, Y, BLK):
                for z0 in range(0, Z, BLK):
                    xs = slice(x0, min(x0+BLK, X))
                    ys = slice(y0, min(y0+BLK, Y))
                    zs = slice(z0, min(z0+BLK, Z))
                    tz[xs, ys, zs] = iavg_z[xs, ys, zs]

    return {"template": new_tmpl, "warps": warp_paths, "iteration": iteration + 1}


def build_template(initial_template_path, subject_paths, grid, patch_cfg, reg_cfg,
                   tmpl_cfg, work_dir, num_gpus, seed_paths=None) -> dict:
    """Run ``tmpl_cfg.n_iterations`` of unbiased template construction.

    This is the SHARED engine for both the 50 um and 20 um stages -- the only
    difference between stages is the params passed in (and, for fine, the
    ``seed_paths`` that pre-warp each subject).
    """
    tmpl = initial_template_path
    warps = None
    for k in range(tmpl_cfg.n_iterations):
        print(f"\n=== template iteration {k + 1}/{tmpl_cfg.n_iterations} ===")
        res = run_iteration(tmpl, subject_paths, grid, patch_cfg, reg_cfg,
                            tmpl_cfg, work_dir, k, num_gpus, seed_paths)
        tmpl, warps = res["template"], res["warps"]
        print(f"  -> {tmpl}")
    return {"template": tmpl, "warps": warps}


def initial_template_mean(subject_paths, grid, out_path, block=256):
    """Voxelwise mean of the on-grid subjects -> initial template (blockwise)."""
    import zarr
    subs = [zarr.open(p, mode="r") for p in subject_paths.values()]
    tz = zarr.open(out_path, mode="w", shape=grid.shape, dtype="f4", chunks=(128, 128, 128))
    X, Y, Z = grid.shape
    for x0 in range(0, X, block):
        for y0 in range(0, Y, block):
            for z0 in range(0, Z, block):
                bx, by, bz = min(block, X-x0), min(block, Y-y0), min(block, Z-z0)
                acc = np.zeros((bx, by, bz), np.float32)
                for s in subs:
                    acc += np.asarray(s[x0:x0+bx, y0:y0+by, z0:z0+bz], np.float32)
                tz[x0:x0+bx, y0:y0+by, z0:z0+bz] = acc / len(subs)
    return out_path


# --------------------------------------------------------------------------- #
# Stage drivers (both end up calling build_template)
# --------------------------------------------------------------------------- #
def _cfgs_from_args(args) -> tuple[PatchConfig, RegConfig, TemplateConfig]:
    """Build all three configs purely from CLI flags (no stage presets)."""
    patch = PatchConfig(core_mm=(args.core_mm,) * 3, halo_mm=args.halo_mm, window="hann")
    reg = RegConfig(
        transform=args.transform, loss_type=args.loss, cc_kernel=args.cc_kernel,
        scales=tuple(args.scales), iterations=tuple(args.iterations),
        optimizer="Adam", optimizer_lr=args.lr,
        smooth_grad_sigma=args.smooth_grad_sigma, smooth_warp_sigma=args.smooth_warp_sigma,
        # negative --max-disp-mm disables the clamp
        max_disp_mm=(None if args.max_disp_mm is not None and args.max_disp_mm < 0
                     else args.max_disp_mm),
        post_blend_sigma_mm=args.post_blend_sigma_mm,
    )
    tmpl = TemplateConfig(n_iterations=args.template_iters,
                          shape_update_step=args.shape_update_step, unbias=args.unbias)
    if len(reg.scales) != len(reg.iterations):
        raise SystemExit(f"--scales ({reg.scales}) and --iterations ({reg.iterations}) "
                         "must have the same length")
    return patch, reg, tmpl


def run_coarse(args) -> dict:
    grid = GridSpec(shape=args.shape, spacing_mm=args.spacing, origin_mm=args.origin)
    patch, reg, tmpl = _cfgs_from_args(args)
    os.makedirs(args.work_dir, exist_ok=True)
    print(f"[coarse] {len(args.subjects)} subjects on grid {grid.shape} @ {grid.spacing_mm} mm")
    print("[coarse] building initial template (voxelwise mean)...")
    t0 = initial_template_mean(args.subjects, grid, _p(args.work_dir, "template_it0.zarr"))
    res = build_template(t0, args.subjects, grid, patch, reg, tmpl,
                         args.work_dir, args.num_gpus)
    print("\nDONE (coarse)\n  template:", res["template"])
    for sid, w in res["warps"].items():
        print(f"  warp[{sid}]: {w}")
    return res


def _resolve_seeds(args, subjects: dict, coarse_iters: int):
    """Default seed paths from --seed-dir using the coarse naming convention."""
    tmpl = args.seed_template or _p(args.seed_dir, f"template_it{coarse_iters}.zarr")
    warps = {}
    if args.seed_warps:                     # explicit pattern with {sid}
        warps = {sid: args.seed_warps.format(sid=sid) for sid in subjects}
    else:
        for sid in subjects:
            warps[sid] = _p(args.seed_dir, f"it{coarse_iters - 1}_warp_{sid}.zarr")
    return tmpl, warps


def run_fine(args) -> dict:
    import zarr
    grid_coarse = GridSpec(shape=args.shape, spacing_mm=args.spacing, origin_mm=args.origin)
    grid_fine = grid_coarse.rescaled(args.fine_spacing)
    patch, reg, tmpl = _cfgs_from_args(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.work_dir, exist_ok=True)

    coarse_iters = int(args.seed_template_iters) if args.seed_template_iters is not None \
        else DEF_TEMPLATE_ITERS
    seed_tmpl, seed_warps = _resolve_seeds(args, args.subjects, coarse_iters)
    print(f"[fine] grid {grid_coarse.shape}@{grid_coarse.spacing_mm} -> "
          f"{grid_fine.shape}@{grid_fine.spacing_mm} mm")
    print(f"[fine] seed template: {seed_tmpl}")

    # 1) initial fine template = upsampled coarse template
    t_init = _p(args.work_dir, "template_fine_init.zarr")
    tz = zarr.open(t_init, mode="w", shape=grid_fine.shape, dtype="f4", chunks=(128, 128, 128))
    resample_scalar_blockwise(zarr.open(seed_tmpl, mode="r"), grid_coarse, grid_fine, tz, device)

    # 2) seed warps = upsampled coarse forward warps
    seed_paths = {}
    for sid, wpath in seed_warps.items():
        spath = _p(args.work_dir, f"seed_{sid}.zarr")
        sz = zarr.open(spath, mode="w", shape=(3, *grid_fine.shape), dtype="f4",
                       chunks=(3, 128, 128, 128))
        resample_field_blockwise(zarr.open(wpath, mode="r"), grid_coarse, grid_fine, sz, device)
        seed_paths[sid] = spath

    # 3) identical loop on the fine grid, seeded
    res = build_template(t_init, args.subjects, grid_fine, patch, reg, tmpl,
                         args.work_dir, args.num_gpus, seed_paths=seed_paths)
    print("\nDONE (fine)\n  template:", res["template"])
    for sid, w in res["warps"].items():
        print(f"  warp[{sid}]: {w}")
    return res


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_subjects(tokens: Optional[list[str]]) -> dict[str, str]:
    if not tokens:
        return dict(SUBJECTS_DEFAULT)
    out = {}
    for t in tokens:
        if "=" not in t:
            raise SystemExit(f"--subjects expects sid=path tokens, got {t!r}")
        sid, path = t.split("=", 1)
        out[sid] = path
    return out


def _parse_triple(s, default):
    if s is None:
        return tuple(default)
    parts = [p for p in s.replace(",", " ").split() if p]
    if len(parts) != 3:
        raise SystemExit(f"expected 3 comma-separated values, got {s!r}")
    return tuple(int(round(float(p))) if float(p).is_integer() else float(p) for p in parts)


def _int_list(s) -> tuple[int, ...]:
    """argparse type for comma/space-separated ints, e.g. '4,2,1'."""
    parts = [p for p in str(s).replace(",", " ").split() if p]
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated ints, got {s!r}")


def _add_common(sub):
    # --- data + grid -------------------------------------------------------- #
    g = sub.add_argument_group("data + grid")
    g.add_argument("--subjects", nargs="+", metavar="sid=path",
                   help="subject_id=path tokens; default = CONFIG SUBJECTS_DEFAULT")
    g.add_argument("--shape", help="coarse grid X,Y,Z in voxels (default from CONFIG)")
    g.add_argument("--spacing", type=float, default=DEF_SPACING_MM,
                   help=f"coarse voxel size mm (default {DEF_SPACING_MM})")
    g.add_argument("--origin", help="grid origin mm X,Y,Z (default from CONFIG)")
    g.add_argument("--work-dir", required=True)
    g.add_argument("--num-gpus", type=int, default=8)

    # --- patch tiling ------------------------------------------------------- #
    p = sub.add_argument_group("patch tiling  (coarse defaults; [fine] values in help)")
    p.add_argument("--core-mm", type=float, default=DEF_CORE_MM,
                   help=f"patch core cube, mm (default {DEF_CORE_MM}; [fine 2.0])")
    p.add_argument("--halo-mm", type=float, default=DEF_HALO_MM,
                   help=f"context halo per side, mm; must exceed max-disp + LNCC radius "
                        f"(default {DEF_HALO_MM}; [fine 0.5])")

    # --- registration ------------------------------------------------------- #
    r = sub.add_argument_group("registration")
    r.add_argument("--scales", type=_int_list, default=_int_list(DEF_SCALES),
                   help=f"in-patch downsample factors, coarse->fine "
                        f"(default {DEF_SCALES}; [fine 2,1])")
    r.add_argument("--iterations", type=_int_list, default=_int_list(DEF_ITERATIONS),
                   help=f"iterations per scale, same length as --scales "
                        f"(default {DEF_ITERATIONS}; [fine 60,40])")
    r.add_argument("--max-disp-mm", type=float, default=DEF_MAX_DISP_MM,
                   help=f"per-patch displacement clamp, mm; keep < halo - cc_kernel*spacing/2; "
                        f"negative disables (default {DEF_MAX_DISP_MM}; [fine 0.2])")
    r.add_argument("--post-blend-sigma-mm", type=float, default=DEF_POST_BLEND,
                   help=f"Gaussian smoothing of the blended field, mm; 0 disables "
                        f"(default {DEF_POST_BLEND}; [fine 0.02])")
    r.add_argument("--transform", choices=["greedy", "syn"], default=DEF_TRANSFORM,
                   help=f"deformable model (default {DEF_TRANSFORM})")
    r.add_argument("--loss", choices=["cc", "mi", "mse"], default=DEF_LOSS,
                   help=f"similarity; 'cc' is fused LNCC (default {DEF_LOSS})")
    r.add_argument("--cc-kernel", type=int, default=DEF_CC_KERNEL,
                   help=f"LNCC window (default {DEF_CC_KERNEL})")
    r.add_argument("--lr", type=float, default=DEF_LR, help=f"Adam lr (default {DEF_LR})")
    r.add_argument("--smooth-grad-sigma", type=float, default=DEF_SMOOTH_GRAD,
                   help=f"Sobolev sigma on the update (default {DEF_SMOOTH_GRAD})")
    r.add_argument("--smooth-warp-sigma", type=float, default=DEF_SMOOTH_WARP,
                   help=f"Sobolev sigma on the displacement (default {DEF_SMOOTH_WARP})")

    # --- unbiased template -------------------------------------------------- #
    t = sub.add_argument_group("unbiased template")
    t.add_argument("--template-iters", type=int, default=DEF_TEMPLATE_ITERS,
                   help=f"outer template iterations (default {DEF_TEMPLATE_ITERS}; "
                        f"coarse 3-5, [fine 1-2])")
    t.add_argument("--shape-update-step", type=float, default=DEF_SHAPE_UPDATE,
                   help=f"mean-shape recenter step (default {DEF_SHAPE_UPDATE})")
    t.add_argument("--unbias", action=argparse.BooleanOptionalAction, default=True,
                   help="recenter template to the population mean shape each iter (default on)")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="hipct_patchreg.py",
        description="Patchwise unbiased-template registration for HiP-CT (one file, two stages).")
    subs = ap.add_subparsers(dest="cmd", required=True)

    subs.add_parser("selftest", help="verify the FireANTs warp convention (run first)")

    c = subs.add_parser("coarse", help="50 um unbiased template")
    _add_common(c)

    f = subs.add_parser("fine", help="20 um refinement, seeded from the coarse stage")
    _add_common(f)
    f.add_argument("--fine-spacing", type=float, default=DEF_FINE_SPACING_MM,
                   help=f"fine voxel size mm (default {DEF_FINE_SPACING_MM})")
    f.add_argument("--seed-dir", help="coarse work dir; seeds derived by naming convention")
    f.add_argument("--seed-template", help="explicit path to the coarse template zarr")
    f.add_argument("--seed-warps", help="explicit warp path pattern containing {sid}")
    f.add_argument("--seed-template-iters", type=int, default=None,
                   help="coarse template_iters used to find default seed filenames")

    args = ap.parse_args(argv)

    if args.cmd == "selftest":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ok = selftest(dev)
        raise SystemExit(0 if ok else 1)

    # normalise shared args
    args.subjects = _parse_subjects(args.subjects)
    args.shape = _parse_triple(args.shape, GRID_SHAPE_DEFAULT)
    args.origin = tuple(float(x) for x in _parse_triple(args.origin, GRID_ORIGIN_DEFAULT))

    if args.cmd == "coarse":
        run_coarse(args)
    elif args.cmd == "fine":
        if not (args.seed_dir or args.seed_template):
            raise SystemExit("fine stage needs --seed-dir (or --seed-template + --seed-warps)")
        run_fine(args)


if __name__ == "__main__":
    main()
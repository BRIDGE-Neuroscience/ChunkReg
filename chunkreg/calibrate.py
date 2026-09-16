"""Measure the constants the derived rules depend on.

Three numbers decide whether a configuration is safe and what it costs, and all
three are properties of a particular backbone on particular hardware rather
than things to look up:

``r_f``
    The feature extractor's receptive-field radius. It is subtracted from the
    halo before anything is left for displacement, so guessing it low makes
    every chunk exceed a clamp the halo cannot actually support, silently.

``bytes_per_voxel_channel``
    Sets which profile fits the device.

``throughput``
    Turns the planner's work estimate into hours.

Measuring the receptive field is direct: perturb one voxel, see how far the
change in the output reaches. For a raw intensity extractor the answer is zero
and the measurement confirms it.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from . import xp
from .config import RunConfig, StageSpec
from .engines import get_engine
from .features import get_extractor

__all__ = ["Calibration", "calibrate", "measure_receptive_field"]


@dataclass
class Calibration:
    features: str
    r_f: int
    r_f_declared: int
    bytes_per_voxel_channel: float
    throughput_ch_vox_per_s: float
    seconds_per_chunk: float
    padded: int
    channels: int
    device: str

    def to_json(self) -> dict:
        return asdict(self)

    @property
    def halo_is_adequate(self) -> bool:
        return self.r_f <= self.r_f_declared

    def format(self) -> str:
        lines = [
            f"calibration on {self.device}",
            f"  features {self.features}, {self.channels} channels, "
            f"padded chunk {self.padded}",
            f"  receptive field radius   {self.r_f} voxels "
            f"(profile declares {self.r_f_declared})",
            f"  bytes per voxel-channel  {self.bytes_per_voxel_channel:.0f}",
            f"  throughput               {self.throughput_ch_vox_per_s:,.0f} ch-vox/s"
            f"  ({self.seconds_per_chunk:.1f} s per chunk)",
        ]
        if not self.halo_is_adequate:
            lines.append(
                f"  WARNING: the measured receptive field is larger than the "
                f"profile assumes, so the halo budget is overdrawn by "
                f"{self.r_f - self.r_f_declared} voxels and the effective "
                f"clamp is smaller than reported. Raise the profile halo by "
                f"at least that much, or set r_f to {self.r_f}."
            )
        return "\n".join(lines)


def measure_receptive_field(
    extractor,
    spacing_mm: float = 0.05,
    size: int = 64,
    threshold: float = 0.01,
    seed: int = 0,
) -> int:
    """How far a single-voxel change propagates through the extractor.

    Returns the radius in voxels beyond which the feature response changes by
    less than ``threshold`` of its peak change. This is the quantity the halo
    bound calls ``r_f``.
    """
    rng = np.random.default_rng(seed)
    from scipy import ndimage

    base = ndimage.gaussian_filter(
        rng.random((size, size, size)).astype(np.float32), 2.0
    )
    base -= base.min()
    base /= max(base.max(), 1e-8)

    poked = base.copy()
    c = size // 2
    poked[c, c, c] = 1.0 - base[c, c, c]

    a = extractor(xp.put(base), spacing_mm)
    b = extractor(xp.put(poked), spacing_mm)
    if xp.is_tensor(a):
        import torch

        delta = (a - b).abs().amax(dim=0)
        peak = float(delta.max())
        if peak <= 0:
            return 0
        idx = torch.nonzero(delta > threshold * peak)
        if idx.numel() == 0:
            return 0
        return int((idx - c).abs().max())
    delta = np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).max(axis=0)
    peak = float(delta.max())
    if peak <= 0:
        return 0

    idx = np.argwhere(delta > threshold * peak)
    if idx.size == 0:
        return 0
    return int(np.abs(idx - c).max())


def calibrate(
    cfg: RunConfig, engine: str | None = None, stages=None
) -> Calibration:
    """Measure the receptive field, per-voxel memory and throughput.

    ``stages`` is what the timed chunk runs; pass the finest level's stages to
    time what dominates the run. Without it a short fixed schedule is used.
    """
    profile = cfg.profile
    extractor = get_extractor(profile.features)
    extractor.setup()
    device = xp.device_name()
    eng = get_engine(engine or cfg.engine_for(device))

    r_f = measure_receptive_field(extractor)

    # Throughput on a chunk small enough to time quickly, scaled to the real
    # padded size. Registration cost is close to linear in voxels once the
    # pyramid schedule is fixed, so this extrapolates honestly.
    #
    # On a GPU the probe is a full padded chunk: a small one mostly measures
    # kernel launch overhead and overstates the cost of a real chunk. The CPU
    # reference keeps a small probe so calibration stays quick there.
    on_gpu = xp.on_gpu()
    probe_side = profile.padded if on_gpu else min(48, profile.padded)
    rng = np.random.default_rng(1)
    if xp.uses_torch():
        import torch

        from . import fields as _fields

        v = _fields.smooth(xp.put(rng.random((probe_side,) * 3)), 2.0)
        shifted = torch.roll(v, shifts=2, dims=0)
    else:
        from scipy import ndimage

        v = ndimage.gaussian_filter(
            rng.random((probe_side,) * 3).astype(np.float32), 2.0
        )
        shifted = np.roll(v, 2, axis=0)
    # One iteration count per scale, whatever the profile's pyramid depth. A
    # fixed two-element schedule raised a length mismatch on every profile with
    # three or more scales, which is exactly the deep-pyramid profile a user
    # would reach for calibration to justify.
    _SCHEDULE = (20, 12, 8, 6)
    iterations = tuple(
        _SCHEDULE[min(i, len(_SCHEDULE) - 1)] for i in range(len(profile.scales))
    )
    stage = StageSpec(kind="greedy", scales=profile.scales, iterations=iterations)
    run_stages = list(stages) if stages else [stage]

    def one_chunk():
        eng.register(extractor(v, 0.05), extractor(shifted, 0.05), 0.05, run_stages)
        if on_gpu:
            torch.cuda.synchronize()

    if on_gpu:
        one_chunk()  # kernel selection and allocator warm-up, not timed
        torch.cuda.reset_peak_memory_stats()
    # Timed end to end, feature extraction included: that is what each chunk
    # of a run costs.
    t0 = time.perf_counter()
    one_chunk()
    elapsed = max(time.perf_counter() - t0, 1e-6)
    measured_bytes = (
        torch.cuda.max_memory_allocated() / (probe_side**3 * profile.channels)
        if on_gpu
        else None
    )

    probe_work = probe_side**3 * profile.channels
    throughput = probe_work / elapsed
    seconds_per_chunk = (profile.padded**3 * profile.channels) / throughput

    device_label, per_voxel_channel = _device_and_memory(profile)
    if measured_bytes is not None:
        per_voxel_channel = float(measured_bytes)

    return Calibration(
        features=profile.features,
        r_f=r_f,
        r_f_declared=profile.r_f,
        bytes_per_voxel_channel=per_voxel_channel,
        throughput_ch_vox_per_s=throughput,
        seconds_per_chunk=seconds_per_chunk,
        padded=profile.padded,
        channels=profile.channels,
        device=device_label,
    )


def _device_and_memory(profile) -> tuple[str, float]:
    """Peak device bytes per voxel-channel, measured if a GPU is present."""
    try:
        import torch
    except ImportError:
        return "cpu (no torch)", 90.0
    if not torch.cuda.is_available():
        return "cpu", 90.0
    name = torch.cuda.get_device_name(0)
    torch.cuda.reset_peak_memory_stats()
    side = min(96, profile.padded)
    x = torch.zeros(
        (1, profile.channels, side, side, side), dtype=torch.float32, device="cuda"
    )
    y = torch.nn.functional.avg_pool3d(x, 2)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del x, y
    torch.cuda.empty_cache()
    # The pooling probe measures only allocator behaviour; the optimiser's own
    # working set dominates, so this is reported as a floor rather than used
    # directly. The published 90 stands until a real registration is profiled.
    floor = peak / (side**3 * profile.channels)
    return name, max(90.0, float(floor))

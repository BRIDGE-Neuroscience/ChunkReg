"""Resolve a run before spending anything on it.

The planner answers, from the configuration and a native grid alone: how many
pyramid levels there are, how many chunks each has, what displacement clamp
each gets, how much device memory a task needs, and how much work the whole run
is. Nothing here touches data, so it runs in milliseconds and is the first
thing to look at when a run seems too expensive.

Work is counted in **padded channel-voxels**, not registrations. Registrations
are not comparable units: a 16-channel chunk costs sixteen times a 1-channel
chunk of the same size, and a padded chunk costs 2.6 times its core. Counting
the quantity that actually scales makes the three levers visible side by side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .config import RunConfig
from .grid import GridSpec, Profile, pyramid, shard_grid, tile

__all__ = ["LevelPlan", "Plan", "plan"]


@dataclass(frozen=True)
class LevelPlan:
    level: int
    grid: GridSpec
    n_chunks: int
    chunk_grid: tuple[int, int, int]
    d_max_mm: float
    d_max_vox: float
    cap: int
    n_tasks: int
    work_ch_vox: float
    """Padded channel-voxels at the cap, for all subjects."""

    @property
    def spacing_mm(self) -> float:
        return self.grid.spacing_mm

    @property
    def n_voxels(self) -> int:
        return self.grid.n_voxels

    @property
    def chunk_width_mm(self) -> float:
        """``L``: the physical width a chunk can see, which bounds the
        deformation wavelength this level can resolve."""
        return self._core * self.grid.spacing_mm

    _core: int = 256


@dataclass(frozen=True)
class Plan:
    native: GridSpec
    profile: Profile
    profile_name: str
    n_subjects: int
    levels: tuple[LevelPlan, ...]
    mem_gb: float
    overhead: float
    throughput_ch_vox_per_s: float
    warnings: tuple[str, ...] = ()

    pyramid_levels: int = 0
    """Levels in the whole pyramid, including any that ``levels.run`` skips."""

    @property
    def depth(self) -> int:
        return len(self.levels) - 1

    @property
    def work_ch_vox(self) -> float:
        return sum(lv.work_ch_vox for lv in self.levels)

    @property
    def gpu_hours(self) -> float:
        return self.work_ch_vox / self.throughput_ch_vox_per_s / 3600.0

    @property
    def n_registrations(self) -> int:
        return sum(lv.n_chunks * lv.cap * self.n_subjects for lv in self.levels)

    def wall_hours(self, n_gpus: int) -> float:
        if n_gpus < 1:
            raise ValueError("n_gpus must be at least 1")
        return self.gpu_hours / n_gpus

    def dominant_level(self) -> LevelPlan:
        return max(self.levels, key=lambda lv: lv.work_ch_vox)

    def format(self, n_gpus: Sequence[int] = (8, 64)) -> str:
        """A table meant to be read before committing to a run."""
        p = self.profile
        out: list[str] = []
        out.append(
            f"profile {self.profile_name}: core {p.core} halo {p.halo} "
            f"padded {p.padded} | {p.channels} ch ({p.features}) | "
            f"scales {list(p.scales)}"
        )
        out.append(
            f"  halo budget: clamp {p.d_max_vox():.0f} + support "
            f"{p.support_vox():.0f} (s_max {p.s_max}) + receptive field "
            f"{p.r_f} = {p.halo}"
        )
        out.append(
            f"  peak device memory {self.mem_gb:.0f} GB/task | "
            f"chunk overhead {self.overhead:.2f}x"
        )
        out.append(
            f"  {self.n_subjects} subject(s), {self.depth + 1} level(s) run"
            + (
                f" of {self.pyramid_levels} in the pyramid"
                if self.pyramid_levels > len(self.levels)
                else ""
            )
            + f", native {self.native.shape} @ {self.native.spacing_mm:g} mm"
        )
        out.append("")
        head = (
            f"{'lvl':>3} {'spacing':>9} {'shape':>22} {'voxels':>10} "
            f"{'chunks':>8} {'L (mm)':>8} {'D_max':>9} {'cap':>4} "
            f"{'tasks':>7} {'work':>10} {'%':>5}"
        )
        out.append(head)
        out.append("-" * len(head))
        total = self.work_ch_vox or 1.0
        for lv in self.levels:
            out.append(
                f"{lv.level:>3} {lv.spacing_mm:>8.4g} "
                f"{str(tuple(lv.grid.shape)):>22} {_si(lv.n_voxels):>10} "
                f"{lv.n_chunks:>8} {lv.chunk_width_mm:>8.1f} "
                f"{lv.d_max_mm:>8.3f} {lv.cap:>4} {lv.n_tasks:>7} "
                f"{_si(lv.work_ch_vox):>10} {100 * lv.work_ch_vox / total:>4.0f}%"
            )
        out.append("-" * len(head))
        out.append(
            f"  work {_si(self.work_ch_vox)} channel-voxels "
            f"({self.n_registrations} registrations at the caps)"
        )
        out.append(
            f"  {self.gpu_hours:,.0f} GPU-hours at "
            f"{_si(self.throughput_ch_vox_per_s)} ch-vox/s"
            + "".join(
                f" | {n} GPUs: {self.wall_hours(n):,.1f} h" for n in n_gpus
            )
        )
        if self.warnings:
            out.append("")
            for w in self.warnings:
                out.append(f"  warning: {w}")
        return "\n".join(out)


def _si(x: float) -> str:
    for unit, scale in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(x) >= scale:
            return f"{x / scale:.2f}{unit}"
    return f"{x:.0f}"


def plan(cfg: RunConfig, native: GridSpec) -> Plan:
    """Resolve a configuration against a native grid.

    The cap is used as the pass count, so the result is an upper bound: the
    stopping rule normally exits a level earlier, and a seeded level that
    inherits a converged shape exits after one pass.
    """
    warnings = list(cfg.validate())
    p = cfg.profile
    grids = pyramid(native, p)
    run_levels = set(cfg.levels.run_levels(grids))
    n = max(cfg.n_subjects, 1)
    per_task = max(1, cfg.slurm.register.chunks_per_task)

    levels: list[LevelPlan] = []
    for k, g in enumerate(grids):
        if k not in run_levels:
            continue  # levels.run leaves it out, so it costs nothing
        chunks = tile(g, p)
        cap = cfg.levels.cap(k)
        if len(chunks) == 1:
            # A single-chunk level has no halo, so nothing bounds it by data
            # dependence. It is bounded by the aperture instead: a fraction of
            # the volume's own extent. It also pays no padding overhead.
            d_mm = cfg.levels.level0_max_disp_frac * min(g.extent_mm)
            d_vox = d_mm / g.spacing_mm
            work = g.n_voxels * p.channels * cap * n
        else:
            d_vox = p.d_max_vox()
            d_mm = p.d_max_mm(g.spacing_mm)
            work = g.n_voxels * p.overhead * p.channels * cap * n
        levels.append(
            LevelPlan(
                level=k,
                grid=g,
                n_chunks=len(chunks),
                chunk_grid=shard_grid(g, p),
                d_max_vox=d_vox,
                d_max_mm=d_mm,
                cap=cap,
                n_tasks=-(-(len(chunks) * n) // per_task),
                work_ch_vox=float(work),
                _core=p.core,
            )
        )

    if len(levels) > 1:
        deepest = levels[-1]
        if deepest.work_ch_vox / max(sum(l.work_ch_vox for l in levels), 1) > 0.9:
            warnings.append(
                "the finest level run is over 90% of the work; check that "
                "the caps for coarser levels are not set too low to converge"
            )

    return Plan(
        native=native,
        profile=p,
        profile_name=cfg.profile_name,
        n_subjects=cfg.n_subjects,
        levels=tuple(levels),
        mem_gb=p.mem_gb(cfg.bytes_per_voxel_channel),
        overhead=p.overhead,
        throughput_ch_vox_per_s=cfg.throughput_ch_vox_per_s,
        warnings=tuple(warnings),
        pyramid_levels=len(grids),
    )

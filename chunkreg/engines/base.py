"""The engine contract.

An engine sees only a chunk: two feature stacks ``(C, Z, Y, X)`` on the same
isotropic grid, a spacing, and a list of stages. It returns a displacement in
millimetres on that same grid, in the package's conventions (see
:mod:`chunkreg.fields`). It knows nothing about pyramids, seeds, blending or
storage, all of which are handled above it.

Returning the achieved iteration counts is part of the contract, not a
convenience: the iteration caps for the next run are set from the 95th
percentile of that histogram, so a stage that early-stops has to say so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from .. import xp as _xp

__all__ = ["RegResult", "Engine", "StageNotSupported"]


class StageNotSupported(NotImplementedError):
    """This engine cannot run a stage kind it was asked for.

    Raised rather than silently skipping, so a configuration that asks a
    reference engine for an affine stage fails loudly instead of quietly
    producing a deformable-only alignment.
    """


@dataclass
class RegResult:
    """What an engine returns for one chunk."""

    disp_mm: np.ndarray
    """``(3, Z, Y, X)`` float32, millimetres, fixed to moving, axis order."""
    loss_curve: list[float] = field(default_factory=list)
    iters_per_scale: list[int] = field(default_factory=list)
    """Achieved iterations per pyramid scale, coarse to fine. Feeds the cap
    tuning described in the design's iteration-schedule section."""
    converged: bool = True
    """False when a stage exhausted its iteration cap with the loss still
    falling, which QC surfaces as a cap that binds."""
    affine: np.ndarray | None = None

    def __post_init__(self) -> None:
        u = (
            _xp.to_float32(self.disp_mm)
            if _xp.is_tensor(self.disp_mm)
            else np.asarray(self.disp_mm, dtype=np.float32)
        )
        if u.ndim != 4 or u.shape[0] != 3:
            raise ValueError(f"disp_mm must be (3, Z, Y, X), got {tuple(u.shape)}")
        self.disp_mm = u

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.disp_mm.shape[1:])

    @property
    def total_iters(self) -> int:
        return int(sum(self.iters_per_scale))


@runtime_checkable
class Engine(Protocol):
    name: str
    supports: frozenset[str]

    def register(
        self,
        fixed: np.ndarray,
        moving: np.ndarray,
        spacing_mm: float,
        stages: Sequence,
        init_affine: np.ndarray | None = None,
        device: str | None = None,
    ) -> RegResult: ...


def check_inputs(fixed: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalise the pair an engine was handed.

    Device arrays stay on their device; host arrays stay on the host.
    """
    if _xp.is_tensor(fixed) or _xp.is_tensor(moving):
        f = (fixed if _xp.is_tensor(fixed) else _xp.put(fixed)).float().contiguous()
        m = (moving if _xp.is_tensor(moving) else _xp.put(moving)).float().contiguous()
    else:
        f = np.ascontiguousarray(fixed, dtype=np.float32)
        m = np.ascontiguousarray(moving, dtype=np.float32)
    if f.ndim != 4 or m.ndim != 4:
        raise ValueError(
            f"expected (C, Z, Y, X) feature stacks, got {tuple(f.shape)} and "
            f"{tuple(m.shape)}"
        )
    if tuple(f.shape) != tuple(m.shape):
        raise ValueError(
            f"fixed {tuple(f.shape)} and moving {tuple(m.shape)} must share a grid and "
            f"channel count; the moving chunk is resampled onto the fixed grid "
            f"before features are extracted"
        )
    return f, m


def check_stages(stages: Sequence, supports: frozenset[str], engine: str) -> None:
    for s in stages:
        kind = getattr(s, "kind", s)
        if kind not in supports:
            raise StageNotSupported(
                f"the {engine} engine cannot run a {kind!r} stage; "
                f"it supports {sorted(supports)}"
            )

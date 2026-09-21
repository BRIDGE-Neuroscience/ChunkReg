"""Several extractors side by side, and a random channel subset per pass.

Combining
    anatomix is learned and sees broad anatomy; MIND-SSC is hand-built and
    sees local self-similarity at a known, small radius. Registering on both
    gives the optimiser the two kinds of evidence at once. The channels are
    concatenated as each part produces them, as anatomix's own
    ``anatomix+mindssc`` registration does: anatomix L2 normalised per voxel,
    MIND-SSC as its raw ``exp(-d / var)`` responses. The correlation loss is
    computed per channel, so no rescaling across parts is needed.

Sampling
    Registration cost and device memory are linear in the channel count.
    :class:`ChannelSample` registers only ``k`` of the channels in each
    template pass and draws a fresh subset for the next, so over a level every
    channel contributes while no single pass pays for all of them. The draw is
    stratified by part (a combined extractor keeps both anatomix and MIND in
    every pass) and seeded by ``(seed, level, iteration)``, so every chunk and
    every worker of a pass, on any machine, registers the same channels. It
    has to: the blend merges residuals across chunk boundaries, and residuals
    solved on different channels would not be answers to the same problem.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

__all__ = ["CombinedFeatures", "ChannelSample", "stratified_pick"]


def _is_tensor(x) -> bool:
    return type(x).__module__.split(".", 1)[0] == "torch"


class CombinedFeatures:
    """The channels of several extractors, concatenated."""

    def __init__(self, parts: Sequence) -> None:
        if not parts:
            raise ValueError("a combined extractor needs at least one part")
        self.parts = list(parts)
        self.name = "+".join(p.name for p in self.parts)
        self.channels = sum(int(p.channels) for p in self.parts)
        self.r_f = max(int(p.r_f) for p in self.parts)

    @property
    def groups(self) -> tuple[int, ...]:
        """Channel count of each part, in order."""
        return tuple(int(p.channels) for p in self.parts)

    def setup(self, device: str | None = None) -> None:
        for p in self.parts:
            p.setup(device)

    def __call__(self, patch, spacing_mm: float, mask=None):
        # As in anatomix, a mask gates the learned channels and leaves MIND-SSC
        # alone: its responses are ratios, and zeroing them is not "no signal".
        outs = [
            p(patch, spacing_mm, None if p.name == "mindssc" else mask)
            for p in self.parts
        ]
        if any(_is_tensor(o) for o in outs):
            import torch

            from .. import xp as _xp

            dev = next(o.device for o in outs if _is_tensor(o))
            outs = [o if _is_tensor(o) else _xp.put(o) for o in outs]
            return torch.cat([o.to(dev, torch.float32) for o in outs], dim=0).contiguous()
        out = np.concatenate([np.asarray(o, dtype=np.float32) for o in outs], axis=0)
        return np.ascontiguousarray(out)


def stratified_pick(
    groups: Sequence[int], k: int, seed: int, level: int, iteration: int
) -> np.ndarray:
    """``k`` channel indices, spread over the groups in proportion to size.

    Every group with channels gets at least one pick when ``k`` allows it.
    Deterministic in ``(seed, level, iteration)``.
    """
    total = int(sum(groups))
    if not 1 <= k <= total:
        raise ValueError(f"cannot sample {k} of {total} channels")
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, int(level), int(iteration)])
    live = [i for i, n in enumerate(groups) if n > 0]
    quota = {i: groups[i] * k / total for i in live}
    take = {i: int(math.floor(quota[i])) for i in live}
    if k >= len(live):
        for i in live:
            take[i] = max(take[i], 1)
    # Hand out what rounding left, largest remainder first, then trim any
    # excess the minimum of one created, from the largest groups.
    while sum(take.values()) < k:
        i = max((j for j in live if take[j] < groups[j]),
                key=lambda j: (quota[j] - take[j], groups[j]))
        take[i] += 1
    while sum(take.values()) > k:
        i = max((j for j in live if take[j] > 1), key=lambda j: take[j] - quota[j])
        take[i] -= 1
    picks = []
    start = 0
    for i, n in enumerate(groups):
        if take.get(i):
            picks.extend(start + rng.choice(n, size=take[i], replace=False))
        start += n
    return np.sort(np.asarray(picks, dtype=np.int64))


class ChannelSample:
    """Register ``k`` of an extractor's channels, a new subset every pass."""

    def __init__(self, inner, k: int, seed: int = 0) -> None:
        self.inner = inner
        self.k = int(k)
        if not 1 <= self.k <= int(inner.channels):
            raise ValueError(
                f"cannot sample {self.k} channels from {inner.name}, which has "
                f"{inner.channels}"
            )
        self.name = f"{inner.name}@{self.k}"
        self.channels = self.k
        self.r_f = int(inner.r_f)
        self.groups = tuple(getattr(inner, "groups", (int(inner.channels),)))
        self.seed = int(seed)
        self.pick = stratified_pick(self.groups, self.k, self.seed, 0, 0)

    def select(self, level: int, iteration: int, seed: int | None = None) -> np.ndarray:
        """Choose the channels for one template pass, and return them."""
        if seed is not None:
            self.seed = int(seed)
        self.pick = stratified_pick(self.groups, self.k, self.seed, level, iteration)
        return self.pick

    def setup(self, device: str | None = None) -> None:
        self.inner.setup(device)

    def __call__(self, patch, spacing_mm: float, mask=None):
        f = self.inner(patch, spacing_mm, mask)
        if _is_tensor(f):
            import torch

            idx = torch.as_tensor(self.pick, device=f.device)
            return f.index_select(0, idx).contiguous()
        return np.ascontiguousarray(np.asarray(f, dtype=np.float32)[self.pick])

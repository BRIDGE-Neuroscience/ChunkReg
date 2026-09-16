"""MIND-SSC: a hand-built multichannel descriptor that needs no network.

Modality Independent Neighbourhood Descriptor with Self-Similarity Context
(Heinrich et al.). Each voxel is described by twelve numbers derived from how
its local patch compares with itself under twelve pairs of small
displacements. Because every number is a ratio of self-comparisons, the
descriptor is invariant to any monotonic intensity change, which is what makes
it modality independent.

It earns its place here for two reasons beyond registration quality. It is the
only multichannel extractor that runs on a CPU with no weights to download, so
the feature-visualisation path can be exercised and judged without a GPU. And
its receptive field is exactly ``dilation + patch radius``, a number that can be
derived rather than measured, which makes it a known-answer test for
``chunkreg setup``.

The twelve channels are the twelve edges of the octahedron formed by the six
axis neighbours: every pair of neighbours lying on different axes.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import ndimage

from .base import normalise_channels

__all__ = ["MindSSCFeatures", "SSC_PAIRS", "SIX_NEIGHBOURS"]

SIX_NEIGHBOURS: tuple[tuple[int, int, int], ...] = (
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
)

# Pairs of neighbours on different axes: the twelve octahedron edges. Pairs on
# the same axis are excluded because comparing a voxel's two opposite
# neighbours measures a second derivative, not a self-similarity.
SSC_PAIRS: tuple[tuple[int, int], ...] = tuple(
    (i, j)
    for i, j in combinations(range(6), 2)
    if (i // 2) != (j // 2)
)
assert len(SSC_PAIRS) == 12


def _shift(vol: np.ndarray, offset, dilation: int) -> np.ndarray:
    """Translate by ``dilation * offset``, replicating at the edge."""
    out = vol
    for axis, step in enumerate(offset):
        if step:
            out = np.roll(out, step * dilation, axis=axis)
            # np.roll wraps; overwrite the wrapped face with the edge value so
            # a descriptor near the boundary is not comparing opposite sides of
            # the volume.
            idx = [slice(None)] * 3
            if step > 0:
                idx[axis] = slice(0, step * dilation)
                edge = [slice(None)] * 3
                edge[axis] = slice(step * dilation, step * dilation + 1)
            else:
                idx[axis] = slice(step * dilation, None)
                edge = [slice(None)] * 3
                edge[axis] = slice(step * dilation - 1, step * dilation)
            out[tuple(idx)] = out[tuple(edge)]
    return out


class MindSSCFeatures:
    """Twelve self-similarity channels, computed on the CPU."""

    name = "mindssc"
    channels = 12

    def __init__(
        self,
        radius: int = 1,
        dilation: int = 2,
        normalisation: str = "l2",
    ) -> None:
        if radius < 0 or dilation < 1:
            raise ValueError("radius must be >= 0 and dilation >= 1")
        self.radius = int(radius)
        self.dilation = int(dilation)
        self.normalisation = normalisation

    @property
    def r_f(self) -> int:
        """Receptive-field radius: the patch reaches ``dilation`` away and the
        patch itself is ``radius`` wide, so nothing outside their sum is read."""
        return self.dilation + self.radius

    def setup(self, device: str | None = None) -> None:
        return None

    def __call__(self, patch, spacing_mm: float, mask=None) -> np.ndarray:
        a = np.asarray(patch, dtype=np.float32)
        if a.ndim != 3:
            raise ValueError(f"expected a (Z, Y, X) patch, got {a.shape}")

        size = 2 * self.radius + 1
        shifted = [_shift(a.copy(), o, self.dilation) for o in SIX_NEIGHBOURS]

        # Patch sum of squared differences for each of the twelve pairs.
        dist = np.empty((12,) + a.shape, dtype=np.float32)
        for c, (i, j) in enumerate(SSC_PAIRS):
            diff = shifted[i] - shifted[j]
            dist[c] = ndimage.uniform_filter(diff * diff, size=size, mode="nearest")

        # The variance estimate is per voxel, so the exponential adapts to local
        # contrast instead of to a global scale. Clipping keeps flat regions,
        # where the mean distance is near zero, from exploding.
        var = dist.mean(axis=0)
        lo, hi = 1e-6, float(np.percentile(var, 99.9)) or 1.0
        var = np.clip(var, lo * max(hi, 1e-6), None)

        out = np.exp(-dist / var[None])
        # Scale so the strongest response at each voxel is one, which removes
        # the remaining dependence on absolute contrast.
        out /= np.maximum(out.max(axis=0, keepdims=True), 1e-8)
        out = normalise_channels(out, self.normalisation)
        if mask is not None:
            out = out * np.asarray(mask, dtype=np.float32)[None]
        return np.ascontiguousarray(out.astype(np.float32))

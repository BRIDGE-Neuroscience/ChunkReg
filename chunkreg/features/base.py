"""The feature-extractor contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["FeatureExtractor", "normalise_channels", "aligned_box"]


@runtime_checkable
class FeatureExtractor(Protocol):
    name: str
    channels: int
    r_f: int
    """Receptive-field radius in voxels. Enters the halo bound, so a wrong
    value costs clamp that the chunk then exceeds without anyone noticing."""

    def setup(self, device: str | None = None) -> None:
        """Load weights. Called once per worker, never per chunk."""

    def __call__(
        self,
        patch: np.ndarray,
        spacing_mm: float,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Map a normalised ``(Z, Y, X)`` patch to ``(C, Z, Y, X)`` channels."""


def aligned_box(extractor, origin, shape) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """The box an extractor wants to see for chunk-independent features over a box.

    An extractor whose features at a voxel depend on how the patch around it
    is tiled (anatomix in sliding windows) widens the box so the tiling can be
    anchored to the grid rather than to the patch; see
    :meth:`chunkreg.features.anatomix.AnatomixFeatures.aligned_box`. Any other
    extractor is a function of position already and returns the box unchanged.
    """
    fn = getattr(extractor, "aligned_box", None)
    if fn is None:
        return tuple(int(o) for o in origin), tuple(int(s) for s in shape)
    return fn(origin, shape)


def normalise_channels(x: np.ndarray, method: str = "l2") -> np.ndarray:
    """Per-voxel normalisation across the channel axis.

    With L2-normalised channels a multichannel local correlation behaves like a
    cosine similarity between feature vectors, which is what lets an
    intensity-era optimiser work on learned features at all.
    """
    if type(x).__module__.split(".", 1)[0] == "torch":
        from ..gpu_ops import normalise_channels as _on_device

        return _on_device(x, method)
    a = np.asarray(x, dtype=np.float32)
    if method == "none":
        return a
    if method == "l2":
        n = np.sqrt(np.sum(a * a, axis=0, keepdims=True, dtype=np.float32))
        return a / np.maximum(n, 1e-6)
    if method == "standardized":
        mean = a.mean(axis=0, keepdims=True)
        std = a.std(axis=0, keepdims=True)
        return (a - mean) / np.maximum(std, 1e-6)
    raise ValueError(
        f"unknown feature normalisation {method!r}; use 'l2', 'standardized' or 'none'"
    )

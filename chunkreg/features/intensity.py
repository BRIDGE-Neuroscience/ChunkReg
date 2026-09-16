"""Raw intensity: the one-channel degenerate case of a feature extractor.

Worth having as a first-class extractor rather than a special case. It has no
receptive field, so at a fixed halo it leaves three times the displacement
clamp that a 24-voxel-receptive-field network does, and it costs a sixteenth of
the registration work. For same-modality data at fine resolution it is often
the right answer, which is what gate G2 exists to decide.
"""

from __future__ import annotations

import numpy as np

__all__ = ["IntensityFeatures"]


class IntensityFeatures:
    name = "intensity"
    channels = 1
    r_f = 0

    def setup(self, device: str | None = None) -> None:
        return None

    def __call__(self, patch, spacing_mm: float, mask=None) -> np.ndarray:
        a = np.asarray(patch, dtype=np.float32)
        if a.ndim != 3:
            raise ValueError(f"expected a (Z, Y, X) patch, got {a.shape}")
        out = a[None]
        if mask is not None:
            out = out * np.asarray(mask, dtype=np.float32)[None]
        return np.ascontiguousarray(out)

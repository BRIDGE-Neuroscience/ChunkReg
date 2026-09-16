"""anatomix features projected to fewer channels.

Registration work is linear in the channel count, so going from sixteen
channels to four cuts the cost of a run fourfold and drops the device class it
needs from 80 GB to 24 GB. A projection fitted once on a sample of feature
vectors keeps most of what the channels encode, because the sixteen anatomix
channels are far from independent.

The projection is fitted once, at the coarsest level, and then frozen. Fitting
it per chunk would be a different basis in every chunk, so features either side
of a chunk boundary would not be comparable and the blend would be merging
solutions to different problems.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .anatomix import AnatomixFeatures
from .base import normalise_channels

__all__ = ["PCAFeatures"]


class PCAFeatures:
    """anatomix features linearly projected onto their leading components."""

    name = "pca"

    def __init__(
        self,
        n_components: int = 8,
        backbone: str = "anatomix",
        normalisation: str = "l2",
        r_f: int = 24,
        **kwargs,
    ) -> None:
        self.channels = int(n_components)
        self.r_f = int(r_f)
        self.normalisation = normalisation
        self._inner = AnatomixFeatures(
            backbone=backbone, normalisation="none", r_f=r_f, **kwargs
        )
        self._basis: np.ndarray | None = None
        self._mean: np.ndarray | None = None

    def setup(self, device: str | None = None) -> None:
        self._inner.setup(device)

    @property
    def fitted(self) -> bool:
        return self._basis is not None

    def fit(self, patches, spacing_mm: float, sample: int = 200_000, seed: int = 0) -> None:
        """Fit the projection on a sample of feature vectors.

        ``patches`` is an iterable of normalised intensity patches, typically
        the coarsest level of each subject. Only a random sample of voxels is
        used: the basis is a property of the feature distribution, and a few
        hundred thousand vectors determine it as well as all of them.
        """
        rng = np.random.default_rng(seed)
        from .. import xp as _xp

        rows = []
        for patch in patches:
            f = _xp.get(self._inner(patch, spacing_mm))
            flat = f.reshape(f.shape[0], -1).T
            take = min(len(flat), max(1, sample // 8))
            idx = rng.choice(len(flat), size=take, replace=False)
            rows.append(flat[idx])
        x = np.concatenate(rows, axis=0).astype(np.float64)
        self._mean = x.mean(axis=0)
        _, _, vt = np.linalg.svd(x - self._mean, full_matrices=False)
        self._basis = vt[: self.channels].astype(np.float32)
        self._mean = self._mean.astype(np.float32)

    def __call__(self, patch, spacing_mm: float, mask=None) -> np.ndarray:
        if self._basis is None:
            raise RuntimeError(
                "the PCA projection has not been fitted. Run "
                "'chunkreg setup' first, or load a saved basis with "
                "PCAFeatures.load(); fitting per chunk would give every chunk "
                "a different basis."
            )
        f = self._inner(patch, spacing_mm)
        from .. import xp as _xp

        if _xp.is_tensor(f):
            import torch

            basis = torch.as_tensor(self._basis, device=f.device)
            mean = torch.as_tensor(self._mean, device=f.device)
            flat = f.reshape(f.shape[0], -1) - mean[:, None]
            out = normalise_channels(
                (basis @ flat).reshape((self.channels,) + tuple(f.shape[1:])),
                self.normalisation,
            )
            if mask is not None:
                out = out * _xp.put(mask)[None]
            return out.contiguous()
        flat = f.reshape(f.shape[0], -1) - self._mean[:, None]
        out = (self._basis @ flat).reshape((self.channels,) + f.shape[1:])
        out = normalise_channels(out, self.normalisation)
        if mask is not None:
            out = out * np.asarray(mask, dtype=np.float32)[None]
        return np.ascontiguousarray(out.astype(np.float32))

    # -- persistence -------------------------------------------------------- #
    def save(self, path) -> None:
        if self._basis is None:
            raise RuntimeError("nothing to save: the projection is not fitted")
        np.savez(
            Path(path), basis=self._basis, mean=self._mean, channels=self.channels
        )

    def load(self, path) -> "PCAFeatures":
        d = np.load(Path(path))
        self._basis = d["basis"].astype(np.float32)
        self._mean = d["mean"].astype(np.float32)
        self.channels = int(d["channels"])
        return self

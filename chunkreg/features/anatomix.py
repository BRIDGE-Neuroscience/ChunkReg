"""anatomix features: a pretrained 3D UNet used as a modality-agnostic encoder.

The network is loaded once per worker and run under ``no_grad`` on each padded
chunk. Features are never written to disk: at sixteen channels in half
precision they are thirty-two bytes per voxel, which at native resolution is
hundreds of gigabytes per subject for something that is cheaper to recompute
than to read back.

Two choices here differ from the anatomix reference pipeline, both because this
is a chunked pipeline rather than a whole-image one:

* **Normalisation is not recomputed here.** anatomix min-max normalises each
  image it is handed. Applied per chunk that would give neighbouring chunks
  different intensity mappings, so their features would disagree in exactly the
  overlaps the blend has to trust. The store supplies a per-volume global
  window instead, and this extractor takes the patch already normalised.
* **Sliding-window overlap is 0.25, not 0.8.** The reference default is tuned
  for one whole image where the cost is paid once. Overlap 0.8 is roughly a
  hundred times the compute of non-overlapping windows, which at a few hundred
  thousand chunk registrations is the difference between a run and a
  non-starter. A padded chunk that fits the window in one pass skips the
  sliding window entirely.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import normalise_channels

__all__ = ["AnatomixFeatures", "load_backbone"]

_BACKBONE_CHANNELS = {
    "anatomix": 16,
    "anatomix-dev": 16,
    "anatomix-dev-vit": 16,
}


def load_backbone(name: str, device: str | None = None):
    """Load a pretrained anatomix backbone, with an actionable error if absent."""
    try:
        import torch
        from anatomix.model.load_from_hf import load_from_hf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "anatomix features need torch and the anatomix package. Install "
            "with: pip install 'chunkreg[gpu]' and "
            "pip install git+https://github.com/neel-dey/anatomix.git\n"
            "Use profile 'i1' (raw intensity) to run without them."
        ) from exc
    model = load_from_hf(name)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class AnatomixFeatures:
    """Feature channels from a pretrained anatomix UNet."""

    name = "anatomix"

    def __init__(
        self,
        backbone: str = "anatomix",
        normalisation: str = "l2",
        window: int = 128,
        overlap: float = 0.25,
        sw_batch: int = 4,
        mode: str = "gaussian",
        sigma: float = 0.25,
        r_f: int = 24,
        device: str | None = None,
    ) -> None:
        if backbone not in _BACKBONE_CHANNELS:
            raise ValueError(
                f"unknown backbone {backbone!r}; known: "
                f"{', '.join(sorted(_BACKBONE_CHANNELS))}"
            )
        self.backbone = backbone
        self.channels = _BACKBONE_CHANNELS[backbone]
        self.normalisation = normalisation
        self.window = int(window)
        self.overlap = float(overlap)
        self.sw_batch = int(sw_batch)
        self.mode = mode
        self.sigma = float(sigma)
        self.r_f = int(r_f)
        self.device = device
        self._model: Any = None

    def setup(self, device: str | None = None) -> None:
        """Load weights once per worker, never once per chunk."""
        if self._model is None:
            self.device = device or self.device
            self._model = load_backbone(self.backbone, self.device)

    def __call__(self, patch, spacing_mm: float, mask=None) -> np.ndarray:
        import torch

        if self._model is None:
            self.setup()
        a = np.asarray(patch, dtype=np.float32)
        if a.ndim != 3:
            raise ValueError(f"expected a (Z, Y, X) patch, got {a.shape}")

        dev = next(self._model.parameters()).device
        x = torch.from_numpy(a)[None, None].to(dev)
        with torch.no_grad():
            if max(a.shape) <= self.window:
                # The common case: a padded chunk sized to fit the device also
                # fits the window, so one forward pass beats any tiling.
                feats = self._model(x)
            else:
                feats = self._sliding(x)
        out = feats[0].float().cpu().numpy()
        out = normalise_channels(out, self.normalisation)
        if mask is not None:
            out = out * np.asarray(mask, dtype=np.float32)[None]
        return np.ascontiguousarray(out.astype(np.float32))

    def _sliding(self, x):
        from monai.inferers import sliding_window_inference

        batch = self.sw_batch
        while True:
            try:
                return sliding_window_inference(
                    x,
                    roi_size=(self.window,) * 3,
                    sw_batch_size=batch,
                    predictor=self._model,
                    overlap=self.overlap,
                    mode=self.mode,
                    sigma_scale=self.sigma,
                )
            except RuntimeError as exc:  # pragma: no cover - needs a GPU
                if "out of memory" not in str(exc).lower() or batch == 1:
                    raise
                batch = max(1, batch // 2)

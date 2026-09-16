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
    """Feature channels from a pretrained anatomix UNet.

    By default a padded chunk goes through the network in one forward pass,
    in bfloat16 on a GPU. Tiling a 352-cubed chunk into overlapping 128-cubed
    windows ran the network over about three times the chunk's voxels and was
    the larger half of a chunk's cost. A chunk too big for the device falls
    back to tiling, and chunks at least that big go straight to tiling after
    the first failure. Set ``window`` to force tiling at that size.
    """

    name = "anatomix"

    def __init__(
        self,
        backbone: str = "anatomix",
        normalisation: str = "l2",
        window: int | None = None,
        overlap: float = 0.25,
        sw_batch: int = 4,
        mode: str = "gaussian",
        sigma: float = 0.25,
        r_f: int = 24,
        device: str | None = None,
        amp: bool = True,
        fallback_window: int = 128,
        pad_multiple: int = 32,
    ) -> None:
        if backbone not in _BACKBONE_CHANNELS:
            raise ValueError(
                f"unknown backbone {backbone!r}; known: "
                f"{', '.join(sorted(_BACKBONE_CHANNELS))}"
            )
        self.backbone = backbone
        self.channels = _BACKBONE_CHANNELS[backbone]
        self.normalisation = normalisation
        self.window = None if window is None else int(window)
        self.overlap = float(overlap)
        self.sw_batch = int(sw_batch)
        self.mode = mode
        self.sigma = float(sigma)
        self.r_f = int(r_f)
        self.device = device
        self.amp = bool(amp)
        self.fallback_window = int(fallback_window)
        self.pad_multiple = int(pad_multiple)
        self._tile_from: int | None = None
        self._model: Any = None

    def setup(self, device: str | None = None) -> None:
        """Load weights once per worker, never once per chunk."""
        if self._model is None:
            from .. import xp as _xp

            self.device = device or self.device
            if self.device is None and _xp.uses_torch():
                self.device = str(_xp.torch_device())
            self._model = load_backbone(self.backbone, self.device)

    def __call__(self, patch, spacing_mm: float, mask=None) -> np.ndarray:
        import torch

        from .. import xp as _xp

        if self._model is None:
            self.setup()
        if _xp.is_tensor(patch):
            src = patch.to(torch.float32)
        else:
            src = torch.from_numpy(np.ascontiguousarray(patch, dtype=np.float32))
        if src.ndim != 3:
            raise ValueError(f"expected a (Z, Y, X) patch, got {tuple(src.shape)}")

        dev = next(self._model.parameters()).device
        x = src[None, None].to(dev)
        with torch.no_grad(), self._autocast(dev):
            feats = self._features(x)
        # Normalised and masked on the device, in full precision. The features
        # only come back to the host when this process computes there.
        out = normalise_channels(feats[0].float(), self.normalisation)
        if mask is not None:
            m = mask if _xp.is_tensor(mask) else torch.from_numpy(np.asarray(mask, np.float32))
            out = out * m.to(out.device, torch.float32)[None]
        if _xp.uses_torch():
            return out.contiguous()
        return np.ascontiguousarray(out.cpu().numpy().astype(np.float32))

    # -- forward passes ----------------------------------------------------- #
    def _autocast(self, dev):
        import contextlib

        import torch

        if self.amp and dev.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _features(self, x):
        import torch

        size = int(x[0, 0].numel())
        if self.window is not None and max(x.shape[2:]) > self.window:
            return self._sliding(x, self.window)
        if self._tile_from is not None and size >= self._tile_from:
            return self._sliding(x, self.fallback_window)
        try:
            return self._whole(x)
        except RuntimeError as exc:  # pragma: no cover - needs a GPU
            if "out of memory" not in str(exc).lower():
                raise
            if x.device.type == "cuda":
                torch.cuda.empty_cache()
            self._tile_from = size
            return self._sliding(x, self.fallback_window)

    def _whole(self, x):
        """One forward pass, padded so every downsampling divides evenly."""
        import torch.nn.functional as F

        shape = [int(n) for n in x.shape[2:]]
        m = self.pad_multiple
        pad = [(-n) % m for n in shape]
        if any(pad):
            # Constant zero, as the sliding-window path pads a volume's edge.
            x = F.pad(x, [0, pad[2], 0, pad[1], 0, pad[0]])
        out = self._model(x)
        return out[..., : shape[0], : shape[1], : shape[2]]

    def _sliding(self, x, window: int):
        from monai.inferers import sliding_window_inference

        batch = self.sw_batch
        while True:
            try:
                return sliding_window_inference(
                    x,
                    roi_size=(window,) * 3,
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

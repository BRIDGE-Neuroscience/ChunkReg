"""The production engine: FireANTs on multichannel feature stacks.

FireANTs is a GPU Riemannian optimiser over diffeomorphisms. This adapter hands
it one chunk at a time and converts what it returns back into the package's
conventions.

Two adapter details carry most of the risk, and both are pinned by the
convention selftest rather than trusted:

* **Feature stacks are not images.** FireANTs builds an ``Image`` from an ITK
  object, which would mean writing sixteen channels through SimpleITK for every
  chunk. Instead a one-channel ``Image`` carries the chunk's geometry and
  ``FakeBatchedImages`` substitutes the real feature tensor, which is the same
  route the anatomix reference pipeline takes.
* **The warp comes back as a sampling grid**, normalised to ``[-1, 1]`` with
  ``align_corners=True`` and with its last axis ordered ``(x, y, z)``. Both the
  normalisation and the axis reversal are undone in one place, in
  :func:`chunkreg.fields.grid_to_disp_mm`.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .. import fields as _fields
from .base import RegResult, check_inputs, check_stages

__all__ = ["FireAntsEngine"]

_INSTALL_HINT = (
    "The FireANTs backend is not installed. Install it with the script "
    "anatomix ships:\n"
    "    bash anatomix/registration/registration_backend/install_fireants.sh\n"
    "which pins the neel-dey/FireANTs fork and builds the fused CUDA ops. "
    "Use engine 'demons' to run on CPU without it."
)


def _imports():
    try:
        import torch
        from fireants.io.image import BatchedImages, FakeBatchedImages, Image
        from fireants.registration.affine import AffineRegistration
        from fireants.registration.greedy import GreedyRegistration
        from fireants.registration.moments import MomentsRegistration
        from fireants.registration.rigid import RigidRegistration
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(f"{_INSTALL_HINT}\n(underlying error: {exc})") from exc
    try:
        from fireants.registration.syn import SyNRegistration
    except ImportError:
        SyNRegistration = None
    return dict(
        torch=torch,
        Image=Image,
        BatchedImages=BatchedImages,
        FakeBatchedImages=FakeBatchedImages,
        Moments=MomentsRegistration,
        Rigid=RigidRegistration,
        Affine=AffineRegistration,
        Greedy=GreedyRegistration,
        SyN=SyNRegistration,
    )


class FireAntsEngine:
    """Multi-stage registration of feature stacks on a GPU."""

    name = "fireants"
    supports = frozenset({"moments", "rigid", "affine", "greedy", "syn"})

    def __init__(self, device: str | None = None, progress: bool = False) -> None:
        self.device = device
        self.progress = bool(progress)
        self._api: dict[str, Any] | None = None

    def _lazy(self) -> dict[str, Any]:
        if self._api is None:
            self._api = _imports()
        return self._api

    # -- geometry ----------------------------------------------------------- #
    def _geometry(self, shape, spacing_mm: float, device):
        """A one-channel Image carrying only the chunk's grid.

        The chunk is isotropic and axis aligned, so spacing is uniform and the
        direction is identity. The array it holds is a placeholder; the real
        channels are substituted by FakeBatchedImages.
        """
        api = self._lazy()
        import SimpleITK as sitk

        # SimpleITK indexes (x, y, z), the reverse of the array order.
        itk = sitk.GetImageFromArray(np.zeros(tuple(shape), dtype=np.float32))
        itk.SetSpacing((float(spacing_mm),) * 3)
        try:
            img = api["Image"](itk, device=device)
        except TypeError:
            img = api["Image"](itk)
        return api["BatchedImages"]([img])

    def _wrap(self, batch, stack, device):
        api = self._lazy()
        torch = api["torch"]
        t = torch.as_tensor(np.ascontiguousarray(stack), dtype=torch.float32)
        return api["FakeBatchedImages"](t[None].to(device), batch)

    # -- stages ------------------------------------------------------------- #
    def register(
        self,
        fixed: np.ndarray,
        moving: np.ndarray,
        spacing_mm: float,
        stages: Sequence,
        init_affine: np.ndarray | None = None,
        device: str | None = None,
    ) -> RegResult:
        f, m = check_inputs(fixed, moving)
        check_stages(stages, self.supports, self.name)
        api = self._lazy()
        torch = api["torch"]
        dev = device or self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        shape = tuple(f.shape[1:])
        geom = self._geometry(shape, spacing_mm, dev)
        fixed_b = self._wrap(geom, f, dev)
        moving_b = self._wrap(geom, m, dev)

        affine = init_affine
        grid = None
        iters: list[int] = []
        curve: list[float] = []

        for stage in stages:
            kind = getattr(stage, "kind", stage)
            common = dict(
                fixed_images=fixed_b,
                moving_images=moving_b,
                loss_type=getattr(stage, "loss", "cc"),
                cc_kernel_size=getattr(stage, "cc_kernel", 7),
                optimizer="Adam",
                optimizer_lr=getattr(stage, "lr", 0.5),
                progress_bar=self.progress,
                tolerance=getattr(stage, "tolerance", 1e-6),
            )
            if kind != "moments":
                common["scales"] = list(getattr(stage, "scales", (1,)))
                common["iterations"] = list(getattr(stage, "iterations", (50,)))

            if kind == "moments":
                reg = api["Moments"](fixed_images=fixed_b, moving_images=moving_b)
            elif kind in ("rigid", "affine"):
                cls = api["Rigid"] if kind == "rigid" else api["Affine"]
                reg = cls(
                    **common,
                    init_moment=affine,
                    normalize_translation=True,
                    translation_lr=getattr(stage, "translation_lr", None)
                    or getattr(stage, "lr", 0.5),
                )
            else:
                cls = api["Greedy"] if kind == "greedy" else api["SyN"]
                if cls is None:
                    raise ImportError(
                        "this FireANTs build has no SyNRegistration; use "
                        "a 'greedy' stage instead"
                    )
                reg = cls(
                    **common,
                    deformation_type="compositive",
                    smooth_grad_sigma=getattr(stage, "smooth_grad_sigma", 1.0),
                    smooth_warp_sigma=getattr(stage, "smooth_warp_sigma", 0.5),
                    init_affine=affine,
                )
            reg.optimize()

            if kind in ("moments", "rigid", "affine"):
                affine = _linear_of(reg)
            else:
                grid = _grid_of(reg)
            iters.extend(_iters_of(reg, stage))
            curve.extend(_loss_of(reg))

        if grid is None:
            raise ValueError(
                "no deformable stage ran, so there is no displacement field to "
                "return. Add a 'greedy' or 'syn' stage."
            )
        disp = _fields.grid_to_disp_mm(
            grid.detach().cpu().numpy(), shape, spacing_mm
        )
        return RegResult(
            disp_mm=disp,
            loss_curve=curve,
            iters_per_scale=iters,
            converged=_converged(curve),
            affine=None if affine is None else np.asarray(affine),
        )


def _grid_of(reg):
    """The sampling grid a deformable stage converged to."""
    for getter in ("get_warped_coordinates", "get_warp_field", "get_sample_grid"):
        fn = getattr(reg, getter, None)
        if callable(fn):
            try:
                return fn()
            except TypeError:
                return fn(reg.fixed_images, reg.moving_images)
    if hasattr(reg, "warp"):
        return reg.warp
    raise RuntimeError(
        "could not obtain a sampling grid from this FireANTs build; inspect "
        "dir(reg) and extend chunkreg.engines.fireants._grid_of. Run "
        "'chunkreg selftest' to confirm the convention once you have."
    )


def _linear_of(reg):
    for attr in ("get_affine_matrix", "get_matrix", "affine"):
        got = getattr(reg, attr, None)
        if callable(got):
            return got()
        if got is not None:
            return got
    return None


def _iters_of(reg, stage) -> list[int]:
    got = getattr(reg, "iterations_run", None)
    if got:
        return list(got)
    # No early-stop record available: report the caps, which is the
    # conservative reading for the histogram that tunes the next run's caps.
    return list(getattr(stage, "iterations", ()) or [])


def _loss_of(reg) -> list[float]:
    for attr in ("loss_history", "losses", "loss_curve"):
        got = getattr(reg, attr, None)
        if got:
            return [float(x) for x in got]
    return []


def _converged(curve: Sequence[float]) -> bool:
    """A loss still descending at exit means the iteration cap bound."""
    if len(curve) < 6:
        return True
    tail = np.asarray(curve[-5:], dtype=np.float64)
    span = abs(curve[0] - curve[-1]) or 1.0
    return bool(abs(tail[0] - tail[-1]) / span < 0.01)

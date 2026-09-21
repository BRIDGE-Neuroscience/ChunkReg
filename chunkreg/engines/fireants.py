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

import logging
import os
from typing import Any, Sequence

import numpy as np

from .. import fields as _fields
from .. import xp as _xp
from .base import RegResult, check_inputs, check_stages

__all__ = ["FireAntsEngine", "quieten"]


def quieten() -> None:
    """Stop FireANTs narrating every iteration of every chunk.

    FireANTs reports through a tqdm bar per scale per stage and an INFO line
    per stage. That is useful watching one registration and useless inside a
    run: a pass at a chunked level is hundreds of registrations, tqdm redraws
    per iteration, and the scheduler merges stderr into the job log. The
    result was thousands of lines a minute, in which the one line per pass
    that says what the run is actually doing cannot be found.

    Both are silenced rather than filtered afterwards, because a log that has
    to be grepped to be read has already cost the thing it was for. Set
    ``CHUNKREG_PROGRESS=1``, or build the engine with ``progress=True``, to
    watch a single registration.
    """
    logging.getLogger("fireants").setLevel(logging.WARNING)
    try:
        import functools

        import tqdm
        import tqdm.auto
    except ImportError:  # pragma: no cover - tqdm ships with the GPU extras
        return
    # tqdm has no global switch, so the classes FireANTs might reach for are
    # given a disabled default. Idempotent: workers build an engine per level.
    for module in (tqdm, tqdm.auto):
        cls = getattr(module, "tqdm", None)
        if cls is None or getattr(cls, "_chunkreg_quiet", False):
            continue
        cls.__init__ = functools.partialmethod(cls.__init__, disable=True)
        cls._chunkreg_quiet = True

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

    def __init__(self, device: str | None = None, progress: bool | None = None) -> None:
        self.device = device
        if progress is None:
            progress = os.environ.get("CHUNKREG_PROGRESS", "").strip().lower() in (
                "1", "true", "yes", "on",
            )
        self.progress = bool(progress)
        self._api: dict[str, Any] | None = None
        self._geometries: dict[tuple, Any] = {}

    def _lazy(self) -> dict[str, Any]:
        if self._api is None:
            self._api = _imports()
            if not self.progress:
                quieten()
        return self._api

    # -- geometry ----------------------------------------------------------- #
    def _geometry(self, shape, spacing_mm: float, device):
        """A one-channel Image carrying only the chunk's grid.

        The chunk is isotropic and axis aligned, so spacing is uniform and the
        direction is identity. The array it holds is a placeholder; the real
        channels are substituted by FakeBatchedImages.

        Built once per chunk shape and kept: almost every chunk of a level has
        the same padded shape, and building one allocates a placeholder volume
        on the host and copies it to the device.
        """
        key = (tuple(int(n) for n in shape), float(spacing_mm), str(device))
        cached = self._geometries.get(key)
        if cached is not None:
            return cached
        if len(self._geometries) > 16:
            self._geometries.clear()
        self._geometries[key] = geom = self._build_geometry(shape, spacing_mm, device)
        return geom

    def _build_geometry(self, shape, spacing_mm: float, device):
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

    @staticmethod
    def _stage_plan(api, kind: str, stage, common: dict, carry) -> tuple[Any, dict]:
        """The class and the exact arguments one stage is built from.

        Kept separate from :meth:`register` and free of tensors so that a
        build's real signatures can be checked against it directly. Twice now a
        mismatch here has been found only by running on a cluster, and both
        times ``**kwargs`` meant the wrong name was accepted rather than
        refused.

        ``carry`` is whatever the preceding linear stage left behind: a
        ``MomentsRegistration``, which hands the next solver its own init
        arguments, or the matrix a rigid or affine stage converged to.
        """
        if kind == "moments":
            # Moments runs no optimiser, so it takes none of the schedule.
            kwargs = {
                k: common[k] for k in ("fixed_images", "moving_images") if k in common
            }
            kwargs.update(FireAntsEngine._moments_scale(api["Moments"], stage))
            return api["Moments"], kwargs

        if kind in ("rigid", "affine"):
            cls = api["Rigid"] if kind == "rigid" else api["Affine"]
            return cls, dict(
                **common,
                **_linear_init(kind, carry),
                normalize_translation=True,
                translation_lr=getattr(stage, "translation_lr", None)
                or getattr(stage, "lr", 0.5),
            )

        cls = api["Greedy"] if kind == "greedy" else api["SyN"]
        if cls is None:
            raise ImportError(
                "this FireANTs build has no SyNRegistration; use a 'greedy' "
                "stage instead"
            )
        return cls, dict(
            **common,
            deformation_type="compositive",
            smooth_grad_sigma=getattr(stage, "smooth_grad_sigma", 1.0),
            smooth_warp_sigma=getattr(stage, "smooth_warp_sigma", 0.5),
            init_affine=_affine_matrix(api, carry),
        )

    @staticmethod
    def _moments_scale(cls, stage) -> dict:
        """``scale`` for a moments stage, when the build requires one.

        Some builds take the scale the moments are computed at and some do not,
        so it is passed only when the signature asks. It is a downsampling
        factor, and moments are a global statistic -- a centre of mass and a
        second-moment match -- so it changes what the estimate costs rather
        than what it is. The stage's finest scale is used, 1 by default: the
        one value that means the same thing however a build reads it. A moments
        stage only runs at level 0, where the volume is a single chunk.
        """
        import inspect

        try:
            params = inspect.signature(cls.__init__).parameters
        except (TypeError, ValueError):  # pragma: no cover - exotic builds
            return {}
        scale = params.get("scale")
        if scale is None or scale.default is not inspect.Parameter.empty:
            return {}
        return {"scale": int(min(getattr(stage, "scales", None) or (1,)))}

    def _wrap(self, batch, stack, device):
        api = self._lazy()
        torch = api["torch"]
        if _xp.is_tensor(stack):
            t = stack.to(device=device, dtype=torch.float32)
        else:
            t = torch.as_tensor(np.ascontiguousarray(stack), dtype=torch.float32).to(device)
        return api["FakeBatchedImages"](t[None], batch)

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
        dev = device or self.device
        if dev is None:
            dev = (
                str(_xp.torch_device())
                if _xp.uses_torch()
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )

        shape = tuple(f.shape[1:])
        geom = self._geometry(shape, spacing_mm, dev)
        fixed_b = self._wrap(geom, f, dev)
        moving_b = self._wrap(geom, m, dev)

        carry = init_affine
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
                tolerance=getattr(stage, "tolerance", 1e-6),
            )
            if kind != "moments":
                common["scales"] = list(getattr(stage, "scales", (1,)))
                common["iterations"] = list(getattr(stage, "iterations", (50,)))

            cls, kwargs = self._stage_plan(api, kind, stage, common, carry)
            if "progress_bar" in _declared(cls):
                kwargs["progress_bar"] = self.progress
            reg = _construct(cls, kwargs, kind)
            reg.optimize()

            if kind in ("moments", "rigid", "affine"):
                carry = _carry_of(reg, kind)
            else:
                grid = _grid_of(reg)
            iters.extend(_iters_of(reg, stage))
            curve.extend(_loss_of(reg))

        if grid is None:
            raise ValueError(
                "no deformable stage ran, so there is no displacement field to "
                "return. Add a 'greedy' or 'syn' stage."
            )
        # Converted on the device; it only leaves if this process computes on
        # the host.
        disp = _fields.grid_to_disp_mm(grid.detach().to(dev), shape, spacing_mm)
        if not _xp.uses_torch():
            disp = _xp.get(disp)
        return RegResult(
            disp_mm=disp,
            loss_curve=curve,
            iters_per_scale=iters,
            converged=_converged(curve),
            affine=None if not _is_matrix(carry) else _xp.get(carry),
        )


def _declared(cls) -> set[str]:
    """Every argument name this class accepts, including its parents'.

    A FireANTs stage class lists its own arguments and forwards ``**kwargs`` up
    to ``AbstractRegistration``, so a name missing from the subclass may still
    be a real argument one level up. Checking only the subclass declared
    ``progress_bar`` an unknown and dropped it, which is how a run ended up
    narrating every iteration of every chunk into the job log.
    """
    import inspect

    names: set[str] = set()
    for base in inspect.getmro(cls):
        init = base.__dict__.get("__init__")
        if init is None:
            continue
        try:
            params = inspect.signature(init).parameters
        except (TypeError, ValueError):  # pragma: no cover - builtins
            continue
        names |= {
            n
            for n, p in params.items()
            if n != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
        }
    return names


def _construct(cls, kwargs: dict, kind: str):
    """Build a FireANTs registration, checking the arguments first.

    Every FireANTs stage class ends its signature with ``**kwargs``. That makes
    a wrong argument *name* silent: the value is accepted, swallowed and never
    used, and the stage runs as though it had never been given. Passing an
    affine's initialiser as ``init_moment`` instead of ``init_rigid`` does not
    fail, it just starts the affine from nothing -- which converges somewhere
    plausible and wrong, at the one level where the global component is
    estimated.

    So the signature is checked before the call rather than after it. Anything
    the adapter means to pass has to be a parameter the build actually
    declares, and anything the build requires has to be something the adapter
    passes. This adapter is written against one documented API and FireANTs is
    a moving fork, so the check states which stage and which parameter.
    """
    import inspect

    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic builds
        return cls(**kwargs)

    declared = _declared(cls)
    swallows = any(p.kind is p.VAR_KEYWORD for p in params.values())
    unknown = [n for n in kwargs if n not in declared]
    required = [
        n
        for n, p in params.items()
        if n != "self"
        and p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    missing = [n for n in required if n not in kwargs]

    if unknown or missing:
        fate = (
            "accepted by **kwargs and silently ignored"
            if swallows
            else "rejected outright"
        )
        raise TypeError(
            f"this FireANTs build's {cls.__name__} does not match the adapter "
            f"for the {kind!r} stage.\n"
            f"  it requires:      {', '.join(required) or '(nothing)'}\n"
            f"  adapter omitted:  {', '.join(missing) or '(nothing)'}\n"
            f"  build rejects:    {', '.join(unknown) or '(nothing)'}"
            + (f"  [{fate}]" if unknown else "")
            + f"\nAdjust chunkreg/engines/fireants.py to match this build, then "
            f"re-run 'chunkreg selftest --engine fireants'."
        )
    return cls(**kwargs)


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


_MOMENTS_INIT = {
    "rigid": "get_rigid_init_dict",
    "affine": "get_affine_init_dict",
}
"""How a moments stage hands itself to the solver that follows it.

FireANTs gives ``MomentsRegistration`` one accessor per downstream solver
because the solvers want different things: rigid takes the rotation as
``[N, D, D]`` and the translation as ``[N, D]``, separately. The ``[N, D, D+1]``
affine that moments also exposes is not interchangeable with either, and
passing it as ``init_moment`` fails inside ``get_rotation_matrix``.
"""

_LINEAR_RESULT = {
    "rigid": "get_rigid_matrix",
    "affine": "get_affine_matrix",
}
"""The converged transform of a linear stage, under the names anatomix uses."""


def _is_matrix(value) -> bool:
    """A transform, as opposed to a registration object carrying one."""
    return value is not None and hasattr(value, "shape")


def _carry_of(reg, kind: str):
    """What a linear stage leaves for the stage after it.

    A rigid or affine stage leaves its matrix. A moments stage leaves *itself*,
    because what the next solver needs depends on which solver it is, and only
    the moments object can produce it.

    A miss is raised rather than returned as ``None``. Silently dropping it
    does not crash: the next stage simply starts from nothing and converges
    somewhere plausible, at the one level where the global component is
    estimated, so nothing downstream can recover it.
    """
    if kind == "moments":
        if not any(hasattr(reg, n) for n in (*_MOMENTS_INIT.values(), "get_affine_init")):
            raise RuntimeError(
                f"this FireANTs build's {type(reg).__name__} exposes none of "
                f"{', '.join((*_MOMENTS_INIT.values(), 'get_affine_init'))}, so "
                f"the stages after it would start from nothing.\n"
                f"  it does have: {_relevant(reg)}\n"
                f"Add the right one to chunkreg.engines.fireants._MOMENTS_INIT."
            )
        return reg

    getter = getattr(reg, _LINEAR_RESULT[kind], None)
    if getter is None:
        raise RuntimeError(
            f"the {kind!r} stage ran, but this FireANTs build's "
            f"{type(reg).__name__} has no {_LINEAR_RESULT[kind]}(), so every "
            f"stage after it would start from nothing and this level would "
            f"lose its global alignment.\n"
            f"  it does have: {_relevant(reg)}\n"
            f"Add the right one to chunkreg.engines.fireants._LINEAR_RESULT, "
            f"then re-run 'chunkreg selftest --engine fireants'."
        )
    return getter()


def _relevant(reg) -> str:
    words = ("affine", "matrix", "rigid", "moment", "transform", "init")
    found = sorted(
        a for a in dir(reg)
        if not a.startswith("_") and any(w in a.lower() for w in words)
    )
    return ", ".join(found) or "(nothing that looks relevant)"


def _linear_init(kind: str, carry) -> dict:
    """How a linear stage is initialised from whatever preceded it."""
    if carry is None:
        return {}
    getter = getattr(carry, _MOMENTS_INIT[kind], None)
    if getter is not None:
        return dict(getter())
    return {"init_rigid": carry} if kind == "affine" else {"init_moment": carry}


def _affine_matrix(api, carry):
    """A homogeneous ``[N, D+1, D+1]`` map for a deformable stage to start from.

    A rigid or affine stage already produced one. A moments stage produces the
    ``[D, D+1]`` top block instead, so it is completed to a square matrix the
    same way anatomix completes it.
    """
    if carry is None or _is_matrix(carry):
        return carry
    init = getattr(carry, "get_affine_init", None)
    if init is None:
        return None
    torch = api["torch"]
    block = init().detach()
    if block.ndim == 2:
        block = block.unsqueeze(0)
    n, d = block.shape[0], block.shape[-1] - 1
    out = torch.eye(d + 1, device=block.device, dtype=block.dtype)
    out = out.unsqueeze(0).repeat(n, 1, 1)
    out[:, :d] = block
    return out


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

"""A CPU reference engine: multi-scale diffeomorphic demons.

This exists so the pipeline is runnable and testable without a GPU, a CUDA
toolkit or a FireANTs build. It is not the production engine and does not try
to be: it is deterministic, dependency-free beyond numpy and scipy, and good
enough to recover the smooth deformations the architecture tests apply.

Method. At each pyramid scale, iterate the symmetric demons update

    v = (F - M_w) * grad(M_w) / (|grad(M_w)|^2 + (F - M_w)^2 / K^2)

summed over feature channels, smooth it by ``sigma_g``, compose it onto the
running displacement, and smooth the total by ``sigma_w``. Smoothing the update
is the fluid regulariser and smoothing the total is the elastic one; together
they are what keeps the result close to diffeomorphic without an explicit
exponential map.

Local correlation is handled by normalising both images by their local mean and
standard deviation before taking the demons force. Matching locally normalised
images by sum of squares is the standard first-order stand-in for maximising
local correlation, and it costs two box filters rather than a full LNCC
gradient.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import ndimage

from .. import fields as _fields
from .base import RegResult, check_inputs, check_stages

__all__ = ["DemonsEngine"]


def _downsample(a: np.ndarray, factor: int) -> np.ndarray:
    """Anti-aliased decimation of a ``(C, Z, Y, X)`` stack."""
    if factor == 1:
        return a
    smoothed = ndimage.gaussian_filter(
        a, sigma=(0,) + (0.5 * factor,) * 3, mode="nearest"
    )
    return np.ascontiguousarray(smoothed[:, ::factor, ::factor, ::factor])


def _local_normalise(a: np.ndarray, radius: int) -> np.ndarray:
    """Subtract the local mean and divide by the local standard deviation."""
    if radius < 1:
        return a
    size = (1,) + (2 * radius + 1,) * 3
    mean = ndimage.uniform_filter(a, size=size, mode="nearest")
    sq = ndimage.uniform_filter(a * a, size=size, mode="nearest")
    var = np.maximum(sq - mean * mean, 0.0)
    return (a - mean) / np.sqrt(var + 1e-6)


def _smooth_field(u: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return u
    return ndimage.gaussian_filter(u, sigma=(0, sigma, sigma, sigma), mode="nearest")


class DemonsEngine:
    """Multi-scale demons on feature channels. CPU, deterministic."""

    name = "demons"
    supports = frozenset({"moments", "greedy"})

    def __init__(self, max_step_vox: float = 2.0, k_factor: float = 1.0) -> None:
        self.max_step_vox = float(max_step_vox)
        self.k_factor = float(k_factor)

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
        if init_affine is not None:
            raise NotImplementedError(
                "the demons engine takes no initial affine; seed it with a "
                "displacement field instead"
            )

        shape = f.shape[1:]
        total = np.zeros((3,) + shape, dtype=np.float32)
        loss_curve: list[float] = []
        iters: list[int] = []
        converged = True

        for stage in stages:
            kind = getattr(stage, "kind", stage)
            if kind == "moments":
                total = _fields.compose(
                    self._moments(f, m, total, spacing_mm), total, spacing_mm
                )
                iters.append(1)
                continue
            total, curve, stage_iters, ok = self._greedy(
                f, m, total, spacing_mm, stage
            )
            loss_curve.extend(curve)
            iters.extend(stage_iters)
            converged = converged and ok

        return RegResult(
            disp_mm=total,
            loss_curve=loss_curve,
            iters_per_scale=iters,
            converged=converged,
        )

    def _moments(
        self, f: np.ndarray, m: np.ndarray, current: np.ndarray, spacing_mm: float
    ) -> np.ndarray:
        """Centre-of-mass translation, as a constant displacement field."""
        warped = self._warp_stack(m, current, spacing_mm)
        fw = np.clip(f.sum(axis=0), 0, None)
        mw = np.clip(warped.sum(axis=0), 0, None)
        if fw.sum() <= 0 or mw.sum() <= 0:
            return np.zeros_like(current)
        com_f = np.asarray(ndimage.center_of_mass(fw), dtype=np.float32)
        com_m = np.asarray(ndimage.center_of_mass(mw), dtype=np.float32)
        shift = (com_m - com_f) * np.float32(spacing_mm)
        out = np.zeros_like(current)
        for d in range(3):
            out[d] = shift[d]
        return out

    def _greedy(
        self,
        f: np.ndarray,
        m: np.ndarray,
        total: np.ndarray,
        spacing_mm: float,
        stage,
    ) -> tuple[np.ndarray, list[float], list[int], bool]:
        scales = tuple(getattr(stage, "scales", (1,)))
        iterations = tuple(getattr(stage, "iterations", (50,)))
        sigma_g = float(getattr(stage, "smooth_grad_sigma", 1.0))
        sigma_w = float(getattr(stage, "smooth_warp_sigma", 0.5))
        tol = float(getattr(stage, "tolerance", 1e-6))
        loss_type = getattr(stage, "loss", "cc")
        radius = max(1, int(getattr(stage, "cc_kernel", 7)) // 2)

        curve: list[float] = []
        achieved: list[int] = []
        converged = True
        full_shape = f.shape[1:]

        for scale, n_iter in zip(scales, iterations):
            fs = _downsample(f, scale)
            ms = _downsample(m, scale)
            if loss_type in ("cc", "masked_cc"):
                fs = _local_normalise(fs, radius)
                ms_norm = _local_normalise(ms, radius)
            else:
                ms_norm = ms
            sub_shape = fs.shape[1:]
            sub_spacing = spacing_mm * scale

            u = self._to_shape(total, sub_shape)
            fgrad = self._gradients(fs, sub_spacing)

            prev = np.inf
            used = 0
            for it in range(int(n_iter)):
                warped = self._warp_stack(ms_norm, u, sub_spacing)
                diff = fs - warped
                loss = float(np.mean(diff * diff))
                curve.append(loss)
                used = it + 1

                # Symmetric forces: averaging the fixed and warped-moving
                # gradients keeps the update well defined where either image is
                # locally flat, which plain Thirion forces are not.
                g = 0.5 * (fgrad + self._gradients(warped, sub_spacing))

                gsq = np.sum(g * g, axis=(0, 1))
                dsq = np.sum(diff * diff, axis=0)
                # The demons denominator. Gradients are per millimetre, so the
                # quotient is already a displacement in millimetres and must
                # not be rescaled by the spacing again.
                #
                # The residual term carries a length scale, and it has to be
                # the one that makes it comparable to the gradient term. With
                # a bare 1/k^2 the term is orders of magnitude smaller than
                # |g|^2, so it never regularises anything, and wherever the
                # image is locally flat the denominator collapses and the step
                # diverges. Dividing by the step length squared puts both terms
                # in the same units and makes the update self-limiting as the
                # gradient vanishes.
                sigma_x = self.max_step_vox * sub_spacing * self.k_factor
                denom = gsq + dsq / (sigma_x**2)
                floor = 1e-6 * float(gsq.mean()) + 1e-12
                step = np.einsum("cdzyx,czyx->dzyx", g, diff) / np.maximum(denom, floor)
                # Regions with no local structure carry no information about
                # where anything moved; leave them to the regulariser.
                step *= (gsq > 1e-3 * float(gsq.mean())).astype(np.float32)
                step = self._limit(step, self.max_step_vox * sub_spacing)
                step = _smooth_field(step, sigma_g)

                u = _smooth_field(_fields.compose(step, u, sub_spacing), sigma_w)

                if prev - loss < tol * max(abs(prev), 1e-12) and it > 2:
                    break
                prev = loss

            achieved.append(used)
            if used >= int(n_iter) and int(n_iter) > 3:
                converged = False
            total = self._to_shape(u, full_shape)

        return total.astype(np.float32, copy=False), curve, achieved, converged

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _gradients(stack: np.ndarray, spacing_mm: float) -> np.ndarray:
        """Spatial gradients of a ``(C, Z, Y, X)`` stack, as ``(C, 3, Z, Y, X)``.

        Differentiated with respect to world millimetres, which is what makes
        the demons quotient a displacement in millimetres.
        """
        return np.stack(
            [
                np.stack(np.gradient(stack[c], spacing_mm), axis=0)
                for c in range(stack.shape[0])
            ],
            axis=0,
        ).astype(np.float32, copy=False)

    @staticmethod
    def _warp_stack(stack: np.ndarray, u: np.ndarray, spacing_mm: float) -> np.ndarray:
        return np.stack(
            [_fields.warp(stack[c], u, spacing_mm) for c in range(stack.shape[0])],
            axis=0,
        )

    @staticmethod
    def _limit(u: np.ndarray, max_mm: float) -> np.ndarray:
        return _fields.clamp(u, max_mm)

    @staticmethod
    def _to_shape(u: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        """Resample a field between in-chunk pyramid scales.

        Vectors are in millimetres, so only the lattice changes; this is the
        same property that makes promoting a seed between pyramid levels exact.
        """
        if tuple(u.shape[1:]) == tuple(shape):
            return u
        zoom = [1.0] + [s / n for s, n in zip(shape, u.shape[1:])]
        return ndimage.zoom(u, zoom, order=1, mode="nearest").astype(
            np.float32, copy=False
        )

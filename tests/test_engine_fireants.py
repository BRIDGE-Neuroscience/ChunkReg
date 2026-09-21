"""The FireANTs adapter's build-compatibility layer.

FireANTs is a moving fork, and its stage classes do not take the same
arguments from one build to the next: the fork pinned by anatomix's installer
requires the scale a moments stage is computed at, and the API this adapter
was written against does not. The adapter therefore decides what to pass by
looking at the signature in front of it.

These tests need neither a GPU nor FireANTs. They drive the introspection
against stand-in classes with each signature, which is the part that has to be
right; whether FireANTs then does the arithmetic correctly is what
``chunkreg selftest --engine fireants`` is for.
"""

from __future__ import annotations

import pytest

from chunkreg.config import StageSpec
from chunkreg.engines.fireants import FireAntsEngine, _construct


class _ScaleRequired:
    """The fork pinned by install_fireants.sh."""

    def __init__(self, scale, fixed_images, moving_images, moments=1):
        self.scale = scale


class _NoScale:
    """The API the adapter was originally written against."""

    def __init__(self, fixed_images, moving_images, moments=1):
        self.scale = None


class _ScaleOptional:
    def __init__(self, fixed_images, moving_images, scale=1):
        self.scale = scale


def _kwargs(cls, stage):
    _, kwargs = FireAntsEngine._stage_plan(
        {"Moments": cls}, "moments", stage,
        {"fixed_images": "F", "moving_images": "M"}, None,
    )
    return kwargs


def test_the_scale_is_passed_only_to_a_build_that_requires_it():
    stage = StageSpec(kind="moments")
    assert "scale" in _kwargs(_ScaleRequired, stage)
    assert "scale" not in _kwargs(_NoScale, stage)
    # An optional one is left alone: the build's own default is a better guess
    # than the adapter's.
    assert "scale" not in _kwargs(_ScaleOptional, stage)


def test_the_moments_scale_is_the_stages_finest():
    """1 by default, which means the same thing however a build reads it."""
    assert _kwargs(_ScaleRequired, StageSpec(kind="moments"))["scale"] == 1
    assert _kwargs(_ScaleRequired, StageSpec(kind="moments", scales=(4, 2)))["scale"] == 2


def test_a_moments_stage_is_built_on_either_signature():
    stage = StageSpec(kind="moments")
    for cls in (_ScaleRequired, _NoScale, _ScaleOptional):
        assert isinstance(_construct(cls, _kwargs(cls, stage), "moments"), cls)


def test_a_signature_mismatch_names_the_stage_and_what_is_missing():
    """A bare TypeError from inside __init__ names neither."""
    with pytest.raises(TypeError) as exc:
        _construct(_ScaleRequired, {"fixed_images": "F", "moving_images": "M"}, "moments")
    message = str(exc.value)
    assert "_ScaleRequired" in message
    assert "'moments' stage" in message
    assert "adapter omitted:  scale" in message
    assert "chunkreg/engines/fireants.py" in message


def test_an_argument_the_build_refuses_is_named_too():
    with pytest.raises(TypeError, match="build rejects:    smooth_grad_sigma"):
        _construct(
            _NoScale,
            {"fixed_images": "F", "moving_images": "M", "smooth_grad_sigma": 1.0},
            "greedy",
        )


# --------------------------------------------------------------------------- #
# Against the real signatures
# --------------------------------------------------------------------------- #
# Copied verbatim from the build that anatomix's install_fireants.sh pins, as
# reported by inspect.signature on the cluster. Only the names matter: a stage
# class ends in **kwargs, so a name the adapter gets wrong is accepted and
# ignored rather than refused, and the stage then runs as if it had never been
# given the argument. These stand-ins are how that is caught here instead of
# after a queue wait.
class _RealMoments:
    """Including the accessors the real class exposes for the next solver."""

    def get_rigid_init_dict(self):
        return {"init_moment": _Shaped((1, 3, 3)), "init_translation": _Shaped((1, 3))}

    def get_affine_init_dict(self):
        return {"init_rigid": _Shaped((1, 3, 4))}

    def get_affine_init(self):
        return _Shaped((3, 4))

    def __init__(self, scale, fixed_images, moving_images, blur=True, moments=1,
                 orientation="rot", transl_mode="com", loss_type="cc",
                 loss_params={}, mi_kernel_type="gaussian",
                 cc_kernel_type="rectangular", tolerance=1e-6,
                 max_tolerance_iters=10, cc_kernel_size=3, custom_loss=None,
                 perform_scaling=False, **kwargs): ...


class _RealRigid:
    def __init__(self, scales, iterations, fixed_images, moving_images,
                 loss_type="cc", optimizer="Adam", optimizer_params={},
                 optimizer_lr=0.03, loss_params={}, mi_kernel_type="gaussian",
                 cc_kernel_type="rectangular", tolerance=1e-6,
                 max_tolerance_iters=10, cc_kernel_size=3, init_translation=None,
                 init_moment=None, scaling=False, custom_loss=None,
                 around_center=True, blur=True, normalize_translation=False,
                 translation_lr=None, **kwargs): ...


class _RealAffine:
    def __init__(self, scales, iterations, fixed_images, moving_images,
                 loss_type="cc", optimizer="Adam", optimizer_params={},
                 loss_params={}, optimizer_lr=0.03, mi_kernel_type="gaussian",
                 cc_kernel_type="rectangular", cc_kernel_size=3, tolerance=1e-6,
                 max_tolerance_iters=10, around_center=True, init_rigid=None,
                 custom_loss=None, blur=True, normalize_translation=False,
                 translation_lr=None, scale_bounds=None, **kwargs): ...


class _RealGreedy:
    def __init__(self, scales, iterations, fixed_images, moving_images,
                 loss_type="cc", deformation_type="compositive", optimizer="Adam",
                 optimizer_params={}, optimizer_lr=0.5, integrator_n=7,
                 mi_kernel_type="gaussian", cc_kernel_type="rectangular",
                 cc_kernel_size=7, smooth_warp_sigma=0.5, smooth_grad_sigma=1.0,
                 loss_params={}, reduction="mean", tolerance=1e-6,
                 max_tolerance_iters=10, init_affine=None, warp_reg=None,
                 displacement_reg=None, blur=True, freeform=False,
                 custom_loss=None, **kwargs): ...


class _Shaped:
    """Stands in for a tensor where only the shape is under test."""

    def __init__(self, shape):
        self.shape = tuple(shape)


REAL_API = {
    "Moments": _RealMoments, "Rigid": _RealRigid,
    "Affine": _RealAffine, "Greedy": _RealGreedy, "SyN": None,
}


def _plan(kind, stage, carry=None):
    common = dict(
        fixed_images="F", moving_images="M",
        loss_type=getattr(stage, "loss", "cc"),
        cc_kernel_size=getattr(stage, "cc_kernel", 7),
        optimizer="Adam", optimizer_lr=getattr(stage, "lr", 0.5),
        tolerance=getattr(stage, "tolerance", 1e-6),
    )
    if kind != "moments":
        common["scales"] = list(stage.scales)
        common["iterations"] = list(stage.iterations)
    return FireAntsEngine._stage_plan(REAL_API, kind, stage, common, carry)


@pytest.mark.parametrize("kind", ["moments", "rigid", "affine", "greedy"])
def test_every_level_zero_stage_builds_against_the_pinned_build(kind):
    """moments -> rigid -> affine -> greedy is what level 0 runs."""
    stage = StageSpec(kind=kind)
    cls, kwargs = _plan(kind, stage, carry=_Shaped((1, 3, 4)))
    assert isinstance(_construct(cls, kwargs, kind), cls)


def test_moments_is_given_no_optimiser_schedule():
    """MomentsRegistration declares no optimizer, lr, scales or iterations."""
    _, kwargs = _plan("moments", StageSpec(kind="moments"))
    assert set(kwargs) == {"fixed_images", "moving_images", "scale"}


def test_progress_bar_is_not_passed_to_a_build_that_has_no_such_argument():
    for kind in ("rigid", "affine", "greedy"):
        _, kwargs = _plan(kind, StageSpec(kind=kind), carry=None)
        assert "progress_bar" not in kwargs


def test_a_linear_stage_whose_transform_cannot_be_found_is_not_silently_dropped():
    """Losing it costs the level its global alignment, unrecoverably."""
    from chunkreg.engines.fireants import _carry_of

    class _Opaque:
        def get_rigid_transform_matrix(self):  # a name the adapter lacks
            return "M"

    with pytest.raises(RuntimeError, match="has no get_rigid_matrix"):
        _carry_of(_Opaque(), "rigid")
    # The message names the candidate, or the next run guesses again.
    try:
        _carry_of(_Opaque(), "rigid")
    except RuntimeError as exc:
        assert "get_rigid_transform_matrix" in str(exc)


def test_a_linear_stage_hands_on_the_matrix_anatomix_reads():
    from chunkreg.engines.fireants import _carry_of

    class _Rigid:
        def get_rigid_matrix(self):
            return "R"

    class _Affine:
        def get_affine_matrix(self):
            return "A"

    assert _carry_of(_Rigid(), "rigid") == "R"
    assert _carry_of(_Affine(), "affine") == "A"


def test_moments_hands_the_next_solver_its_own_init_arguments():
    """Not its affine matrix: rigid wants the rotation and translation apart.

    Passing the [D, D+1] affine as init_moment is what failed on the cluster,
    inside get_rotation_matrix, which wants [N, D, D].
    """
    moments = _RealMoments(1, "F", "M")
    _, rigid = _plan("rigid", StageSpec(kind="rigid"), carry=moments)
    assert rigid["init_moment"].shape == (1, 3, 3)
    assert rigid["init_translation"].shape == (1, 3)
    assert "init_rigid" not in rigid

    _, aff = _plan("affine", StageSpec(kind="affine"), carry=moments)
    assert aff["init_rigid"].shape == (1, 3, 4)
    assert "init_moment" not in aff


def test_a_rigid_matrix_initialises_the_affine_that_follows_it():
    _, aff = _plan("affine", StageSpec(kind="affine"), carry=_Shaped((1, 3, 4)))
    assert aff["init_rigid"].shape == (1, 3, 4)


def test_the_moments_affine_is_squared_up_for_a_deformable_stage():
    """As anatomix does: matrix[:, :3] = moments.get_affine_init()."""
    torch = pytest.importorskip("torch")
    from chunkreg.engines.fireants import _affine_matrix

    class _M:
        def get_affine_init(self):
            return torch.arange(12, dtype=torch.float32).reshape(3, 4)

    out = _affine_matrix({"torch": torch}, _M())
    assert tuple(out.shape) == (1, 4, 4)
    assert torch.equal(out[0, :3], torch.arange(12, dtype=torch.float32).reshape(3, 4))
    assert torch.equal(out[0, 3], torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_a_stage_with_nothing_before_it_is_given_no_initialiser():
    for kind in ("rigid", "affine"):
        _, kwargs = _plan(kind, StageSpec(kind=kind), carry=None)
        assert not {"init_moment", "init_rigid", "init_translation"} & set(kwargs)
    _, greedy = _plan("greedy", StageSpec(kind="greedy"), carry=None)
    assert greedy["init_affine"] is None


# --------------------------------------------------------------------------- #
# Arguments a subclass forwards to its parent
# --------------------------------------------------------------------------- #
class _Abstract:
    """Stands in for AbstractRegistration, which is where FireANTs keeps the
    arguments every stage shares."""

    def __init__(self, fixed_images, moving_images, progress_bar=True,
                 loss_type="cc", tolerance=1e-6, **kwargs):
        self.progress_bar = progress_bar


class _Forwarding(_Abstract):
    def __init__(self, scales, iterations, fixed_images, moving_images,
                 optimizer="Adam", optimizer_lr=0.5, cc_kernel_size=7, **kwargs):
        super().__init__(fixed_images, moving_images, **kwargs)


def test_an_argument_the_parent_declares_is_not_treated_as_unknown():
    """Checking only the subclass dropped progress_bar, so every iteration of
    every chunk was narrated into the job log."""
    from chunkreg.engines.fireants import _declared

    assert "progress_bar" in _declared(_Forwarding)
    assert "scales" in _declared(_Forwarding)
    assert "kwargs" not in _declared(_Forwarding)

    built = _construct(
        _Forwarding,
        {"scales": [1], "iterations": [1], "fixed_images": "F",
         "moving_images": "M", "progress_bar": False},
        "greedy",
    )
    assert built.progress_bar is False


def test_a_name_no_class_in_the_chain_declares_is_still_caught():
    with pytest.raises(TypeError, match="build rejects"):
        _construct(
            _Forwarding,
            {"scales": [1], "iterations": [1], "fixed_images": "F",
             "moving_images": "M", "progres_bar": False},   # typo
            "greedy",
        )


def test_the_engine_defaults_to_quiet_and_the_environment_can_override(monkeypatch):
    from chunkreg.engines.fireants import FireAntsEngine

    monkeypatch.delenv("CHUNKREG_PROGRESS", raising=False)
    assert FireAntsEngine().progress is False
    monkeypatch.setenv("CHUNKREG_PROGRESS", "1")
    assert FireAntsEngine().progress is True
    assert FireAntsEngine(progress=False).progress is False, "an explicit setting wins"

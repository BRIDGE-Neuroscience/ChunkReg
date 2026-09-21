"""Pairwise registration: one moving volume onto one fixed volume.

Pairwise is the groupwise loop with the template held fixed, so what needs
testing is not the level loop again but the places where holding it fixed
changes the answer: the stopping rule (which can no longer read template
motion), the template at each level (which is the fixed subject's own data,
not an average), and the deliverable field (which maps fixed to moving).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chunkreg import fields
from chunkreg.config import LevelPolicy, RunConfig, StageSpec, SubjectSpec
from chunkreg.grid import GridSpec, Profile, pyramid
from chunkreg.pipelines import build_template, register_pair
from chunkreg.pipelines.groupwise import PassReport, _should_stop
from chunkreg.runners import LocalRunner
from chunkreg.store import Volume, ingest_array

from .conftest import blob

SPACING = 0.05
SHAPE = (64, 64, 64)
SHIFT = 3
"""Native voxels the moving volume is displaced by. An integer shift is the
deformation the conventions selftest uses, for the same reason it is used
here: the right answer is known exactly, on one axis, with one sign, so a
transposition or a sign flip anywhere in the pipeline shows up as a failure
rather than as a slightly worse number."""


@pytest.fixture
def pair_profile():
    """Like ``small_profile``, but wide enough in the halo to load.

    ``small_profile`` leaves a clamp of seven voxels, just under the eight
    ``RunConfig.validate`` insists on, which is fine for a config built in
    process and never checked. These tests write configs out and read them
    back, which does check, so this one has the two extra halo voxels that
    take the clamp to nine.
    """
    return Profile(
        core=32,
        halo=14,
        inner_chunk=8,
        lattice_factor=2,
        channels=1,
        features="intensity",
        r_f=0,
        k=3,
        sigma_g=0.5,
        sigma_w=0.5,
        scales=(2, 1),
    )


@pytest.fixture
def pair(pair_profile):
    """A fixed volume and a moving one shifted from it by a known amount."""
    fixed = blob(SHAPE, seed=42, smooth=2.0)
    moving = np.roll(fixed, SHIFT, axis=0)
    for sid, data in (("fix", fixed), ("mov", moving)):
        ingest_array(
            (data * 4000).astype(np.uint16),
            f"subjects/{sid}.zarr",
            SPACING,
            pair_profile,
        )
    return fixed, moving


@pytest.fixture
def identical(pair_profile):
    """Two stores holding the same volume. The right field is zero."""
    fixed = blob(SHAPE, seed=7, smooth=2.0)
    for sid in ("fix", "mov"):
        ingest_array(
            (fixed * 4000).astype(np.uint16),
            f"subjects/{sid}.zarr",
            SPACING,
            pair_profile,
        )
    return fixed


def make_cfg(profile, **kw) -> RunConfig:
    base = dict(
        root=".",
        subjects=(
            SubjectSpec("fix", "subjects/fix.zarr"),
            SubjectSpec("mov", "subjects/mov.zarr"),
        ),
        profile=profile,
        # A real preset name: a config is written back as a preset plus
        # the overrides that change it, so a made-up name cannot be.
        profile_name="i1",
        backend="memory",
        gpu_mem_gb=1000.0,
        levels=LevelPolicy(
            caps=(3, 2, 2),
            level0_stages=(
                StageSpec(kind="greedy", scales=(4, 2, 1), iterations=(40, 25, 15)),
            ),
            seeded_stages=(
                StageSpec(kind="greedy", scales=(2, 1), iterations=(25, 15)),
            ),
            sharpen_laplacian_levels=(),
        ),
    )
    base.update(kw)
    return RunConfig(**base)


# --------------------------------------------------------------------------- #
# The stopping rule
# --------------------------------------------------------------------------- #
def _report(d99: float, step: float = 0.0) -> PassReport:
    return PassReport(
        level=1, iteration=0, d99_mm=d99, ubar_mm=0.0, step_mm=step,
        fold_frac=0.0, retries=0, skipped=0, seconds=0.0,
    )


def test_a_fixed_template_level_does_not_stop_on_a_large_residual(pair_profile):
    """The regression: a held-fixed template reports no motion, ever.

    ``step_mm`` comes from the update pass, which a pairwise run never
    dispatches, so the merged histogram is empty and the percentile is 0.0.
    Reading template convergence off that made every level -- and in
    particular the finest, which has no next halo to test against -- exit
    after a single pass however far from converged it was.
    """
    cfg = make_cfg(pair_profile, template_subject="fix")
    grid = GridSpec(SHAPE, SPACING)
    big = 20 * SPACING  # forty times the sub-voxel threshold

    stopped, reason = _should_stop(cfg, _report(big), 0.0, grid, 0, cap=4)
    assert not stopped, f"stopped at the finest level on a {big} mm residual: {reason}"

    stopped, reason = _should_stop(cfg, _report(big), 0.5 * SPACING, grid, 0, cap=4)
    assert not stopped, f"stopped on a residual larger than the next halo: {reason}"


def test_a_fixed_template_level_stops_once_its_residual_is_sub_voxel(pair_profile):
    cfg = make_cfg(pair_profile, template_subject="fix")
    grid = GridSpec(SHAPE, SPACING)
    small = 0.4 * cfg.levels.residual_vox * SPACING

    stopped, reason = _should_stop(cfg, _report(small), 0.0, grid, 0, cap=4)
    assert stopped and reason == "residual is sub-voxel"


def test_a_fixed_template_level_stops_once_the_next_halo_can_absorb_it(pair_profile):
    """Above sub-voxel but inside the next level's clamp is still a stop."""
    cfg = make_cfg(pair_profile, template_subject="fix")
    grid = GridSpec(SHAPE, SPACING)
    d99 = 2 * SPACING

    assert not _should_stop(cfg, _report(d99), 0.0, grid, 0, cap=4)[0]
    stopped, reason = _should_stop(cfg, _report(d99), 3 * SPACING, grid, 0, cap=4)
    assert stopped and reason == "residual fits the next halo"


def test_a_fixed_template_level_still_honours_its_cap(pair_profile):
    cfg = make_cfg(pair_profile, template_subject="fix")
    grid = GridSpec(SHAPE, SPACING)
    stopped, reason = _should_stop(cfg, _report(20 * SPACING), 0.0, grid, 3, cap=4)
    assert stopped and reason == "cap reached"


def test_an_estimated_template_keeps_its_own_rule(pair_profile):
    """The groupwise rule is unchanged: the template motion still binds."""
    cfg = make_cfg(pair_profile)
    grid = GridSpec(SHAPE, SPACING)
    tiny = 0.01 * SPACING

    # Sub-voxel residual but the template is still moving: not converged.
    moving_template = _report(tiny, step=5 * SPACING)
    assert not _should_stop(cfg, moving_template, 0.0, grid, 0, cap=4)[0]

    settled = _report(tiny, step=0.1 * SPACING)
    stopped, reason = _should_stop(cfg, settled, 0.0, grid, 0, cap=4)
    assert stopped and reason == "template converged"


# --------------------------------------------------------------------------- #
# The whole loop
# --------------------------------------------------------------------------- #
def _inner_median(u: np.ndarray, margin: int = 16) -> np.ndarray:
    """Per-component median of a field, away from the boundary.

    The median, not the mean: an integer roll wraps one face, and the padded
    boxes at the volume edge are clipped, so a handful of voxels carry large
    and meaningless displacements. They move a mean and cannot move a median.
    """
    inner = (slice(None),) + tuple(slice(margin, -margin) for _ in range(3))
    return np.median(np.asarray(u[inner]).reshape(3, -1), axis=1)


def test_register_pair_recovers_a_known_shift(pair, pair_profile):
    """The field a pairwise run delivers maps fixed coordinates to moving ones.

    The moving volume is the fixed one rolled by ``SHIFT`` along axis 0, so the
    delivered field must be ``+SHIFT`` voxels on component 0 and zero on the
    other two -- the same convention, sign and axis order that
    :func:`chunkreg.selftest.run_selftest` pins for a single chunk, here
    carried through seeding, every level, the blend and the final settle.
    """
    cfg = make_cfg(pair_profile)
    result = register_pair(cfg, fixed="fix", moving="mov", runner=LocalRunner())

    assert set(result.fields) == {"mov"}, "the fixed subject gets no field of its own"
    assert result.fixed_template == "fix"

    grid = result.template.grid(0)
    u = result.fields["mov"].read_dense((0, 0, 0), grid.shape)
    got = _inner_median(u)
    want = (SHIFT * SPACING, 0.0, 0.0)
    tol = 1.0 * SPACING
    for d in range(3):
        assert abs(got[d] - want[d]) <= tol, (
            f"component {d}: recovered {got[d]:+.4f} mm, wanted {want[d]:+.4f} mm "
            f"(tolerance {tol:.4f}); full median {got}"
        )


def test_register_pair_leaves_identical_volumes_alone(identical, pair_profile):
    """Nothing to correct means a field of zero, not a field of noise.

    This is the sharpest end-to-end check available that does not depend on
    the engine's accuracy: the demons update vanishes identically when the two
    images agree, so anything non-zero here came from the pipeline around it --
    a seed composed the wrong way, a blend weighting that does not sum to one,
    a promote that resamples off by a voxel.
    """
    cfg = make_cfg(pair_profile)
    result = register_pair(cfg, fixed="fix", moving="mov", runner=LocalRunner())

    grid = result.template.grid(0)
    u = result.fields["mov"].read_dense((0, 0, 0), grid.shape)
    inner = (slice(None),) + tuple(slice(16, -16) for _ in range(3))
    worst = float(np.abs(u[inner]).max())
    assert worst <= 0.5 * SPACING, (
        f"registering a volume to itself moved it by {worst:.4f} mm "
        f"({worst / SPACING:.2f} voxels)"
    )


def test_the_template_of_a_pairwise_run_is_the_fixed_subject(pair, pair_profile):
    """Not an average of the two: the fixed anatomy, at every level."""
    cfg = make_cfg(pair_profile)
    result = register_pair(cfg, fixed="fix", moving="mov", runner=LocalRunner())

    grid = result.template.grid(0)
    got = result.template.read_padded(0, (0, 0, 0), grid.shape, normalise=True)
    src = Volume.open("subjects/fix.zarr", "memory")
    want = src.read_padded(src.n_levels - 1, (0, 0, 0), grid.shape, normalise=True)
    assert np.allclose(got, want, atol=1e-5)


def test_a_pairwise_level_runs_more_than_one_pass_when_it_needs_to(pair, pair_profile):
    """What the stopping-rule fix buys, end to end.

    The finest level has no successor, so before the fix it exited on pass 0
    unconditionally. Here it is given a cap above one and a residual target it
    cannot reach, so only the cap can stop it.
    """
    cfg = make_cfg(
        pair_profile,
        levels=LevelPolicy(
            caps=(1, 3),  # the pyramid is two levels deep here
            level0_stages=(
                StageSpec(kind="greedy", scales=(4, 2, 1), iterations=(40, 25, 15)),
            ),
            seeded_stages=(
                StageSpec(kind="greedy", scales=(2, 1), iterations=(25, 15)),
            ),
            sharpen_laplacian_levels=(),
            residual_vox=1e-9,
        ),
    )
    result = register_pair(cfg, fixed="fix", moving="mov", runner=LocalRunner())
    assert result.levels[-1].n_passes == 3, (
        f"the finest level ran {result.levels[-1].n_passes} pass(es); a fixed "
        f"template must not collapse the stopping rule to one"
    )


def test_a_pairwise_summary_leaves_out_the_template_motion_column(pair, pair_profile):
    cfg = make_cfg(pair_profile)
    result = register_pair(cfg, fixed="fix", moving="mov", runner=LocalRunner())
    assert "step mm" not in result.summary()
    assert "d99 mm" in result.summary()


def test_template_subject_alone_is_enough_for_a_pairwise_run(pair, pair_profile):
    """``chunkreg run`` on a two-subject config with a fixed template.

    This is the route a distributed pairwise run takes: no in-memory config
    surgery, so the workers reading the config file build the same pair.
    """
    cfg = make_cfg(pair_profile, template_subject="fix")
    result = build_template(cfg, runner=LocalRunner())
    assert set(result.fields) == {"mov"}
    assert result.fixed_template == "fix"


def test_a_pairwise_run_refuses_a_subject_it_does_not_have(pair, pair_profile):
    cfg = make_cfg(pair_profile)
    with pytest.raises(ValueError, match="not in this run's subject list"):
        register_pair(cfg, fixed="fix", moving="nobody", runner=LocalRunner())


def test_a_pairwise_run_refuses_to_register_a_volume_to_itself(pair, pair_profile):
    cfg = make_cfg(pair_profile)
    with pytest.raises(ValueError, match="registering a volume to itself"):
        register_pair(cfg, fixed="fix", moving="fix", runner=LocalRunner())


# --------------------------------------------------------------------------- #
# The derived configuration
# --------------------------------------------------------------------------- #
def test_a_pair_runs_under_a_root_of_its_own(pair, pair_profile):
    """Otherwise a pair resumes from the cohort run's finished passes.

    Every run writes ``levels/L*/pass_it*.json`` and seeds from
    ``levels/L*/seeded.json``, and every run resumes from whatever it finds.
    A pair sharing the cohort's root would replay the cohort's passes as its
    own and seed level 0 from the cohort's template -- silently, because
    resuming is exactly what those files are for.
    """
    from chunkreg.pipelines import pair_config

    cfg = make_cfg(pair_profile)
    paired = pair_config(cfg, "fix", "mov")
    assert Path(paired.root) != Path(cfg.root).resolve()
    assert Path(paired.root).name == "fix__mov"
    assert paired.template_subject == "fix"
    assert paired.subject_ids == ("fix", "mov")
    assert paired.registered_ids == ("mov",)


def test_a_pair_config_keeps_pointing_at_the_cohorts_stores(pair, pair_profile):
    """The root moves, so the subject paths have to stop being relative to it."""
    from chunkreg.pipelines import pair_config

    cfg = make_cfg(pair_profile)
    paired = pair_config(cfg, "fix", "mov")
    for sid in ("fix", "mov"):
        assert paired.subject_path(sid) == cfg.subject_path(sid).absolute()
        assert Volume.exists(paired.subject_path(sid), "memory")


def test_a_pair_config_is_absolute_so_another_process_can_load_it(pair, pair_profile):
    """A worker or batch node need not start where the driver did."""
    from chunkreg.pipelines import pair_config

    paired = pair_config(make_cfg(pair_profile), "fix", "mov")
    assert Path(paired.root).is_absolute()
    for sub in paired.subjects:
        assert Path(sub.path).is_absolute()
        if sub.source is not None:
            assert Path(sub.source).is_absolute()


def test_register_pair_writes_the_config_it_dispatches_by(pair, pair_profile):
    """What makes a pairwise run reachable by more than one process."""
    from chunkreg.config import load_config
    from chunkreg.pipelines import pair_config

    cfg = make_cfg(pair_profile)
    register_pair(cfg, fixed="fix", moving="mov", runner=LocalRunner())

    written = Path(pair_config(cfg, "fix", "mov").root) / "config.json"
    assert written.exists(), "nothing for a worker to load"
    assert load_config(written) == pair_config(cfg, "fix", "mov")


def test_two_pairs_of_one_cohort_do_not_share_a_root(pair, pair_profile):
    from chunkreg.pipelines import pair_config

    cfg = make_cfg(pair_profile)
    a = pair_config(cfg, "fix", "mov")
    b = pair_config(cfg, "mov", "fix")
    assert a.root != b.root


def test_a_pair_root_can_be_chosen(pair, pair_profile, tmp_path):
    from chunkreg.pipelines import pair_config

    where = tmp_path / "somewhere-else"
    paired = pair_config(make_cfg(pair_profile), "fix", "mov", root=where)
    assert Path(paired.root) == where

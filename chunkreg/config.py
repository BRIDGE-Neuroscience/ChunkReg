"""Run configuration: profiles, stages, level policy, validation.

A run is described by a chunk profile and a subject list. Almost everything
else is derived: the pyramid depth from the grid, the displacement clamp from
the halo, the number of template passes from a stopping rule. What remains
configurable is either a resource decision (which profile, which scheduler) or
a policy ceiling (caps, retry thresholds, retention).

Validation happens at load time and is deliberately strict. A profile whose
halo cannot accommodate its own similarity kernel and feature receptive field
produces a non-positive clamp, which would let chunks disagree without bound in
their overlaps; that is rejected here rather than discovered after a level has
run.
"""

from __future__ import annotations

import dataclasses
import math
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Sequence

from .grid import GridSpec, Profile

__all__ = [
    "StageSpec",
    "TUNABLE_STAGE_FIELDS",
    "LevelPolicy",
    "IngestSpec",
    "RetryPolicy",
    "Retention",
    "Resources",
    "SlurmConfig",
    "SubjectSpec",
    "GridRequest",
    "RunConfig",
    "PROFILES",
    "get_profile",
    "load_config",
    "save_config",
    "unused_level_overrides",
    "skipped_level_overrides",
    "ConfigError",
    "LEVEL_ORDER",
]


class ConfigError(ValueError):
    """A configuration that cannot be run as written."""


LEVEL_ORDER = "ome"
"""How chunkreg numbers pyramid levels: as OME-Zarr numbers its datasets, with
level 0 at full resolution. Recorded in every run root it writes."""

MIN_D_MAX_VOX = 8.0
"""Smallest clamp worth running. Below this a chunk can barely move a feature
past the interpolation footprint, and the level cannot make progress."""


# --------------------------------------------------------------------------- #
# Named profiles
# --------------------------------------------------------------------------- #
PROFILES: dict[str, dict[str, Any]] = {
    # 16 anatomix channels. The reference profile; needs an 80 GB device.
    "a16": dict(channels=16, features="anatomix", r_f=24),
    # anatomix projected to 8 channels by a PCA fitted once at the coarsest level.
    "a8": dict(channels=8, features="pca:8", r_f=24),
    "a4": dict(channels=4, features="pca:4", r_f=24),
    # Raw intensity. No receptive field, so the whole halo past the kernel
    # support becomes clamp: three times the reach of the feature profiles.
    "i1": dict(channels=1, features="intensity", r_f=0),
}


def get_profile(name: str, **overrides: Any) -> Profile:
    """Build a :class:`Profile` from a named preset."""
    if name not in PROFILES:
        raise ConfigError(
            f"unknown profile {name!r}; known profiles: {', '.join(sorted(PROFILES))}"
        )
    kwargs = dict(PROFILES[name])
    kwargs.update(overrides)
    return Profile(**kwargs)


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StageSpec:
    """One registration stage handed to an engine."""

    kind: Literal["moments", "rigid", "affine", "greedy", "syn"] = "greedy"
    scales: tuple[int, ...] = (2, 1)
    iterations: tuple[int, ...] = (50, 30)
    loss: str = "cc"
    cc_kernel: int = 7
    lr: float = 0.5
    translation_lr: float | None = None
    smooth_grad_sigma: float = 1.0
    smooth_warp_sigma: float = 0.5
    tolerance: float = 1e-6
    gradient_checkpointing: bool = True

    KINDS = ("moments", "rigid", "affine", "greedy", "syn")

    def __post_init__(self) -> None:
        if self.kind not in self.KINDS:
            raise ConfigError(
                f"unknown stage kind {self.kind!r}; known: {list(self.KINDS)}"
            )
        object.__setattr__(self, "scales", tuple(int(s) for s in self.scales))
        object.__setattr__(self, "iterations", tuple(int(i) for i in self.iterations))
        if self.kind == "moments":
            return
        if not self.scales:
            raise ConfigError(f"stage {self.kind!r} needs at least one scale")
        if len(self.scales) != len(self.iterations):
            raise ConfigError(
                f"stage {self.kind!r}: scales {self.scales} and iterations "
                f"{self.iterations} must have the same length"
            )
        if list(self.scales) != sorted(self.scales, reverse=True):
            raise ConfigError(
                f"stage {self.kind!r}: scales must run coarse to fine, "
                f"got {self.scales}"
            )
        if any(i <= 0 for i in self.iterations):
            raise ConfigError(f"stage {self.kind!r}: iterations must be positive")

    @property
    def s_max(self) -> int:
        return int(max(self.scales)) if self.scales else 1

    @property
    def is_deformable(self) -> bool:
        return self.kind in ("greedy", "syn")

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_obj(cls, obj: Any) -> "StageSpec":
        """Accept ``"rigid"``, ``{"greedy": {...}}`` or ``{"kind": ..., ...}``."""
        if isinstance(obj, str):
            return cls(kind=obj)  # type: ignore[arg-type]
        if not isinstance(obj, dict):
            raise ConfigError(f"cannot read a stage from {obj!r}")
        if "kind" in obj:
            return cls(**obj)
        if len(obj) != 1:
            raise ConfigError(
                f"a stage mapping must be {{kind: params}}, got keys {list(obj)}"
            )
        (kind, params), = obj.items()
        return cls(kind=kind, **(params or {}))  # type: ignore[arg-type]


TUNABLE_STAGE_FIELDS: frozenset[str] = frozenset({
    "scales",
    "iterations",
    "lr",
    "translation_lr",
    "smooth_grad_sigma",
    "smooth_warp_sigma",
    "cc_kernel",
    "loss",
    "tolerance",
})
"""Stage fields a per-level override may set.

Deliberately not the whole of :class:`StageSpec`. ``kind`` is excluded because
changing the kind of a stage is a different pipeline, not a retune, and the
halo budget is validated against the stage list a level resolves to; letting a
patch swap ``greedy`` for ``syn`` would move that check out from under itself.
Use ``level_stages`` to replace a level's stage list outright.
"""


def _patch_stage(stage: StageSpec, patch: dict[str, Any]) -> StageSpec:
    """Apply a per-level override to one stage.

    ``moments`` carries no schedule, so it is returned untouched rather than
    being handed iterations it would silently ignore.
    """
    if stage.kind == "moments":
        return stage
    unknown = set(patch) - TUNABLE_STAGE_FIELDS
    if unknown:
        raise ConfigError(
            f"unknown per-level stage parameter(s) {sorted(unknown)}; "
            f"settable: {sorted(TUNABLE_STAGE_FIELDS)}"
        )
    kw: dict[str, Any] = {}
    for key, value in patch.items():
        if key in ("scales", "iterations"):
            kw[key] = tuple(int(v) for v in value)
        elif key in ("cc_kernel",):
            kw[key] = int(value)
        elif key == "loss":
            kw[key] = str(value)
        elif key == "translation_lr":
            kw[key] = None if value is None else float(value)
        else:
            kw[key] = float(value)
    return replace(stage, **kw)


DEFAULT_COARSEST_STAGES: tuple[StageSpec, ...] = (
    StageSpec(kind="moments", scales=(), iterations=()),
    StageSpec(kind="rigid", scales=(4, 2, 1), iterations=(200, 100, 50), lr=0.01),
    StageSpec(kind="affine", scales=(4, 2, 1), iterations=(200, 100, 50), lr=0.01),
    StageSpec(kind="greedy", scales=(8, 4, 2, 1), iterations=(120, 80, 50, 30)),
)
"""The coarsest level is one chunk, so its halo is irrelevant and it can afford
the deep pyramid that estimates the global component nothing else can see."""

DEFAULT_SEEDED_STAGES: tuple[StageSpec, ...] = (
    StageSpec(kind="greedy", scales=(2, 1), iterations=(50, 30)),
)
"""Seeded levels solve a residual the seed already placed within the clamp, so
two scales suffice; a deeper in-chunk pyramid would spend the whole halo on
kernel support (see :meth:`Profile.d_max_vox`)."""


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #
_RENAMED_LEVEL_KEYS = {
    "level0_stages": "coarsest_stages",
    "level0_max_disp_frac": "coarsest_max_disp_frac",
}
"""``levels`` keys from the coarse-first numbering, refused rather than aliased."""


@dataclass(frozen=True)
class LevelPolicy:
    """Ceilings and thresholds for the level loop.

    Levels are numbered as OME-Zarr numbers its datasets, and as a subject
    store holds them: level ``0`` is full resolution (the run grid), level
    ``l`` has ``2**l`` times its spacing, and the highest level, ``K``, is the
    coarsest, the first at which the whole volume fits one chunk. The loop
    runs ``K, K-1, ..., 0``. Every level-keyed or level-indexed setting below
    uses that number.

    ``K`` follows from the grid and the chunk core, not from the
    configuration, so a level number names the coarsest level only once the
    pyramid is known. Settings for the coarsest level whatever the depth are
    the ``coarsest_*`` ones.
    """

    caps: tuple[int, ...] = (2, 3, 4, 6, 8)
    """Maximum template passes per level, indexed by level: ``caps[0]`` is full
    resolution, ``caps[1]`` the level above it, and so on towards the
    coarsest. A pyramid deeper than the list repeats its last value for the
    extra coarse levels, so the last entry is the budget of the coarsest
    level and of anything coarser."""
    coarsest_stages: tuple[StageSpec, ...] = DEFAULT_COARSEST_STAGES
    """Stages of the coarsest level, the single chunk where the whole-volume
    alignment is estimated. Unseeded and unclamped by any halo, so it may use
    a deep in-chunk pyramid."""
    seeded_stages: tuple[StageSpec, ...] = DEFAULT_SEEDED_STAGES
    """Stages of every other level, each seeded from the level above it."""
    residual_frac: float = 0.5
    """Stop a level once its residual falls below this fraction of the next
    level's clamp, so the finer level's halo can absorb what is left."""
    ubar_vox: float = 1.0
    """Stop once the template recentring step is below this many voxels."""
    residual_vox: float = 0.5
    """Stop a held-fixed-template level once its residual is below this many
    voxels.

    Only pairwise runs read this. An estimated template converges when it stops
    moving, which is what ``ubar_vox`` measures; a template held fixed never
    moves, so that test is vacuous and the residual is the only thing left that
    converges. Below half a voxel at the 99th percentile the pass is correcting
    less than the level can represent, and the next level will resolve it
    better than another pass here."""
    stop_percentile: float = 99.0
    """Which percentile of displacement magnitude the stopping rule reads.

    The design note specified 99.9. In practice a small number of voxels sit
    pinned at the clamp for as long as the clamp is active, so a 99.9th
    percentile is bounded below by the clamp itself and the rule can never
    clear. The 99th percentile still ignores nothing that matters and does
    converge."""
    shape_update_step: float = 0.25
    sharpen_laplacian_levels: tuple[int, ...] | None = None
    """Levels whose updated template is unsharp-masked against the blur of
    averaging. ``None`` is the two coarsest levels, ``K`` and ``K - 1``,
    whatever the depth; a list names levels by number (``0`` = full
    resolution)."""
    min_tissue_fraction: float = 0.02
    """Chunks with less foreground than this emit their seed unchanged."""
    coarsest_max_disp_frac: float = 0.25
    """The coarsest level is a single chunk, so no halo bounds it. This bounds
    it by the aperture instead: a displacement approaching the size of the
    volume cannot be a real correspondence, whatever the similarity says.
    Applies to any level that fits one chunk, which only the coarsest does."""
    require_subvoxel: bool = False
    """With a template held fixed, stop a level only when its residual is
    sub-voxel (or at the cap), not as soon as it would fit the next level's
    halo. For a weak similarity signal at the coarse levels, where the point is
    to converge there before handing a stable field down."""
    plateau_passes: int | None = None
    """With a template held fixed, stop a level once its residual percentile
    has fallen by less than ``plateau_rel`` (relative) over this many passes.
    For a cross-modal pair whose residual never goes sub-voxel: Dice stops
    improving long before the stopping rule would otherwise fire, and the
    level would run to its cap. ``None`` disables it."""
    plateau_rel: float = 0.05
    plateau_after: int | None = None
    """The pass from which the plateau test may fire. ``None`` means after the
    level's ``anneal`` schedule has run its course (or from the first pass
    without one): while the smoothing is still changing, the residual moves
    for reasons that have nothing to do with convergence."""
    anneal: dict[int, dict[str, Any]] = field(default_factory=dict)
    """Per-level smoothing schedule over the passes of that level, keyed by
    level number (``0`` = full resolution): ``{"smooth_grad_sigma": [start, end], "smooth_warp_sigma":
    [start, end], "passes": N}``. Pass ``k`` uses ``start * (end / start) **
    min(k / (N - 1), 1)`` -- geometric from heavy to light over the first N
    passes, then held -- on every deformable stage of the level, after
    ``level_params``. Heavy smoothing first keeps the early passes, where a weak
    similarity is least reliable, from warping to extremes; the later passes
    refine with less."""

    level_stages: dict[int, tuple[StageSpec, ...]] = field(default_factory=dict)
    """Per-level replacement of the whole stage list, keyed by level number
    (``0`` = full resolution).

    The escape hatch for a level that needs a different pipeline rather than a
    different tuning: an extra affine stage at the level below the coarsest,
    say. Overrides ``coarsest_stages`` and ``seeded_stages`` for the levels
    it names."""

    level_params: dict[int, dict[str, Any]] = field(default_factory=dict)
    """Per-level tuning applied on top of whichever stage list a level
    resolves to, keyed by level number (``0`` = full resolution).

    This is the knob for structures that need different regularisation at
    different resolutions: weaker smoothing and more iterations at the fine
    levels (the low numbers) where cerebellar foliation lives, stiffer at the
    coarse levels that carry the global shape. Only
    :data:`TUNABLE_STAGE_FIELDS` may be set, and the patch reaches every
    non-moments stage of that level."""

    run: tuple[int | float, ...] | str | None = None
    """Which levels to run. ``None`` or ``"all"`` runs every level.

    A tuple names levels: an ``int`` is a level number (``0`` = full
    resolution) and a ``float`` is a spacing in millimetres, matched against
    the pyramid when it is known. ``"listed"`` runs the coarsest level plus
    every level named in ``level_params`` or ``level_stages``. Leaving out the
    finest levels (the lowest numbers) stops the run early; leaving out a
    middle level hands the solution straight to the next finer level that
    does run. The coarsest level always has to run, because the global
    alignment is estimated there and nothing finer can recover it."""

    stop_at: int | float | None = None
    """The finest level to run, as a level number (``0`` = full resolution)
    or a spacing in millimetres.

    Early stopping without listing every level: ``"200um"`` runs every level
    ``run`` selects down to 200 um and stops there, writing the final fields
    at that resolution; ``2`` stops at level 2, four times the native
    spacing. A later run with a finer ``stop_at`` (a smaller number or
    spacing) carries on from where this one stopped."""

    def run_levels(self, grids: Sequence[GridSpec]) -> tuple[int, ...]:
        """Resolve ``run`` and ``stop_at`` against a pyramid into levels.

        ``grids`` is indexed by level, as :func:`chunkreg.grid.pyramid`
        returns it. The levels come back in the order they run: coarsest
        first, finest last.
        """
        coarsest = len(grids) - 1
        source = "levels.run"
        if self.run is None or self.run == "all":
            chosen = set(range(len(grids)))
        else:
            entries: tuple = tuple(self.run)
            if self.run == "listed":
                entries = (coarsest, *self.tuned_levels())
                source = "levels.run 'listed' (the coarsest level plus the tuned levels)"
            chosen = {_find_level(grids, e, source) for e in entries}
        if self.stop_at is not None:
            last = _find_level(grids, self.stop_at, "levels.stop_at")
            chosen = {k for k in chosen if k >= last}
        if coarsest not in chosen:
            raise ConfigError(
                f"{source} leaves out level {coarsest} "
                f"({grids[coarsest].spacing_mm * 1000:g} um), the coarsest. The "
                f"coarsest level is where the whole-volume alignment is "
                f"estimated, so every run starts there. A run that stopped part "
                f"way resumes on its own when started again."
            )
        return tuple(sorted(chosen, reverse=True))

    def cap(self, level: int) -> int:
        """The pass budget of a level: ``caps[level]``, or the last cap past the end."""
        if not self.caps:
            return 1
        return int(self.caps[min(int(level), len(self.caps) - 1)])

    def base_stages(self, level: int, *, coarsest: int) -> tuple[StageSpec, ...]:
        """The stage list for a level, before per-level tuning is applied.

        ``coarsest`` is the coarsest level of the run's pyramid, ``K``.
        """
        return self._base(level, int(level) == int(coarsest))

    def _base(self, level: int, is_coarsest: bool) -> tuple[StageSpec, ...]:
        override = self.level_stages.get(int(level))
        if override:
            return tuple(override)
        return self.coarsest_stages if is_coarsest else self.seeded_stages

    def _tuned(self, level: int, is_coarsest: bool) -> tuple[StageSpec, ...]:
        k = int(level)
        base = self._base(k, is_coarsest)
        patch = self.level_params.get(k)
        if not patch:
            return base
        try:
            return tuple(_patch_stage(s, patch) for s in base)
        except ConfigError as exc:
            raise ConfigError(f"level {k}: {exc}") from None

    def stages(self, level: int, *, coarsest: int) -> tuple[StageSpec, ...]:
        """The stage list a level actually runs.

        Resolution order is ``level_stages[level]``, else ``coarsest_stages``
        at the coarsest level (``level == coarsest``) or ``seeded_stages``
        below it, with ``level_params[level]`` patched over the result.
        ``coarsest`` is ``K``, the coarsest level of the run's pyramid: it is
        what the level number is compared against, since it depends on the
        grid rather than on the configuration.
        """
        return self._tuned(level, int(level) == int(coarsest))

    def stages_for_pass(
        self, level: int, iteration: int, *, coarsest: int
    ) -> tuple[StageSpec, ...]:
        """:meth:`stages` with this level's ``anneal`` schedule applied for pass ``iteration``."""
        stages = self.stages(level, coarsest=coarsest)
        sched = self.anneal.get(int(level))
        if not sched:
            return stages
        n = int(sched.get("passes", 1))
        t = 1.0 if n <= 1 else min(max(int(iteration), 0) / (n - 1), 1.0)
        patch = {}
        for key in ("smooth_grad_sigma", "smooth_warp_sigma"):
            if key in sched:
                start, end = (float(x) for x in sched[key])
                patch[key] = start * (end / start) ** t
        return tuple(replace(s, **patch) if s.is_deformable else s for s in stages)

    def tuned_levels(self) -> tuple[int, ...]:
        """Levels named by a per-level override, in ascending order."""
        return tuple(sorted(set(self.level_stages) | set(self.level_params)))

    def sharpens(self, level: int, *, coarsest: int) -> bool:
        """Is this level's updated template unsharp-masked?

        ``sharpen_laplacian_levels``, or by default the two coarsest levels.
        """
        if self.sharpen_laplacian_levels is None:
            return int(level) >= int(coarsest) - 1
        return int(level) in self.sharpen_laplacian_levels


@dataclass(frozen=True)
class RetryPolicy:
    """What a register task does when a chunk solution folds or diverges."""

    fold_frac: float = 0.001
    ladder: tuple[str, ...] = ("sigma_w_x2", "clamp_x0.5", "emit_seed")

    def __post_init__(self) -> None:
        known = {"sigma_w_x2", "clamp_x0.5", "emit_seed"}
        bad = set(self.ladder) - known
        if bad:
            raise ConfigError(
                f"unknown retry steps {sorted(bad)}; known: {sorted(known)}"
            )


@dataclass(frozen=True)
class Retention:
    keep_history_levels: tuple[int, ...] | None = None
    """Levels whose history to keep, by number (``0`` = full resolution).
    ``None`` is the two coarsest levels."""
    delete_task_arrays: bool = True
    delete_accumulators: bool = True
    residual_map_max_voxels: int | None = None
    """Keep a level's residual QC map only while its lattice has at most this
    many voxels. ``None`` keeps every level's. At 20 um the map is a
    3700-cubed float32 volume; ``2**30`` keeps the maps down to 80 um."""


@dataclass(frozen=True)
class Resources:
    gpus: int = 0
    cpus: int = 8
    mem_gb: int = 32
    time: str = "02:00:00"
    chunks_per_task: int = 16


@dataclass(frozen=True)
class SlurmConfig:
    partition: str = "gpu"
    account: str | None = None
    poll_seconds: float = 30.0
    """How often the driver asks sacct for array state."""
    max_wait_s: float | None = None
    """Give up waiting for a pass after this many seconds. ``None`` waits
    indefinitely, which is right for a queue that can hold a job for days but
    wrong for an array that will never reach a terminal state; set it when a
    run should fail rather than hang."""
    register: Resources = Resources(gpus=1, cpus=8, mem_gb=64, time="04:00:00")
    blend: Resources = Resources(gpus=0, cpus=8, mem_gb=32, time="02:00:00")
    update: Resources = Resources(gpus=0, cpus=8, mem_gb=32, time="01:00:00")


@dataclass(frozen=True)
class SubjectSpec:
    id: str
    path: str
    source: str | None = None
    """Foreign-format file or slice directory to ingest from.

    Present so that ingest is part of the run configuration rather than a
    hand-typed command per subject: ``chunkreg setup`` ingests every subject
    that names a source and has no store yet."""
    spacing_mm: tuple[float, float, float] | None = None
    """This scan's voxel size as ``(z, y, x)`` millimetres, for a source that
    does not state one or states it wrongly. Overrides the top-level
    ``spacing_mm`` for this subject."""
    warm_start: dict | None = None
    """Start this subject from a known alignment instead of from scratch.

    ``affine_um``: a 4x4 (or 3x4) matrix, or the path of a JSON file holding
    one, mapping this scan's physical micrometres to the fixed scan's physical
    micrometres (the pairwise affine as it is usually estimated, moving to
    fixed). The scan is then *not* resampled onto the run grid: its store reads
    the scan in place through the affine at every level
    (:mod:`chunkreg.affine_view`), which is what makes a 20 um run of a
    7000-cubed scan start in seconds. Needs ``grid.reference`` (or
    ``template_subject``) to name the fixed subject, whose own sampling is the
    run grid. ``background``: ``[threshold, value]``, source voxels at or above
    the threshold become ``value`` before interpolation (BigBrain's white
    background: ``[250, 0]``). ``field``: a chunkreg field store to seed the
    coarsest level from, in the run grid's world frame; its vectors are
    millimetres, so only its lattice is resampled."""


@dataclass(frozen=True)
class GridRequest:
    """How the run grid is chosen. See :mod:`chunkreg.cohort`."""

    spacing_mm: float | str = "finest"
    """The finest level's spacing: a number, ``finest`` or ``coarsest``."""
    shape: tuple[int, int, int] | None = None
    """The run grid's shape. ``None`` fits every scan."""
    align: str = "centre"
    """``centre`` or ``corner``: how each scan is placed on the grid."""
    reference: str | None = None
    """A subject whose own sampling is the run grid.

    For registering onto one particular volume rather than building a template
    from a group: the named subject's shape and voxel size become the grid, it
    is copied onto it unchanged, and every other scan is resampled onto it and
    cropped to its field of view. The output then lands on the lattice the
    answer is expected on.

    Without it the grid is the smallest box holding every scan, at the finest
    voxel in the cohort, which is right for a cohort and wrong for a target: a
    moving scan sampled more finely than the fixed one would otherwise drag
    the whole run to its resolution."""


@dataclass(frozen=True)
class IngestSpec:
    """How ``chunkreg setup`` turns a source file into a store."""

    dtype: str = "auto"
    """Storage dtype. ``auto`` keeps an integer source's type and stores a
    floating source as float32."""
    median_radius: float | None = None
    """Radius of an optional median denoise, in voxels. ``None`` disables it."""
    percentiles: tuple[float, float] = (0.5, 99.5)
    """Intensity window, measured once at the coarsest level of each subject."""
    link: bool = True
    """Read a zarr scan that is already exactly on the run grid in place,
    writing only the coarser levels, instead of copying it. ``false`` always
    copies, for scans on storage too slow or too far away to read from."""


# --------------------------------------------------------------------------- #
# Run configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RunConfig:
    root: str
    subjects: tuple[SubjectSpec, ...]
    profile: Profile
    profile_name: str = "a16"
    spacing_mm: float | None = None
    """Voxel size of any source that does not state its own. Individual
    subjects can override it; the run grid itself is set by ``grid``."""
    grid: GridRequest = GridRequest()
    runner: Literal["local", "slurm"] = "local"
    backend: Literal["zarr", "memory"] = "zarr"
    levels: LevelPolicy = LevelPolicy()
    ingest: IngestSpec = IngestSpec()
    retry: RetryPolicy = RetryPolicy()
    retention: Retention = Retention()
    slurm: SlurmConfig = SlurmConfig()
    template_subject: str | None = None
    """Hold this subject fixed as the template instead of estimating one.

    Pairwise registration is the groupwise loop with the template held fixed,
    so it is a setting rather than a separate pipeline. The named subject gets
    no field of its own, its volume is the template at every level, and the
    unbiasing update pass is skipped: there is nothing to unbias when the
    target is one particular anatomy rather than a population mean."""

    scratch: str | None = None
    """Where the transient objects of a pass (task arrays, accumulators) are
    written: a directory, absolute or relative to ``root``. Default
    ``<root>/scratch``. A pass at 20 um writes a terabyte of task arrays and
    deletes them again once blended, so on a node that has local scratch this
    points there and the shared filesystem never sees them. Every worker of a
    local run is on the one node, which is what makes that safe; a resumed
    run on another node simply redoes the unfinished pass."""
    gpu_mem_gb: float = 80.0
    bytes_per_voxel_channel: float = 90.0
    """Measured during ``chunkreg setup``; 90 is anatomix-on-FireANTs with
    LNCC, 75 with gradient checkpointing."""
    throughput_ch_vox_per_s: float = 1.5e6
    """Also measured during ``setup``. The default is a placeholder equal to a
    576-cubed single-channel chunk in 60 seconds."""
    seed: int = 0
    device: str = "auto"
    """Where the arithmetic runs. ``cuda`` (or ``cuda:N``) keeps every pass on
    the GPU and refuses to start without one; ``cpu`` runs the NumPy reference
    path; ``auto`` picks ``cuda`` when a GPU is visible. ``torch-cpu`` runs the
    GPU code on the CPU and exists for testing."""
    engine: str = "auto"
    """Registration engine. ``auto`` is FireANTs on a GPU device and the demons
    reference on the CPU."""
    gpus: int | str = "all"
    """How many GPUs a local run uses at once. ``all`` uses every GPU the job
    can see."""
    workers_per_gpu: int | str = 1
    """Worker processes sharing one GPU.

    One worker per GPU leaves the card idle whenever its worker is reading a
    chunk, writing a task array or decompressing a shard, which at a low
    channel count is a large fraction of a task. A second worker on the same
    card fills those gaps with the other's compute, at the cost of holding two
    registrations in memory at once.

    That trade only exists when two tasks fit. ``auto`` takes the profile's
    per-task estimate and allows a second worker when both fit inside 80% of
    ``gpu_mem_gb``, which at sixteen channels they do not. The default is 1:
    the memory estimate is a model until ``chunkreg setup`` measures
    ``bytes_per_voxel_channel`` on the actual backbone, and an out-of-memory
    failure mid-level is worse than an idle gap."""

    def workers_on_each_gpu(self) -> tuple[int, str]:
        """Workers to run per GPU, and the arithmetic behind the number."""
        per_task = self.profile.mem_gb(self.bytes_per_voxel_channel)
        budget = 0.8 * self.gpu_mem_gb
        if self.workers_per_gpu == "auto":
            n = 2 if 2 * per_task <= budget else 1
            why = (
                f"auto: {n} x {per_task:.0f} GB fits the {budget:.0f} GB "
                f"usable of a {self.gpu_mem_gb:.0f} GB device"
            )
            return n, why
        n = int(self.workers_per_gpu)
        why = f"{n} x {per_task:.0f} GB against {budget:.0f} GB usable"
        if n * per_task > budget:
            why += " -- over budget, expect out-of-memory failures"
        return n, why

    def engine_for(self, device: str) -> str:
        """The engine to use once the device is known."""
        if self.engine != "auto":
            return self.engine
        return "fireants" if str(device).startswith("cuda") else "demons"

    # -- paths -------------------------------------------------------------- #
    @property
    def root_path(self) -> Path:
        return Path(self.root)

    def level_dir(self, level: int) -> Path:
        """``levels/L<l>``: ``L0`` is full resolution, the highest the coarsest."""
        return self.root_path / "levels" / f"L{level}"

    def template_path(self, level: int) -> Path:
        return self.level_dir(level) / "template.zarr"

    def field_path(self, level: int, subject: str) -> Path:
        return self.level_dir(level) / "fields" / f"{subject}.zarr"

    def final_field_path(self, subject: str) -> Path:
        """Where a finished run leaves a subject's field.

        Distinct from ``field_path`` at the last level because the deliverable
        has the final template motion folded in and the level store does not.
        """
        return self.root_path / "fields" / f"{subject}.zarr"

    def recentre_path(self, level: int) -> Path:
        """Where a pass records the template motion the next pass must undo."""
        return self.level_dir(level) / "recentre.zarr"

    def pass_record_path(self, level: int, iteration: int) -> Path:
        """Where a finished pass records that it finished, and how it went.

        Task arrays are transient and retention deletes them as soon as the
        blend that consumed them has verified, so they cannot be the evidence
        that a pass completed. This record is durable and small.
        """
        return self.level_dir(level) / f"pass_it{iteration}.json"

    def seed_marker_path(self, level: int) -> Path:
        """Written once a level has its starting template and fields.

        Seeding the coarsest level and promoting into a finer level both
        overwrite the level's fields, so a resumed run must never do either
        twice. This is how it knows.
        """
        return self.level_dir(level) / "seeded.json"

    @property
    def level_order_path(self) -> Path:
        """The record of how this run root numbers its levels.

        Written when a run first creates ``levels/``. A root whose ``levels/``
        has progress but no such record was made under the earlier
        coarse-first numbering, where ``L0`` was the coarsest level.
        """
        return self.root_path / "levels" / "level_order.json"

    def check_level_order(self) -> None:
        """Refuse a run root made under the earlier coarse-first numbering.

        There, ``levels/L0`` was the coarsest level and ``L<K>`` full
        resolution, the reverse of now, and nothing else on disk says which
        way round a root is. Resuming one would replay its records against
        the wrong resolutions. A root with no progress, or one this version
        made, passes.
        """
        if self.level_order_path.exists():
            try:
                order = json.loads(
                    self.level_order_path.read_text(encoding="utf-8")
                ).get("level_order")
            except (OSError, ValueError, AttributeError):
                order = None
            if order != LEVEL_ORDER:
                raise ConfigError(
                    f"{self.level_order_path} records level_order {order!r}; "
                    f"this version of chunkreg only reads {LEVEL_ORDER!r} "
                    f"(level 0 is full resolution)"
                )
            return
        levels = self.root_path / "levels"
        if not levels.is_dir() or not any(levels.glob("L*")):
            return  # nothing has run here yet
        raise ConfigError(
            f"{self.root_path} holds a run made with the earlier coarse-first "
            f"level numbering: there levels/L0 is the coarsest level and the "
            f"highest L is full resolution, the reverse of this version, where "
            f"level 0 is full resolution as in OME-Zarr. Its progress cannot "
            f"be resumed here. Finish it with the previous chunkreg version, "
            f"or start a new run root (a different 'root' in the config)."
        )

    def record_level_order(self) -> None:
        """Mark this run root as numbering its levels the OME way."""
        if self.level_order_path.exists():
            return
        self.level_order_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.level_order_path.with_name(self.level_order_path.name + ".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "level_order": LEVEL_ORDER,
                    "meaning": "levels/L<l> is pyramid level l, which is OME-Zarr "
                    "dataset l of a subject store: L0 is full resolution, the "
                    "highest l the coarsest",
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        tmp.replace(self.level_order_path)

    @property
    def grid_record_path(self) -> Path:
        """The run grid and each scan's placement on it, written at ingest."""
        return self.root_path / "grid.json"

    @property
    def scratch_root(self) -> Path:
        """Everything a finished pass may delete lives under here."""
        if self.scratch:
            p = Path(self.scratch)
            return p if p.is_absolute() else self.root_path / p
        return self.root_path / "scratch"

    def scratch_dir(self, level: int, iteration: int) -> Path:
        return self.scratch_root / f"L{level}_it{iteration}"

    def task_path(self, level: int, iteration: int, task_id: int) -> Path:
        return self.scratch_dir(level, iteration) / f"task_{task_id:06d}.zarr"

    def accumulator_path(self, level: int, iteration: int, which: str) -> Path:
        if which not in ("isum", "wsum"):
            raise ValueError(f"unknown accumulator {which!r}")
        return self.scratch_dir(level, iteration) / f"{which}.zarr"

    @property
    def subject_ids(self) -> tuple[str, ...]:
        return tuple(s.id for s in self.subjects)

    @property
    def n_subjects(self) -> int:
        return len(self.subjects)

    @property
    def registered_ids(self) -> tuple[str, ...]:
        """Subjects that get a field of their own.

        Everything in the cohort except a subject being held fixed as the
        template, which is a target rather than a member of the group.
        """
        return tuple(s for s in self.subject_ids if s != self.template_subject)

    def pair_root(self, fixed: str, moving: str) -> Path:
        """Where a pairwise run of these two subjects keeps its own output.

        Below the cohort's root but separate from it, because a pairwise run
        writes the same file names a groupwise run does and every run resumes
        from what it finds.
        """
        return self.root_path / "pairs" / f"{fixed}__{moving}"

    def subject_path(self, subject: str) -> Path:
        for s in self.subjects:
            if s.id == subject:
                p = Path(s.path)
                return p if p.is_absolute() else self.root_path / p
        raise KeyError(f"no subject {subject!r} in this run")

    # -- validation --------------------------------------------------------- #
    def validate(self, n_levels: int | None = None) -> list[str]:
        """Raise on anything unrunnable; return non-fatal warnings.

        ``n_levels`` is the depth of the pyramid the run will use, when it is
        known. A level number only names the coarsest level, whose stages the
        halo does not bound, once the depth is known; without it a per-level
        override that would only be valid at the coarsest level is reported
        as a warning here and checked exactly once the run has its pyramid.
        """
        warnings: list[str] = []
        p = self.profile

        d_max = p.d_max_vox()
        if d_max < MIN_D_MAX_VOX:
            raise ConfigError(
                f"profile leaves a clamp of {d_max:.1f} voxels, below the "
                f"minimum of {MIN_D_MAX_VOX}. Halo {p.halo} must cover the "
                f"kernel and smoothing support ({p.support_vox():.1f} at "
                f"s_max={p.s_max}) plus the feature receptive field ({p.r_f}). "
                f"Raise halo to at least "
                f"{int(p.support_vox() + p.r_f + MIN_D_MAX_VOX)}, lower the "
                f"in-chunk scales, or use a profile with a smaller receptive "
                f"field."
            )

        # The coarsest level's stages and the generic seeded stages always
        # exist; every level named by a per-level override is resolved here
        # too, so a bad patch fails at load time rather than part way through
        # a run.
        self._check_stages("coarsest_stages", self.levels.coarsest_stages, True, warnings)
        self._check_stages("seeded_stages", self.levels.seeded_stages, False, warnings)
        coarsest = None if n_levels is None else int(n_levels) - 1
        self._check_tuned_levels(coarsest, warnings)

        run = self.levels.run
        if isinstance(run, str) and run not in ("all", "listed"):
            raise ConfigError(
                f"levels.run must be 'all', 'listed' or a list of levels, got {run!r}"
            )
        if isinstance(run, tuple):
            if not run:
                raise ConfigError("levels.run is empty; leave it out to run every level")
            if any(isinstance(e, int) and e < 0 for e in run):
                raise ConfigError(f"levels.run has a negative level: {list(run)}")
            if (
                coarsest is not None
                and all(isinstance(e, int) for e in run)
                and coarsest not in run
            ):
                raise ConfigError(
                    f"levels.run {list(run)} leaves out level {coarsest}, the "
                    f"coarsest. The coarsest level is where the whole-volume "
                    f"alignment is estimated, so every run starts there. A run "
                    f"that stopped part way resumes on its own when started again."
                )

        for level, spec in sorted(self.levels.level_stages.items()):
            if level < 0:
                raise ConfigError(f"level_stages has a negative level {level}")
            if not spec:
                raise ConfigError(f"level_stages[{level}] is empty")
        for level in sorted(self.levels.level_params):
            if level < 0:
                raise ConfigError(f"level_params has a negative level {level}")

        if not self.subjects:
            raise ConfigError("a run needs at least one subject")
        if self.template_subject is not None:
            if self.template_subject not in self.subject_ids:
                raise ConfigError(
                    f"template_subject {self.template_subject!r} is not in the "
                    f"subject list {list(self.subject_ids)}"
                )
            if not self.registered_ids:
                raise ConfigError(
                    f"template_subject {self.template_subject!r} is the only "
                    f"subject, so there is nothing left to register against it"
                )
        ids = [s.id for s in self.subjects]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ConfigError(f"duplicate subject ids: {dupes}")
        fixed_id = self.grid.reference or self.template_subject
        for s in self.subjects:
            ws = s.warm_start or {}
            if "affine_um" not in ws:
                continue
            if s.source is None:
                raise ConfigError(
                    f"subject {s.id!r} has a warm_start affine but no 'source'; the "
                    f"affine says where to read the scan, so the scan has to be named"
                )
            if fixed_id is None or fixed_id == s.id:
                raise ConfigError(
                    f"subject {s.id!r} has a warm_start affine, which maps it onto the "
                    f"fixed subject's frame, but no other subject is named by "
                    f"grid.reference or template_subject"
                )
            if (self.subjects[[x.id for x in self.subjects].index(fixed_id)].warm_start or {}).get("affine_um"):
                raise ConfigError(
                    f"the fixed subject {fixed_id!r} cannot itself have a warm_start affine"
                )

        mem = p.mem_gb(self.bytes_per_voxel_channel)
        if mem > 0.8 * self.gpu_mem_gb:
            warnings.append(
                f"profile {self.profile_name!r} needs {mem:.0f} GB per padded "
                f"chunk, above 80% of the declared {self.gpu_mem_gb:.0f} GB "
                f"device. Use gradient checkpointing, a lower channel count, "
                f"or a smaller core."
            )

        if self.runner == "slurm" and self.slurm.register.gpus < 1:
            raise ConfigError("the register pass needs at least one GPU")

        if not 0.0 < self.levels.shape_update_step <= 1.0:
            raise ConfigError(
                f"shape_update_step must be in (0, 1], got "
                f"{self.levels.shape_update_step}"
            )
        if self.levels.residual_frac <= 0:
            raise ConfigError("residual_frac must be positive")
        if not 0.0 < self.levels.stop_percentile <= 100.0:
            raise ConfigError(
                f"levels.stop.percentile must be in (0, 100], got "
                f"{self.levels.stop_percentile}"
            )
        if any(c < 1 for c in self.levels.caps):
            raise ConfigError(
                f"every cap must allow at least one pass, got {list(self.levels.caps)}"
            )
        if not 0.0 <= self.levels.min_tissue_fraction <= 1.0:
            raise ConfigError(
                f"min_tissue_fraction must be a fraction, got "
                f"{self.levels.min_tissue_fraction}"
            )
        if not 0.0 <= self.retry.fold_frac <= 1.0:
            raise ConfigError(
                f"retry.fold_frac must be a fraction, got {self.retry.fold_frac}"
            )
        if self.spacing_mm is not None and self.spacing_mm <= 0:
            raise ConfigError(f"spacing_mm must be positive, got {self.spacing_mm}")
        g = self.grid
        if isinstance(g.spacing_mm, str):
            if g.spacing_mm not in ("finest", "coarsest"):
                raise ConfigError(
                    f"grid.spacing_mm must be a number, 'finest' or 'coarsest', "
                    f"got {g.spacing_mm!r}"
                )
        elif not g.spacing_mm > 0:
            raise ConfigError(f"grid.spacing_mm must be positive, got {g.spacing_mm}")
        if g.shape is not None and (len(g.shape) != 3 or min(g.shape) < 1):
            raise ConfigError(f"grid.shape must be three positive integers, got {g.shape}")
        if g.align not in ("centre", "corner"):
            raise ConfigError(f"grid.align must be 'centre' or 'corner', got {g.align!r}")
        if g.reference is not None:
            if g.reference not in self.subject_ids:
                raise ConfigError(
                    f"grid.reference is {g.reference!r}, which is not one of "
                    f"the subjects {list(self.subject_ids)}"
                )
            if not (isinstance(g.spacing_mm, str) and g.spacing_mm == "finest"):
                raise ConfigError(
                    f"grid.reference is {g.reference!r} and grid.spacing_mm is "
                    f"{g.spacing_mm!r}. The reference subject's own voxel size "
                    f"is the run spacing, so setting both asks for two "
                    f"different grids; drop one."
                )
            if g.shape is not None:
                raise ConfigError(
                    f"grid.reference is {g.reference!r} and grid.shape is "
                    f"{list(g.shape)}. The reference subject's own shape is the "
                    f"run grid's, so setting both asks for two different grids; "
                    f"drop one."
                )
        if isinstance(self.levels.stop_at, int) and self.levels.stop_at < 0:
            raise ConfigError(f"levels.stop_at is negative: {self.levels.stop_at}")

        dev = str(self.device).lower()
        if dev not in ("auto", "cpu", "cuda", "torch-cpu") and not (
            dev.startswith("cuda:") and dev[5:].isdigit()
        ):
            raise ConfigError(
                f"device must be 'auto', 'cuda', 'cuda:N', 'cpu' or 'torch-cpu', "
                f"got {self.device!r}"
            )
        from .engines import _REGISTRY as _ENGINES

        if self.engine != "auto" and self.engine not in _ENGINES:
            raise ConfigError(
                f"unknown engine {self.engine!r}; use 'auto' or one of "
                f"{sorted(_ENGINES)}"
            )
        if dev.startswith("cuda") and self.engine == "demons":
            raise ConfigError(
                "engine 'demons' is the CPU reference and cannot run on a GPU "
                "device; use 'fireants' or leave engine as 'auto'"
            )
        if not (self.gpus == "all" or (isinstance(self.gpus, int) and self.gpus >= 1)):
            raise ConfigError(f"gpus must be 'all' or a positive integer, got {self.gpus!r}")

        return warnings

    def check_level_overrides(self, n_levels: int) -> list[str]:
        """Check every per-level override against a pyramid of known depth.

        The part of :meth:`validate` that needs the depth: with it, a level
        number is the coarsest level or it is not, and each override is held
        to exactly the rules of the level it names. The run calls this once
        it has its pyramid, before any work. Returns the warnings.
        """
        warnings: list[str] = []
        self._check_tuned_levels(int(n_levels) - 1, warnings)
        return warnings

    def _check_tuned_levels(self, coarsest: int | None, warnings: list[str]) -> None:
        for level in self.levels.tuned_levels():
            where = f"level {level}"
            if coarsest is not None:
                is_top = level == coarsest
                self._check_stages(
                    where, self.levels._tuned(level, is_top), is_top, warnings
                )
                continue
            # The depth is not known yet, so the level is taken as a seeded
            # one, which it is for every pyramid deeper than it. Only if it is
            # unrunnable there but would be fine as the coarsest level is the
            # verdict deferred to the run, which knows the depth.
            try:
                self._check_stages(
                    where, self.levels._tuned(level, False), False, warnings
                )
            except ConfigError as as_seeded:
                as_top: list[str] = []
                try:
                    self._check_stages(
                        where, self.levels._tuned(level, True), True, as_top
                    )
                except ConfigError:
                    raise as_seeded from None
                warnings.append(
                    f"{where}'s override is only valid if level {level} is the "
                    f"coarsest level of the pyramid ({as_seeded}); it is checked "
                    f"again once the run knows the pyramid depth"
                )

    def _check_stages(
        self,
        where: str,
        stages: Sequence[StageSpec],
        coarsest: bool,
        warnings: list[str],
    ) -> None:
        """Check one level's stage list against the profile's halo budget.

        The coarsest level is a single chunk with no halo, so only seeded
        levels are held to the budget.
        """
        p = self.profile
        if not stages:
            raise ConfigError(f"{where} has no stages")
        if not any(s.is_deformable for s in stages):
            warnings.append(
                f"{where} has no deformable stage; it can only produce a "
                f"linear alignment"
            )
        if coarsest:
            return
        for s in stages:
            if s.is_deformable and s.s_max > p.s_max:
                raise ConfigError(
                    f"{where}: stage {s.kind!r} uses scales up to "
                    f"{s.s_max} but the profile budgets its halo for "
                    f"s_max={p.s_max}. Either set profile scales to "
                    f"{s.scales} (which would leave a clamp of "
                    f"{p.halo - p.support_vox(s.s_max) - p.r_f:.1f} "
                    f"voxels) or lower the stage scales."
                )
            if not s.is_deformable:
                continue
            # The halo is budgeted from the profile's kernel and sigmas. A
            # stage that asks for a wider kernel or more smoothing reaches
            # further than that budget, and the clamp, which is computed
            # from the profile, then promises more displacement than the
            # halo can really support.
            need = s.s_max * (
                (s.cc_kernel - 1) / 2.0
                + 3.0 * max(s.smooth_grad_sigma, s.smooth_warp_sigma)
            )
            budget = p.support_vox()
            if need > budget + 1e-9:
                left = p.halo - need - p.r_f
                msg = (
                    f"{where}: stage {s.kind!r} (cc_kernel "
                    f"{s.cc_kernel}, sigmas {s.smooth_grad_sigma:g}/"
                    f"{s.smooth_warp_sigma:g}, scales up to {s.s_max}) needs "
                    f"{need:.1f} voxels of halo for its kernel and smoothing, "
                    f"but the profile budgets {budget:.1f}. The clamp of "
                    f"{p.d_max_vox():.1f} voxels is overstated; only about "
                    f"{left:.1f} are really covered. Lower the kernel or "
                    f"sigmas, or raise the profile's k, sigma_g, sigma_w and "
                    f"halo to match."
                )
                # Refuse only when the halo covers no displacement at all;
                # a smaller real clamp than advertised is a warning, as
                # small test profiles run with default stages hit it.
                if left <= 0:
                    raise ConfigError(msg)
                warnings.append(msg)

    # -- serialisation ------------------------------------------------------ #
    def to_json(self) -> dict:
        def enc(v):
            if dataclasses.is_dataclass(v) and not isinstance(v, type):
                return {k: enc(x) for k, x in dataclasses.asdict(v).items()}
            if isinstance(v, (list, tuple)):
                return [enc(x) for x in v]
            if isinstance(v, dict):
                return {k: enc(x) for k, x in v.items()}
            if isinstance(v, Path):
                return str(v)
            return v

        return {k: enc(v) for k, v in dataclasses.asdict(self).items()}

    def fingerprint(self) -> str:
        """Stable hash of the run configuration, recorded in store provenance."""
        import hashlib

        blob = json.dumps(self.to_json(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def to_config_dict(self) -> dict:
        """This run as a mapping :func:`load_config` reads back unchanged.

        :meth:`to_json` is a dataclass dump for hashing and provenance; it is
        not the file schema and does not load. This is the file schema, and it
        exists because a run has to be able to write a configuration as well as
        read one: a derived run (a pair drawn out of a cohort, a registration
        set up from two paths) is dispatched to worker processes and batch
        nodes by config file, so a config that lives only in memory cannot
        leave the driver. See :func:`chunkreg.pipelines.register_pair`.

        Only what differs from the defaults is written, so the result reads
        like a configuration someone wrote rather than a dump of every knob.
        """
        out: dict[str, Any] = {"root": self.root}

        out["subjects"] = [
            {
                k: v
                for k, v in (
                    ("id", sub.id),
                    ("path", sub.path),
                    ("source", sub.source),
                    (
                        "spacing_mm",
                        None if sub.spacing_mm is None else list(sub.spacing_mm),
                    ),
                    ("warm_start", None if sub.warm_start is None else dict(sub.warm_start)),
                )
                if v is not None
            }
            for sub in self.subjects
        ]

        out["profile"] = self.profile_name
        # The profile carried here can differ from the named preset two ways,
        # and they have to be written back separately. profile_overrides reach
        # any field directly. The features block instead *derives* the
        # extractor spec and, from it, the channel count -- so those two are
        # left out of the overrides and reconstructed as a features block
        # below. Writing them as overrides would not survive the round trip:
        # load_config applies the features block after the overrides, and a
        # missing block does not mean "leave it alone", it means "take the
        # default", which for an anatomix profile is MIND switched on.
        try:
            base = get_profile(self.profile_name)
        except ConfigError:
            raise ConfigError(
                f"this run's profile_name is {self.profile_name!r}, which is "
                f"not one of the presets {sorted(PROFILES)}. A configuration "
                f"is written as a preset plus the overrides that change it, so "
                f"a profile with no preset behind it cannot be written in a "
                f"form load_config would read back."
            ) from None
        overrides = {
            f.name: getattr(self.profile, f.name)
            for f in dataclasses.fields(self.profile)
            if f.name not in ("features", "channels")
            and getattr(self.profile, f.name) != getattr(base, f.name)
        }
        if overrides:
            out["profile_overrides"] = {
                k: list(v) if isinstance(v, tuple) else v
                for k, v in overrides.items()
            }
        features = _features_config_block(self.profile)
        if _with_features(base, None) != _with_features(base, features):
            out["features"] = features

        for key, value, default in (
            ("spacing_mm", self.spacing_mm, None),
            ("template_subject", self.template_subject, None),
            ("runner", self.runner, "local"),
            ("backend", self.backend, "zarr"),
            ("device", self.device, "auto"),
            ("engine", self.engine, "auto"),
            ("gpus", self.gpus, "all"),
            ("workers_per_gpu", self.workers_per_gpu, 1),
            ("seed", self.seed, 0),
            ("gpu_mem_gb", self.gpu_mem_gb, 80.0),
            ("bytes_per_voxel_channel", self.bytes_per_voxel_channel, 90.0),
            ("throughput_ch_vox_per_s", self.throughput_ch_vox_per_s, 1.5e6),
            ("scratch", self.scratch, None),
        ):
            if value != default:
                out[key] = value

        grid = _diff_block(self.grid, GridRequest())
        if grid:
            if "shape" in grid:
                grid["shape"] = list(grid["shape"])
            out["grid"] = grid

        levels = self._levels_config_dict()
        if levels:
            out["levels"] = levels

        for key, obj, default in (
            ("ingest", self.ingest, IngestSpec()),
            ("retry", self.retry, RetryPolicy()),
            ("retention", self.retention, Retention()),
        ):
            block = _diff_block(obj, default)
            if block:
                out[key] = {
                    k: list(v) if isinstance(v, tuple) else v for k, v in block.items()
                }

        slurm = _diff_block(self.slurm, SlurmConfig(), skip=("register", "blend", "update"))
        for name in ("register", "blend", "update"):
            res = _diff_block(
                getattr(self.slurm, name), getattr(SlurmConfig, name)
            )
            if res:
                slurm[name] = res
        if slurm:
            out["slurm"] = slurm

        return out

    def _levels_config_dict(self) -> dict:
        """The ``levels`` block, including the nested ``stop`` sub-block."""
        default = LevelPolicy()
        out: dict[str, Any] = {}

        stop = {
            key: getattr(self.levels, attr)
            for key, attr in (
                ("residual_frac", "residual_frac"),
                ("ubar_vox", "ubar_vox"),
                ("residual_vox", "residual_vox"),
                ("percentile", "stop_percentile"),
            )
            if getattr(self.levels, attr) != getattr(default, attr)
        }
        if self.levels.require_subvoxel:
            stop["require_subvoxel"] = True
        if self.levels.plateau_passes is not None:
            plateau = {"passes": int(self.levels.plateau_passes), "rel": self.levels.plateau_rel}
            if self.levels.plateau_after is not None:
                plateau["after"] = int(self.levels.plateau_after)
            stop["plateau"] = plateau
        if stop:
            out["stop"] = stop

        for key in (
            "caps", "shape_update_step", "sharpen_laplacian_levels",
            "min_tissue_fraction", "coarsest_max_disp_frac",
        ):
            value = getattr(self.levels, key)
            if value != getattr(default, key):
                out[key] = list(value) if isinstance(value, tuple) else value

        for key in ("coarsest_stages", "seeded_stages"):
            value = getattr(self.levels, key)
            if value != getattr(default, key):
                out[key] = [st.to_json() for st in value]

        if self.levels.level_stages:
            out["level_stages"] = {
                str(k): [st.to_json() for st in v]
                for k, v in sorted(self.levels.level_stages.items())
            }
        if self.levels.level_params:
            out["level_params"] = {
                str(k): dict(v) for k, v in sorted(self.levels.level_params.items())
            }
        if self.levels.anneal:
            out["anneal"] = {str(k): dict(v) for k, v in sorted(self.levels.anneal.items())}

        if self.levels.run is not None:
            out["run"] = (
                self.levels.run
                if isinstance(self.levels.run, str)
                else [_level_ref_json(e) for e in self.levels.run]
            )
        if self.levels.stop_at is not None:
            out["stop_at"] = _level_ref_json(self.levels.stop_at)
        return out


def _features_config_block(profile: Profile) -> dict:
    """The ``features`` block that reproduces a resolved profile's extractor.

    The inverse of :func:`_with_features`, read off the spec string it writes:
    ``anatomix+mindssc@16`` is MIND on, sixteen channels drawn per pass.
    ``sample_channels`` is stated even when it is null, because leaving the key
    out asks for the default rather than for "no sampling".
    """
    from .features import anatomix_window

    spec, _, sample = str(profile.features).partition("@")
    out = {
        "mind": "mindssc" in spec.split("+"),
        "sample_channels": int(sample) if sample else None,
    }
    windows = [w for w in (anatomix_window(p) for p in spec.split("+")) if w is not None]
    if windows:
        out["anatomix_window"] = windows[0]
    return out


def _level_ref_json(ref: int | float):
    """A level reference as the file schema writes it.

    An index stays a number. A spacing must carry its unit: a bare decimal is
    refused on the way back in, deliberately, because ``0.2`` could as easily
    be a typo for level 2 as it could be 200 um.
    """
    return ref if isinstance(ref, int) else f"{ref:g}mm"


def _diff_block(obj, default, skip: Sequence[str] = ()) -> dict:
    """The fields of a dataclass that differ from a reference instance."""
    return {
        f.name: getattr(obj, f.name)
        for f in dataclasses.fields(obj)
        if f.name not in skip and getattr(obj, f.name) != getattr(default, f.name)
    }


def skipped_level_overrides(cfg: "RunConfig", run_levels: Sequence[int]) -> list[int]:
    """Per-level keys for levels that exist but ``levels.run`` leaves out."""
    return [k for k in cfg.levels.tuned_levels() if k not in set(run_levels)]


def unused_level_overrides(cfg: "RunConfig", n_levels: int) -> list[int]:
    """Per-level keys naming a level this run does not have.

    The pyramid depth follows from the grid and the chunk core, so it is not
    known until a subject has been ingested. A key for level 7 of a five-level
    run (levels 0 to 4, 0 being full resolution) is silently inert, which is
    exactly the kind of tuning mistake that looks like the tuning simply not
    working, so the CLI reports it.
    """
    return [k for k in cfg.levels.tuned_levels() if k >= int(n_levels)]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _stages(obj, default: tuple[StageSpec, ...]) -> tuple[StageSpec, ...]:
    if obj is None:
        return default
    return tuple(StageSpec.from_obj(o) for o in obj)


def _level_key(key, what: str) -> int:
    """Read a level number from a mapping key (``0`` = full resolution).

    JSON has no integer keys, so ``{"3": {...}}`` is the only spelling
    available there and has to mean the same as YAML's ``{3: {...}}``.
    """
    try:
        level = int(key)
    except (TypeError, ValueError):
        raise ConfigError(
            f"{what} is keyed by level number (0 = full resolution); "
            f"{key!r} is not one"
        ) from None
    if level < 0:
        raise ConfigError(f"{what} has a negative level {level}")
    return level


_SPACING_UNITS = {"um": 1e-3, "µm": 1e-3, "micron": 1e-3, "microns": 1e-3, "mm": 1.0}
_SPACING_RE = re.compile(r"^([0-9]*\.?[0-9]+(?:e-?[0-9]+)?)\s*(um|µm|microns?|mm)$")


def parse_level_ref(e, where: str = "levels.run") -> int | float:
    """Read one level reference: a number (``2``) or a spacing (``"200um"``).

    A number is a level as OME-Zarr numbers it: ``0`` is full resolution and
    level ``l`` has ``2**l`` times its spacing.

    A spacing comes back as millimetres in a ``float``, an index as an
    ``int``. A bare decimal is refused: ``0.2`` could be a spacing or a typo for
    a level, and guessing wrong silently runs the wrong resolution.
    """
    if isinstance(e, bool):
        raise ConfigError(f"{where} entry {e!r} is not a level")
    if isinstance(e, int):
        return int(e)
    if isinstance(e, float):
        if e.is_integer():
            return int(e)
        raise ConfigError(
            f"{where} entry {e!r} is ambiguous; write a level number such "
            f"as 2 (0 is full resolution), or a spacing with its unit such as "
            f"\"200um\""
        )
    if isinstance(e, str):
        text = e.strip().lower()
        if text.isdigit():
            return int(text)
        m = _SPACING_RE.match(text)
        if m and float(m.group(1)) > 0:
            return float(m.group(1)) * _SPACING_UNITS[m.group(2)]
        raise ConfigError(
            f"{where} entry {e!r} is not a level; use a level number such "
            f"as 2 (0 is full resolution) or a spacing such as \"200um\" or "
            f"\"0.2mm\""
        )
    raise ConfigError(f"{where} entry {e!r} is not a level")


def _find_level(grids: Sequence[GridSpec], ref: int | float, where: str) -> int:
    """The level a level reference names. ``grids`` is indexed by level."""
    ladder = ", ".join(f"{k} = {g.spacing_mm * 1000:g} um" for k, g in enumerate(grids))
    if isinstance(ref, float):
        for k, g in enumerate(grids):
            if math.isclose(g.spacing_mm, ref, rel_tol=1e-3):
                return k
        raise ConfigError(
            f"{where} asks for {ref * 1000:g} um, but this pyramid has levels {ladder}"
        )
    k = int(ref)
    if not 0 <= k < len(grids):
        raise ConfigError(
            f"{where} names level {k}, but this pyramid only has levels {ladder}"
        )
    return k


def _run_spec(obj) -> tuple[int | float, ...] | str | None:
    """Read ``levels.run``: ``"all"``, ``"listed"``, or a list of levels."""
    if obj is None:
        return None
    if isinstance(obj, str):
        if obj not in ("all", "listed"):
            raise ConfigError(
                f"levels.run must be 'all', 'listed' or a list of levels, got {obj!r}"
            )
        return obj
    if not isinstance(obj, (list, tuple)):
        raise ConfigError(f"levels.run must be a list of levels, got {obj!r}")
    return tuple(parse_level_ref(e) for e in obj)


def _voxel(value, where: str) -> tuple[float, float, float] | None:
    """A voxel size from a number or a ``[z, y, x]`` list."""
    from .cohort import as_voxel_mm

    try:
        return as_voxel_mm(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where}: {exc}") from None


def _check_keys(obj: dict, allowed: set[str], where: str) -> None:
    """Reject unknown keys in a nested block.

    A mistyped tuning key that is silently ignored is indistinguishable from
    tuning that had no effect, which is the most expensive kind of mistake to
    diagnose in a run that takes GPU-days.
    """
    unknown = set(obj) - allowed
    if unknown:
        raise ConfigError(
            f"unknown key(s) {sorted(unknown)} in {where}; "
            f"known: {sorted(allowed)}"
        )


def _read_raw(source: str | Path) -> dict:
    """Read a config file. JSON by extension, YAML otherwise."""
    path = Path(source)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from None
    if path.suffix.lower() == ".json":
        try:
            return json.loads(text) or {}
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path} is not valid JSON: {exc}") from None
    import yaml

    try:
        return yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from None


def _with_features(profile: Profile, block) -> Profile:
    """Apply the ``features`` block: MIND-SSC alongside, and channel sampling.

    ``mind`` adds the twelve MIND-SSC channels to the profile's extractor. It
    defaults to on for anatomix, as anatomix's own registration uses them.
    ``sample_channels`` registers only that many channels per template pass,
    a new stratified draw each pass. It defaults to the profile's channel
    count for anatomix profiles whenever MIND makes the total larger, so
    adding MIND leaves device memory and per-pass cost where the profile put
    them.

    ``anatomix_window`` runs anatomix in sliding windows of that many voxels
    (a multiple of 32, at least 64) instead of over a whole chunk at once, so
    each channel sees a few centimetres of context rather than the whole brain.
    Between different subjects at coarse levels that is what makes the channels
    agree locally (BigBrain vs HiP-CT at 640 um: local agreement 0.30 with
    64-voxel windows vs 0.12 whole-volume). ``null`` or absent keeps one pass.
    """
    from .features import anatomix_window, describe

    feat = dict(block or {})
    _check_keys(feat, {"mind", "sample_channels", "anatomix_window"}, "the 'features' block")
    spec, _, _ = str(profile.features).partition("@")
    parts = [p for p in spec.split("+") if p]
    is_anatomix = lambda p: p == "anatomix" or anatomix_window(p) is not None
    has_anatomix = any(is_anatomix(p) for p in parts)
    if "anatomix_window" in feat:
        win = feat["anatomix_window"]
        if win is not None:
            win = int(win)
            if win < 64 or win % 32:
                raise ConfigError(
                    f"features.anatomix_window must be a multiple of 32 and at least 64, got {win}"
                )
        if not has_anatomix and win is not None:
            raise ConfigError("features.anatomix_window is set, but this profile has no anatomix channels")
        parts = [("anatomix" if win is None else f"anatomix:w{win}") if is_anatomix(p) else p for p in parts]
    mind = bool(feat.get("mind", has_anatomix))
    if mind and "mindssc" not in parts:
        parts.append("mindssc")
    if not mind and "mindssc" in parts and len(parts) > 1:
        parts.remove("mindssc")
    spec = "+".join(parts)
    total, r_f = describe(spec)
    sample = feat.get("sample_channels", "default")
    if sample == "default":
        # Only the heavy learned profiles are held to their channel budget;
        # thirteen intensity-plus-MIND channels cost next to nothing.
        sample = profile.channels if has_anatomix and total > profile.channels else None
    if sample is not None:
        sample = int(sample)
        if not 1 <= sample <= total:
            raise ConfigError(
                f"features.sample_channels is {sample}, but {spec} has {total} channels"
            )
        if sample < total:
            spec = f"{spec}@{sample}"
        total = sample
    if spec == profile.features and total == profile.channels:
        return profile
    try:
        return replace(profile, features=spec, channels=total, r_f=max(profile.r_f, r_f))
    except ValueError as exc:
        raise ConfigError(f"features block: {exc}") from None


def save_config(cfg: "RunConfig", path: str | Path) -> Path:
    """Write a run configuration as JSON that :func:`load_config` reads back.

    Used wherever a config derived in memory has to be handed to another
    process: a multi-GPU worker and a batch node each address their task by
    config file, so they rebuild the configuration by loading it rather than
    receiving it.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg.to_config_dict(), indent=2), encoding="utf-8")
    return out


def _warm_start(obj, sid: str) -> dict | None:
    """Normalise a subject's ``warm_start`` block; the matrix stays a path or a list."""
    if obj is None:
        return None
    if not isinstance(obj, dict):
        raise ConfigError(f"subject {sid!r}: warm_start must be a mapping")
    _check_keys(obj, {"affine_um", "background", "field"}, f"subject {sid!r} warm_start")
    out: dict[str, Any] = {}
    aff = obj.get("affine_um")
    if aff is not None:
        if isinstance(aff, str):
            out["affine_um"] = aff
        else:
            rows = [[float(v) for v in row] for row in aff]
            if len(rows) not in (3, 4) or any(len(r) != 4 for r in rows):
                raise ConfigError(
                    f"subject {sid!r}: warm_start.affine_um must be a 4x4 or 3x4 matrix "
                    f"or the path of a JSON file holding one"
                )
            if len(rows) == 3:
                rows.append([0.0, 0.0, 0.0, 1.0])
            out["affine_um"] = rows
    bg = obj.get("background")
    if bg is not None:
        if not (isinstance(bg, (list, tuple)) and len(bg) == 2):
            raise ConfigError(f"subject {sid!r}: warm_start.background must be [threshold, value]")
        out["background"] = [float(bg[0]), float(bg[1])]
    if obj.get("field") is not None:
        out["field"] = str(obj["field"])
    if not out:
        raise ConfigError(f"subject {sid!r}: warm_start names neither affine_um nor field")
    return out


def load_config(source: str | Path | dict) -> RunConfig:
    """Read a run configuration from a JSON or YAML file, or a mapping."""
    raw = _read_raw(source) if isinstance(source, (str, Path)) else dict(source)
    if not isinstance(raw, dict):
        raise ConfigError("a configuration must be a mapping at the top level")

    unknown = set(raw) - {
        "root", "subjects", "profile", "profile_overrides", "spacing_mm", "grid",
        "runner", "backend", "levels", "retry", "retention", "slurm", "gpu_mem_gb",
        "bytes_per_voxel_channel", "throughput_ch_vox_per_s", "seed", "channels",
        "backbone", "ingest", "template_subject", "features", "device", "engine", "gpus",
        "workers_per_gpu", "scratch",
    }
    if unknown:
        raise ConfigError(f"unknown configuration keys: {sorted(unknown)}")

    profile_name = raw.get("profile", "a16")
    overrides = dict(raw.get("profile_overrides") or {})
    if "backbone" in raw:
        overrides["backbone"] = raw["backbone"]
    if "channels" in raw:
        ch = raw["channels"]
        if isinstance(ch, str):
            name = {"intensity": "i1", "16": "a16"}.get(ch)
            if name is None and ch.startswith("pca:"):
                name = f"a{ch.split(':', 1)[1]}"
            if name is None:
                raise ConfigError(f"cannot read channels {ch!r}")
            profile_name = name
        else:
            overrides["channels"] = int(ch)
    try:
        profile = get_profile(profile_name, **overrides)
    except TypeError as exc:
        raise ConfigError(
            f"bad profile_overrides for profile {profile_name!r}: {exc}"
        ) from None

    profile = _with_features(profile, raw.get("features"))

    runner_name = raw.get("runner", "local")
    if runner_name not in ("local", "slurm"):
        raise ConfigError(
            f"unknown runner {runner_name!r}; use 'local' or 'slurm'"
        )
    backend_name = raw.get("backend", "zarr")
    if backend_name not in ("zarr", "memory"):
        raise ConfigError(
            f"unknown backend {backend_name!r}; use 'zarr' or 'memory'"
        )

    subjects = []
    for entry in raw.get("subjects") or []:
        if not isinstance(entry, dict) or "id" not in entry:
            raise ConfigError(
                f"each subject must be a mapping with an 'id', got {entry!r}"
            )
        _check_keys(
            entry, {"id", "path", "source", "spacing_mm", "warm_start"},
            f"subject {entry.get('id')!r}",
        )
        sid = str(entry["id"])
        src = entry.get("source")
        subjects.append(
            SubjectSpec(
                id=sid,
                path=str(entry.get("path") or f"subjects/{sid}.zarr"),
                source=None if src is None else str(src),
                spacing_mm=_voxel(entry.get("spacing_mm"), f"subject {sid!r} spacing_mm"),
                warm_start=_warm_start(entry.get("warm_start"), sid),
            )
        )
    subjects = tuple(subjects)

    lv = dict(raw.get("levels") or {})
    stop = dict(lv.pop("stop", None) or {})
    _refuse_renamed_level_keys(lv)
    _check_keys(
        lv,
        {
            "caps", "coarsest_stages", "seeded_stages", "level_stages",
            "level_params", "shape_update_step", "sharpen_laplacian_levels",
            "min_tissue_fraction", "coarsest_max_disp_frac", "run", "stop_at", "anneal",
        },
        "the 'levels' block",
    )
    _check_keys(
        stop,
        {"residual_frac", "ubar_vox", "residual_vox", "percentile", "require_subvoxel", "plateau"},
        "'levels.stop'",
    )
    plateau = stop.get("plateau")
    plateau_kw: dict[str, Any] = {}
    if plateau is not None:
        if not isinstance(plateau, dict):
            raise ConfigError("levels.stop.plateau must be a mapping {passes, rel, after}")
        _check_keys(plateau, {"passes", "rel", "after"}, "'levels.stop.plateau'")
        if "passes" not in plateau or int(plateau["passes"]) < 1:
            raise ConfigError("levels.stop.plateau needs 'passes' >= 1")
        plateau_kw["plateau_passes"] = int(plateau["passes"])
        rel = float(plateau.get("rel", LevelPolicy.plateau_rel))
        if not 0.0 <= rel < 1.0:
            raise ConfigError(f"levels.stop.plateau.rel must be in [0, 1), got {rel}")
        plateau_kw["plateau_rel"] = rel
        if plateau.get("after") is not None:
            if int(plateau["after"]) < 0:
                raise ConfigError("levels.stop.plateau.after must be >= 0")
            plateau_kw["plateau_after"] = int(plateau["after"])
    anneal = {}
    for k, v in (lv.get("anneal") or {}).items():
        level = _level_key(k, "levels.anneal")
        if not isinstance(v, dict):
            raise ConfigError(f"levels.anneal[{level}] must be a mapping, got {v!r}")
        _check_keys(v, {"smooth_grad_sigma", "smooth_warp_sigma", "passes"}, f"levels.anneal[{level}]")
        entry = {"passes": int(v.get("passes", 1))}
        if entry["passes"] < 1:
            raise ConfigError(f"levels.anneal[{level}].passes must be at least 1")
        for key in ("smooth_grad_sigma", "smooth_warp_sigma"):
            if key in v:
                pair = [float(x) for x in v[key]]
                if len(pair) != 2 or min(pair) <= 0:
                    raise ConfigError(f"levels.anneal[{level}].{key} must be [start, end], both positive")
                entry[key] = pair
        anneal[level] = entry

    level_stages = {
        _level_key(k, "levels.level_stages"): _stages(v, ())
        for k, v in (lv.get("level_stages") or {}).items()
    }
    level_params = {}
    for k, v in (lv.get("level_params") or {}).items():
        level = _level_key(k, "levels.level_params")
        if not isinstance(v, dict):
            raise ConfigError(
                f"levels.level_params[{level}] must be a mapping of stage "
                f"parameters, got {v!r}"
            )
        level_params[level] = dict(v)

    sharpen = lv.get("sharpen_laplacian_levels")
    policy = LevelPolicy(
        caps=tuple(lv.get("caps", LevelPolicy.caps)),
        coarsest_stages=_stages(lv.get("coarsest_stages"), DEFAULT_COARSEST_STAGES),
        seeded_stages=_stages(lv.get("seeded_stages"), DEFAULT_SEEDED_STAGES),
        residual_frac=float(stop.get("residual_frac", LevelPolicy.residual_frac)),
        ubar_vox=float(stop.get("ubar_vox", LevelPolicy.ubar_vox)),
        residual_vox=float(stop.get("residual_vox", LevelPolicy.residual_vox)),
        stop_percentile=float(stop.get("percentile", LevelPolicy.stop_percentile)),
        shape_update_step=float(
            lv.get("shape_update_step", LevelPolicy.shape_update_step)
        ),
        sharpen_laplacian_levels=(
            None if sharpen is None else tuple(int(v) for v in sharpen)
        ),
        min_tissue_fraction=float(
            lv.get("min_tissue_fraction", LevelPolicy.min_tissue_fraction)
        ),
        coarsest_max_disp_frac=float(
            lv.get("coarsest_max_disp_frac", LevelPolicy.coarsest_max_disp_frac)
        ),
        require_subvoxel=bool(stop.get("require_subvoxel", LevelPolicy.require_subvoxel)),
        **plateau_kw,
        anneal=anneal,
        level_stages=level_stages,
        level_params=level_params,
        run=_run_spec(lv.get("run")),
        stop_at=(
            None if lv.get("stop_at") is None
            else parse_level_ref(lv["stop_at"], "levels.stop_at")
        ),
    )

    ing = dict(raw.get("ingest") or {})
    _check_keys(
        ing, {"dtype", "median_radius", "percentiles", "link"}, "the 'ingest' block"
    )
    median = ing.get("median_radius", IngestSpec.median_radius)
    ingest = IngestSpec(
        dtype=str(ing.get("dtype", IngestSpec.dtype)),
        median_radius=None if median is None else float(median),
        percentiles=tuple(
            float(v) for v in ing.get("percentiles", IngestSpec.percentiles)
        ),
        link=bool(ing.get("link", IngestSpec.link)),
    )

    rt = dict(raw.get("retry") or {})
    retry = RetryPolicy(
        fold_frac=float(rt.get("fold_frac", RetryPolicy.fold_frac)),
        ladder=tuple(rt.get("ladder", RetryPolicy.ladder)),
    )

    rn = dict(raw.get("retention") or {})
    keep = rn.get("keep_history_levels")
    retention = Retention(
        keep_history_levels=None if keep is None else tuple(int(v) for v in keep),
        delete_task_arrays=bool(
            rn.get("delete_task_arrays", Retention.delete_task_arrays)
        ),
        delete_accumulators=bool(
            rn.get("delete_accumulators", Retention.delete_accumulators)
        ),
        residual_map_max_voxels=(
            None if rn.get("residual_map_max_voxels") is None
            else int(rn["residual_map_max_voxels"])
        ),
    )

    sl = dict(raw.get("slurm") or {})

    def res(key: str, default: Resources) -> Resources:
        d = dict(sl.get(key) or {})
        return Resources(
            gpus=int(d.get("gpus", default.gpus)),
            cpus=int(d.get("cpus", default.cpus)),
            mem_gb=int(d.get("mem_gb", default.mem_gb)),
            time=str(d.get("time", default.time)),
            chunks_per_task=int(d.get("chunks_per_task", default.chunks_per_task)),
        )

    _wait = sl.get("max_wait_s", SlurmConfig.max_wait_s)
    slurm = SlurmConfig(
        partition=str(sl.get("partition", SlurmConfig.partition)),
        account=sl.get("account"),
        poll_seconds=float(sl.get("poll_seconds", SlurmConfig.poll_seconds)),
        max_wait_s=None if _wait is None else float(_wait),
        register=res("register", SlurmConfig.register),
        blend=res("blend", SlurmConfig.blend),
        update=res("update", SlurmConfig.update),
    )

    gr = dict(raw.get("grid") or {})
    _check_keys(
        gr, {"spacing_mm", "shape", "align", "reference"}, "the 'grid' block"
    )
    grid_spacing = gr.get("spacing_mm", GridRequest.spacing_mm)
    if not isinstance(grid_spacing, str):
        grid_spacing = float(grid_spacing)
    grid = GridRequest(
        spacing_mm=grid_spacing,
        shape=None if gr.get("shape") is None else tuple(int(n) for n in gr["shape"]),
        align=str(gr.get("align", GridRequest.align)),
        reference=(
            None if gr.get("reference") is None else str(gr["reference"])
        ),
    )
    spacing = raw.get("spacing_mm")

    cfg = RunConfig(
        root=str(raw.get("root", ".")),
        subjects=subjects,
        profile=profile,
        profile_name=profile_name,
        spacing_mm=None if spacing is None else float(spacing),
        grid=grid,
        template_subject=(
            None if raw.get("template_subject") is None
            else str(raw["template_subject"])
        ),
        runner=runner_name,
        backend=backend_name,
        levels=policy,
        ingest=ingest,
        retry=retry,
        retention=retention,
        slurm=slurm,
        scratch=None if raw.get("scratch") is None else str(raw["scratch"]),
        gpu_mem_gb=float(raw.get("gpu_mem_gb", 80.0)),
        bytes_per_voxel_channel=float(raw.get("bytes_per_voxel_channel", 90.0)),
        throughput_ch_vox_per_s=float(raw.get("throughput_ch_vox_per_s", 1.5e6)),
        seed=int(raw.get("seed", 0)),
        device=str(raw.get("device", "auto")),
        engine=str(raw.get("engine", "auto")),
        gpus=_gpus(raw.get("gpus", "all")),
        workers_per_gpu=_workers_per_gpu(raw.get("workers_per_gpu", 1)),
    )
    cfg.validate()
    return cfg


def _refuse_renamed_level_keys(lv: dict) -> None:
    """Reject the ``levels`` keys of the coarse-first level numbering.

    Not aliased: a config that still says ``level0_*`` was written when level
    0 was the coarsest, so its other level numbers count the old way too, and
    silently accepting the key would run every one of them at the wrong
    resolution.
    """
    old = [k for k in _RENAMED_LEVEL_KEYS if k in lv]
    if not old:
        return
    renames = ", ".join(f"levels.{k} -> levels.{_RENAMED_LEVEL_KEYS[k]}" for k in old)
    raise ConfigError(
        f"{renames}: renamed with the level numbering. Levels are now numbered "
        f"as OME-Zarr numbers its datasets: level 0 is full resolution and the "
        f"highest level is the coarsest (it was the other way round). Rename "
        f"the key, and renumber every level in the config to match: the keys "
        f"of level_params, level_stages and anneal, the entries of run, "
        f"sharpen_laplacian_levels and retention.keep_history_levels, an "
        f"integer stop_at (a spacing such as \"200um\" needs no change), and "
        f"caps, which is now listed from full resolution towards the coarsest."
    )


def _workers_per_gpu(value) -> int | str:
    if value == "auto":
        return "auto"
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ConfigError(
            f"workers_per_gpu must be 'auto' or a positive integer, got {value!r}"
        )
    try:
        n = int(value)
    except ValueError:
        raise ConfigError(
            f"workers_per_gpu must be 'auto' or a positive integer, got {value!r}"
        ) from None
    if n < 1:
        raise ConfigError(
            f"workers_per_gpu must be 'auto' or a positive integer, got {value!r}"
        )
    return n


def _gpus(value) -> int | str:
    if value == "all":
        return "all"
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ConfigError(f"gpus must be 'all' or a positive integer, got {value!r}")
    try:
        n = int(value)
    except ValueError:
        raise ConfigError(f"gpus must be 'all' or a positive integer, got {value!r}") from None
    if n < 1:
        raise ConfigError(f"gpus must be 'all' or a positive integer, got {value!r}")
    return n

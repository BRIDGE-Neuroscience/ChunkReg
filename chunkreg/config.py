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
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Sequence

from .grid import GridSpec, Profile

__all__ = [
    "StageSpec",
    "LevelPolicy",
    "RetryPolicy",
    "Retention",
    "Resources",
    "SlurmConfig",
    "SubjectSpec",
    "RunConfig",
    "PROFILES",
    "get_profile",
    "load_config",
    "ConfigError",
]


class ConfigError(ValueError):
    """A configuration that cannot be run as written."""


MIN_D_MAX_VOX = 8.0
"""Smallest clamp worth running. Below this a chunk can barely move a feature
past the interpolation footprint, and the level cannot make progress."""


# --------------------------------------------------------------------------- #
# Named profiles
# --------------------------------------------------------------------------- #
PROFILES: dict[str, dict[str, Any]] = {
    # 16 anatomix channels. The reference profile; needs an 80 GB device.
    "a16": dict(channels=16, features="anatomix", r_f=24),
    # anatomix projected to 8 channels by a PCA fitted once at level 0.
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

    def __post_init__(self) -> None:
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


DEFAULT_LEVEL0_STAGES: tuple[StageSpec, ...] = (
    StageSpec(kind="moments", scales=(), iterations=()),
    StageSpec(kind="rigid", scales=(4, 2, 1), iterations=(200, 100, 50), lr=0.01),
    StageSpec(kind="affine", scales=(4, 2, 1), iterations=(200, 100, 50), lr=0.01),
    StageSpec(kind="greedy", scales=(8, 4, 2, 1), iterations=(120, 80, 50, 30)),
)
"""Level 0 is one chunk, so its halo is irrelevant and it can afford the deep
pyramid that estimates the global component nothing else can see."""

DEFAULT_SEEDED_STAGES: tuple[StageSpec, ...] = (
    StageSpec(kind="greedy", scales=(2, 1), iterations=(50, 30)),
)
"""Seeded levels solve a residual the seed already placed within the clamp, so
two scales suffice; a deeper in-chunk pyramid would spend the whole halo on
kernel support (see :meth:`Profile.d_max_vox`)."""


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LevelPolicy:
    """Ceilings and thresholds for the level loop."""

    caps: tuple[int, ...] = (8, 6, 4, 3, 2)
    """Maximum template passes per level, coarse to fine. Padded with its last
    value for deeper pyramids."""
    level0_stages: tuple[StageSpec, ...] = DEFAULT_LEVEL0_STAGES
    seeded_stages: tuple[StageSpec, ...] = DEFAULT_SEEDED_STAGES
    residual_frac: float = 0.5
    """Stop a level once its residual falls below this fraction of the next
    level's clamp, so the finer level's halo can absorb what is left."""
    ubar_vox: float = 1.0
    """Stop once the template recentring step is below this many voxels."""
    stop_percentile: float = 99.0
    """Which percentile of displacement magnitude the stopping rule reads.

    The design note specified 99.9. In practice a small number of voxels sit
    pinned at the clamp for as long as the clamp is active, so a 99.9th
    percentile is bounded below by the clamp itself and the rule can never
    clear. The 99th percentile still ignores nothing that matters and does
    converge."""
    shape_update_step: float = 0.25
    sharpen_laplacian_levels: tuple[int, ...] = (0, 1)
    min_tissue_fraction: float = 0.02
    """Chunks with less foreground than this emit their seed unchanged."""
    level0_max_disp_frac: float = 0.25
    """Level 0 is a single chunk, so no halo bounds it. This bounds it by the
    aperture instead: a displacement approaching the size of the volume cannot
    be a real correspondence, whatever the similarity says."""

    def cap(self, level: int) -> int:
        if not self.caps:
            return 1
        return int(self.caps[min(level, len(self.caps) - 1)])

    def stages(self, level: int) -> tuple[StageSpec, ...]:
        return self.level0_stages if level == 0 else self.seeded_stages


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
    keep_history_levels: tuple[int, ...] = (0, 1)
    delete_task_arrays: bool = True
    delete_accumulators: bool = True


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
    register: Resources = Resources(gpus=1, cpus=8, mem_gb=64, time="04:00:00")
    blend: Resources = Resources(gpus=0, cpus=8, mem_gb=32, time="02:00:00")
    update: Resources = Resources(gpus=0, cpus=8, mem_gb=32, time="01:00:00")


@dataclass(frozen=True)
class SubjectSpec:
    id: str
    path: str


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
    runner: Literal["local", "slurm"] = "local"
    backend: Literal["zarr", "memory"] = "zarr"
    levels: LevelPolicy = LevelPolicy()
    retry: RetryPolicy = RetryPolicy()
    retention: Retention = Retention()
    slurm: SlurmConfig = SlurmConfig()
    gpu_mem_gb: float = 80.0
    bytes_per_voxel_channel: float = 90.0
    """Measured by ``chunkreg calibrate``; 90 is anatomix-on-FireANTs with
    LNCC, 75 with gradient checkpointing."""
    throughput_ch_vox_per_s: float = 1.5e6
    """Also measured by ``calibrate``. The default is a placeholder equal to a
    576-cubed single-channel chunk in 60 seconds."""
    seed: int = 0

    # -- paths -------------------------------------------------------------- #
    @property
    def root_path(self) -> Path:
        return Path(self.root)

    def level_dir(self, level: int) -> Path:
        return self.root_path / "levels" / f"L{level}"

    def template_path(self, level: int) -> Path:
        return self.level_dir(level) / "template.zarr"

    def field_path(self, level: int, subject: str) -> Path:
        return self.level_dir(level) / "fields" / f"{subject}.zarr"

    def recentre_path(self, level: int) -> Path:
        """Where a pass records the template motion the next pass must undo."""
        return self.level_dir(level) / "recentre.zarr"

    def scratch_dir(self, level: int, iteration: int) -> Path:
        return self.root_path / "scratch" / f"L{level}_it{iteration}"

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

    def subject_path(self, subject: str) -> Path:
        for s in self.subjects:
            if s.id == subject:
                p = Path(s.path)
                return p if p.is_absolute() else self.root_path / p
        raise KeyError(f"no subject {subject!r} in this run")

    # -- validation --------------------------------------------------------- #
    def validate(self) -> list[str]:
        """Raise on anything unrunnable; return non-fatal warnings."""
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

        for level, stages in ((0, self.levels.level0_stages), (1, self.levels.seeded_stages)):
            if not stages:
                raise ConfigError(f"level {level} has no stages")
            if not any(s.is_deformable for s in stages):
                warnings.append(
                    f"level {level} has no deformable stage; it can only "
                    f"produce a linear alignment"
                )
            if level > 0:
                for s in stages:
                    if s.is_deformable and s.s_max > p.s_max:
                        raise ConfigError(
                            f"seeded stage {s.kind!r} uses scales up to "
                            f"{s.s_max} but the profile budgets its halo for "
                            f"s_max={p.s_max}. Either set profile scales to "
                            f"{s.scales} (which would leave a clamp of "
                            f"{p.halo - p.support_vox(s.s_max) - p.r_f:.1f} "
                            f"voxels) or lower the stage scales."
                        )

        if not self.subjects:
            raise ConfigError("a run needs at least one subject")
        ids = [s.id for s in self.subjects]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ConfigError(f"duplicate subject ids: {dupes}")

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

        return warnings

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


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _stages(obj, default: tuple[StageSpec, ...]) -> tuple[StageSpec, ...]:
    if obj is None:
        return default
    return tuple(StageSpec.from_obj(o) for o in obj)


def load_config(source: str | Path | dict) -> RunConfig:
    """Read a run configuration from a YAML file or an equivalent mapping."""
    if isinstance(source, (str, Path)):
        import yaml

        raw = yaml.safe_load(Path(source).read_text(encoding="utf-8")) or {}
    else:
        raw = dict(source)

    unknown = set(raw) - {
        "root", "subjects", "profile", "profile_overrides", "spacing_mm", "grid",
        "runner", "backend", "levels", "retry", "retention", "slurm", "gpu_mem_gb",
        "bytes_per_voxel_channel", "throughput_ch_vox_per_s", "seed", "channels",
        "backbone",
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
    profile = get_profile(profile_name, **overrides)

    subjects = tuple(
        SubjectSpec(id=str(s["id"]), path=str(s["path"]))
        for s in (raw.get("subjects") or [])
    )

    lv = dict(raw.get("levels") or {})
    stop = dict(lv.pop("stop", None) or {})
    policy = LevelPolicy(
        caps=tuple(lv.get("caps", LevelPolicy.caps)),
        level0_stages=_stages(lv.get("level0_stages"), DEFAULT_LEVEL0_STAGES),
        seeded_stages=_stages(lv.get("seeded_stages"), DEFAULT_SEEDED_STAGES),
        residual_frac=float(stop.get("residual_frac", LevelPolicy.residual_frac)),
        ubar_vox=float(stop.get("ubar_vox", LevelPolicy.ubar_vox)),
        stop_percentile=float(stop.get("percentile", LevelPolicy.stop_percentile)),
        shape_update_step=float(
            lv.get("shape_update_step", LevelPolicy.shape_update_step)
        ),
        sharpen_laplacian_levels=tuple(
            lv.get("sharpen_laplacian_levels", LevelPolicy.sharpen_laplacian_levels)
        ),
        min_tissue_fraction=float(
            lv.get("min_tissue_fraction", LevelPolicy.min_tissue_fraction)
        ),
    )

    rt = dict(raw.get("retry") or {})
    retry = RetryPolicy(
        fold_frac=float(rt.get("fold_frac", RetryPolicy.fold_frac)),
        ladder=tuple(rt.get("ladder", RetryPolicy.ladder)),
    )

    rn = dict(raw.get("retention") or {})
    retention = Retention(
        keep_history_levels=tuple(
            rn.get("keep_history_levels", Retention.keep_history_levels)
        ),
        delete_task_arrays=bool(
            rn.get("delete_task_arrays", Retention.delete_task_arrays)
        ),
        delete_accumulators=bool(
            rn.get("delete_accumulators", Retention.delete_accumulators)
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

    slurm = SlurmConfig(
        partition=str(sl.get("partition", SlurmConfig.partition)),
        account=sl.get("account"),
        register=res("register", SlurmConfig.register),
        blend=res("blend", SlurmConfig.blend),
        update=res("update", SlurmConfig.update),
    )

    spacing = raw.get("spacing_mm")
    if spacing is None:
        spacing = (raw.get("grid") or {}).get("spacing_mm")

    cfg = RunConfig(
        root=str(raw.get("root", ".")),
        subjects=subjects,
        profile=profile,
        profile_name=profile_name,
        spacing_mm=None if spacing is None else float(spacing),
        runner=raw.get("runner", "local"),
        backend=raw.get("backend", "zarr"),
        levels=policy,
        retry=retry,
        retention=retention,
        slurm=slurm,
        gpu_mem_gb=float(raw.get("gpu_mem_gb", 80.0)),
        bytes_per_voxel_channel=float(raw.get("bytes_per_voxel_channel", 90.0)),
        throughput_ch_vox_per_s=float(raw.get("throughput_ch_vox_per_s", 1.5e6)),
        seed=int(raw.get("seed", 0)),
    )
    cfg.validate()
    return cfg

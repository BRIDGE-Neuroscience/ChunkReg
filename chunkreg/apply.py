"""Using a finished run: warping a volume through its field, and exporting.

Both of these were only ever reachable through the command line, which left
the Python interface able to *build* a template but not to use one. The logic
lives here so that a script and the CLI call the same code.

Warping is blockwise for the same reason every other pass is: the volume a
field was solved for does not fit in memory, and reading each output box with
a margin sized from the field is what keeps the result exact at box edges
rather than fading to zero where the field points outside the box.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import xp as _xp
from .grid import GridSpec, Profile
from .store import Field, Volume

__all__ = [
    "apply_field",
    "export_store",
    "matching_level",
    "scan_view",
    "resample_to_scan",
]


def matching_level(volume: Volume, grid: GridSpec) -> int:
    """The level of ``volume`` that sits on ``grid``.

    The field defines the grid, so the volume has to be read at the level that
    matches it. Taking the level from a flag let the two disagree, which
    produced either a shape error or a silently wrong warp.
    """
    for k in range(volume.n_levels):
        if volume.grid(k).shape == grid.shape:
            return k
    raise ValueError(
        f"no level of {volume.path} is on the field's grid {grid.shape} at "
        f"{grid.spacing_mm} mm; levels are "
        f"{[volume.grid(k).shape for k in range(volume.n_levels)]}"
    )


class _Warped:
    """The warped volume, computed one requested box at a time.

    Quacks like an array for :func:`chunkreg.store.ingest_source`, which pulls
    it core by core and never asks for more than one block at once.
    """

    dtype = np.dtype(np.float32)

    def __init__(self, volume: Volume, level: int, field: Field, grid: GridSpec) -> None:
        self._volume = volume
        self._level = int(level)
        self._field = field
        self._grid = grid
        self.shape = tuple(grid.shape)

    def __getitem__(self, key) -> np.ndarray:
        from .passes._common import warped_subject_block

        origin = tuple(int(k.start) for k in key)
        size = tuple(int(k.stop) - int(k.start) for k in key)
        u = self._field.read_dense(origin, size)
        return _xp.get(
            warped_subject_block(
                self._volume,
                self._level,
                origin,
                size,
                u,
                self._grid.spacing_mm,
                normalise=False,
            )
        )


def apply_field(
    volume: str | Path | Volume,
    field: str | Path | Field,
    out: str | Path,
    level: int | None = None,
    profile: str | Profile = "a16",
    backend: str | None = None,
    target=None,
) -> Volume:
    """Warp a volume through a displacement field into a new store.

    ``level`` defaults to whichever level of the volume is on the field's grid.

    ``target`` names a sampling to deliver the result on instead of the field's
    grid -- for a registration onto a fixed volume, that volume's own store.
    The warp and the resample are composed into one pass rather than run one
    after the other, so the intermediate on the run grid is never written.

    Returns the store that was written.
    """
    from .cohort import Placement, ResampledSource
    from .config import get_profile
    from .store import ingest_source

    vol = volume if isinstance(volume, Volume) else Volume.open(volume, backend)
    fld = field if isinstance(field, Field) else Field.open(field, backend)
    grid = fld.level_grid

    if level is None:
        level = matching_level(vol, grid)
    elif vol.grid(level).shape != grid.shape:
        raise ValueError(
            f"level {level} of {vol.path} is {vol.grid(level).shape} but the "
            f"field is on {grid.shape}; leave level unset to match them"
        )

    source = _Warped(vol, level, fld, grid)
    spacing, origin = grid.spacing_mm, grid.origin_mm
    if target is not None:
        want = _target_placement(target, backend)
        spacing = _isotropic_spacing(want)
        origin = want.origin_mm
        source = ResampledSource(source, Placement.from_grid(grid), want)

    return ingest_source(
        source,
        out,
        spacing,
        profile if isinstance(profile, Profile) else get_profile(profile),
        origin_mm=origin,
        dtype="float32",
        backend=backend,
        overwrite=True,
    )


def export_store(
    store: str | Path | Volume,
    out: str | Path,
    level: int | None = None,
    backend: str | None = None,
    target=None,
) -> None:
    """Write a store's full-resolution level to NIfTI or TIFF.

    ``target`` names a sampling to write on instead of the run grid: a
    :class:`~chunkreg.cohort.Placement`, or a store whose recorded placement
    to take. That is how a result registered onto a fixed volume is written
    out on the fixed scan's own lattice, anisotropic voxels included, rather
    than on the grid the run happened to use.

    Written a group of planes at a time, so exporting does not require the
    volume to fit in memory the way the rest of the pipeline does not.
    """
    from .io_formats import write_volume_blocks

    vol = store if isinstance(store, Volume) else Volume.open(store, backend)
    k = vol.n_levels - 1 if level is None else vol._check_level(level)
    if target is None:
        write_volume_blocks(out, vol.array(k), grid=vol.grid(k))
        return
    view = scan_view(vol, target, level=k, backend=backend)
    write_volume_blocks(
        out, view, voxel_mm=view.target.voxel_mm, origin_mm=view.target.origin_mm
    )


# --------------------------------------------------------------------------- #
# Back to the lattice a scan arrived on
# --------------------------------------------------------------------------- #
def _target_placement(target, backend=None):
    """A :class:`~chunkreg.cohort.Placement` from whatever names one."""
    from .cohort import Placement

    if isinstance(target, Placement):
        return target
    vol = target if isinstance(target, Volume) else Volume.open(target, backend)
    placed = vol.placement
    if placed is None:
        # Every store knows its run grid, so falling back to it is exact for a
        # scan that was already on the grid and is the only sensible reading
        # for a store written without a placement at all.
        return Placement.from_grid(vol.native_grid)
    return placed


def _isotropic_spacing(placement) -> float:
    """The one voxel size a chunkreg store can be written on, or an error."""
    voxel = placement.voxel_mm
    if max(voxel) > min(voxel) * (1 + 1e-6):
        raise ValueError(
            f"the target scan's voxels are {voxel[0]:g} x {voxel[1]:g} x "
            f"{voxel[2]:g} mm. A chunkreg store is on an isotropic grid, so "
            f"there is none to write this on; export it to a file instead, "
            f"which carries the voxel size in its header."
        )
    return float(voxel[0])


def scan_view(
    volume: str | Path | Volume,
    target,
    level: int | None = None,
    backend: str | None = None,
):
    """``volume`` presented on the lattice a scan arrived on, read box by box.

    Everything the pipeline produces sits on the run grid, which is a grid
    chosen for the cohort rather than the sampling any one scan came in on.
    ``target`` names the sampling to go back to: a
    :class:`~chunkreg.cohort.Placement`, or a store whose recorded placement
    to take, which for a registration onto a fixed volume is that volume's own
    store.

    Quacks like an array, so the result streams into a store or a file without
    ever being held whole.
    """
    from .cohort import Placement, ResampledSource

    vol = volume if isinstance(volume, Volume) else Volume.open(volume, backend)
    k = vol.n_levels - 1 if level is None else vol._check_level(level)
    return ResampledSource(
        vol.array(k),
        Placement.from_grid(vol.grid(k)),
        _target_placement(target, backend),
    )


def resample_to_scan(
    volume: str | Path | Volume,
    target,
    out: str | Path,
    level: int | None = None,
    profile: str | Profile = "a16",
    backend: str | None = None,
) -> Volume:
    """Write a store back onto the lattice a scan arrived on.

    The last step of registering onto a particular volume: the field, the
    warped result and the template are all on the run grid, and what the
    answer is wanted on is the fixed scan's own sampling.

    With ``grid.reference`` naming that scan, the run grid already *is* its
    lattice and this copies rather than interpolates. Without it the run grid
    is a compromise across the cohort and this is a real resample.

    A chunkreg store is written, so the scan's sampling has to be isotropic;
    an anisotropic one has no chunkreg grid to be written on, and
    :func:`export_store` takes it straight to a file instead.
    """
    from .config import get_profile
    from .store import ingest_source

    view = scan_view(volume, target, level=level, backend=backend)
    return ingest_source(
        view,
        out,
        _isotropic_spacing(view.target),
        profile if isinstance(profile, Profile) else get_profile(profile),
        origin_mm=view.target.origin_mm,
        dtype="float32",
        backend=backend,
        overwrite=True,
    )

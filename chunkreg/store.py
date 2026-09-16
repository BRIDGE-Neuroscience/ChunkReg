"""On-disk objects: multiscale images, displacement fields, task residuals.

Three store types cover everything the pipeline writes:

``Volume``
    A multiscale scalar image, levels ``s0`` (coarsest, one chunk) to ``sK``
    (native). Every pyramid level reads from the same store.

``Field``
    A displacement field for one subject at one level, stored on a lattice
    coarser than the level by ``profile.lattice_factor`` and read back dense.

``TaskArray``
    The residuals produced by one register task, as a single regular array so
    that a task writes exactly one object.

Sharding is chosen so that **one shard is one chunk core**. That is what makes
every downstream pass owner-computes: a task writes only the shards it owns,
no two tasks touch the same file, and no locking is required anywhere.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from . import backend as _backend
from . import fields as _fields
from . import xp as _xp
from .grid import Chunk, GridSpec, Profile, pyramid, pyramid_depth, shard_grid, tile

__all__ = [
    "Volume",
    "Field",
    "TaskArray",
    "TaskEntry",
    "read_padded",
    "lattice_box",
    "ingest_array",
    "ingest_source",
    "check_volume",
]

_META = "meta.json"


# --------------------------------------------------------------------------- #
# Block reads that survive the volume boundary
# --------------------------------------------------------------------------- #
def read_padded(
    arr,
    origin: Sequence[int],
    shape: Sequence[int],
    lead: int = 0,
    mode: str = "constant",
    cval: float = 0.0,
    dtype=np.float32,
) -> np.ndarray:
    """Read a spatial box, filling anything outside the array.

    ``lead`` is the number of leading non-spatial axes (1 for a ``(3, Z, Y, X)``
    field). ``mode`` is ``"constant"`` for images, where outside means absent,
    or ``"edge"`` for fields, where outside should extend the last valid vector
    rather than snap to zero.
    """
    full = np.asarray(arr.shape[lead:], dtype=np.int64)
    o = np.asarray(origin, dtype=np.int64)
    s = np.asarray(shape, dtype=np.int64)
    if o.size != 3 or s.size != 3:
        raise ValueError("origin and shape must be 3D")
    lead_shape = tuple(int(n) for n in arr.shape[:lead])
    out_shape = lead_shape + tuple(int(n) for n in s)

    lo = np.clip(o, 0, full)
    hi = np.clip(o + s, 0, full)
    if np.any(hi <= lo):
        return np.full(out_shape, cval, dtype=dtype)

    key = (slice(None),) * lead + tuple(
        slice(int(a), int(b)) for a, b in zip(lo, hi)
    )
    sub = np.asarray(arr[key], dtype=dtype)
    before = lo - o
    after = (o + s) - hi
    if np.any(before > 0) or np.any(after > 0):
        pad = [(0, 0)] * lead + [
            (int(b), int(a)) for b, a in zip(before, after)
        ]
        if mode == "edge":
            sub = np.pad(sub, pad, mode="edge")
        else:
            sub = np.pad(sub, pad, mode="constant", constant_values=cval)
    return sub


def write_block(arr, origin: Sequence[int], block: np.ndarray, lead: int = 0) -> None:
    """Write a block, clipping anything that falls outside the array.

    A device array is brought back to the host here, at the last moment.
    """
    block = _xp.get(block)
    full = np.asarray(arr.shape[lead:], dtype=np.int64)
    o = np.asarray(origin, dtype=np.int64)
    s = np.asarray(block.shape[lead:], dtype=np.int64)
    lo = np.clip(o, 0, full)
    hi = np.clip(o + s, 0, full)
    if np.any(hi <= lo):
        return
    b_lo = lo - o
    b_hi = b_lo + (hi - lo)
    src = (slice(None),) * lead + tuple(
        slice(int(a), int(b)) for a, b in zip(b_lo, b_hi)
    )
    dst = (slice(None),) * lead + tuple(
        slice(int(a), int(b)) for a, b in zip(lo, hi)
    )
    arr[dst] = block[src].astype(arr.dtype, copy=False)


def lattice_box(
    origin: Sequence[int], shape: Sequence[int], factor: int
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Lattice extent covering a level-resolution box.

    Lattice voxel ``j`` covers level voxels ``[j*f, (j+1)*f)``. Chunk cores and
    halos are both multiples of the factor (enforced by :class:`Profile`), so a
    chunk's lattice extent is exact rather than rounded.
    """
    o = np.asarray(origin, dtype=np.int64)
    s = np.asarray(shape, dtype=np.int64)
    lo = o // factor
    hi = -(-(o + s) // factor)  # ceil division
    return tuple(int(v) for v in lo), tuple(int(v) for v in (hi - lo))


# --------------------------------------------------------------------------- #
# Volume: multiscale scalar image
# --------------------------------------------------------------------------- #
class Volume:
    """A scalar image as a multiscale pyramid of sharded arrays.

    Levels are named ``s0`` (coarsest) through ``sK`` (native), matching the
    pyramid index used everywhere else, so level ``k`` of a run reads
    ``volume.array(k)`` with no translation.
    """

    def __init__(self, path, meta: dict, backend=None) -> None:
        self.path = Path(path)
        self.meta = meta
        self._backend = _backend.get_backend(backend)
        self._cache: dict[tuple[int, str], Any] = {}

    # -- construction ------------------------------------------------------- #
    @classmethod
    def create(
        cls,
        path,
        native: GridSpec,
        profile: Profile,
        dtype: str = "uint16",
        backend=None,
        overwrite: bool = False,
        provenance: dict | None = None,
        n_levels: int | None = None,
    ) -> "Volume":
        """Create a multiscale store.

        ``n_levels`` overrides the pyramid depth. A template belongs to exactly
        one pyramid level, so its store is created with one level rather than a
        pyramid of its own.
        """
        b = _backend.get_backend(backend)
        if n_levels is None:
            grids = pyramid(native, profile)
        else:
            if n_levels < 1:
                raise ValueError(f"n_levels must be >= 1, got {n_levels}")
            grids = [native.coarsened(2 ** (n_levels - 1 - k)) for k in range(n_levels)]
        meta = {
            "kind": "volume",
            "native_shape": list(native.shape),
            "spacing_mm": native.spacing_mm,
            "origin_mm": list(native.origin_mm),
            "n_levels": len(grids),
            "dtype": str(np.dtype(dtype)),
            "core": profile.core,
            "inner_chunk": profile.inner_chunk,
            "norm_lo": None,
            "norm_hi": None,
            "provenance": provenance or {},
        }
        for k, g in enumerate(grids):
            chunk = [min(profile.inner_chunk, n) for n in g.shape]
            shard = [min(profile.core, n) for n in g.shape]
            shard = [int(math.ceil(s / c) * c) for s, c in zip(shard, chunk)]
            b.create(
                Path(path) / f"s{k}",
                g.shape,
                dtype,
                chunks=chunk,
                shards=shard,
                overwrite=overwrite,
            )
        _backend.write_json(Path(path) / _META, meta, b)
        return cls(path, meta, b)

    @classmethod
    def open(cls, path, backend=None) -> "Volume":
        b = _backend.get_backend(backend)
        meta = _backend.read_json(Path(path) / _META, b)
        if meta.get("kind") != "volume":
            raise ValueError(f"{path} is not a volume store")
        return cls(path, meta, b)

    @classmethod
    def exists(cls, path, backend=None) -> bool:
        return _backend.json_exists(Path(path) / _META, _backend.get_backend(backend))

    # -- geometry ----------------------------------------------------------- #
    @property
    def n_levels(self) -> int:
        return int(self.meta["n_levels"])

    @property
    def native_grid(self) -> GridSpec:
        return GridSpec(
            tuple(self.meta["native_shape"]),
            self.meta["spacing_mm"],
            tuple(self.meta["origin_mm"]),
        )

    def grid(self, level: int) -> GridSpec:
        k = self._check_level(level)
        return self.native_grid.coarsened(2 ** (self.n_levels - 1 - k))

    def grids(self) -> list[GridSpec]:
        return [self.grid(k) for k in range(self.n_levels)]

    def _check_level(self, level: int) -> int:
        k = int(level)
        if not 0 <= k < self.n_levels:
            raise IndexError(
                f"level {k} out of range for a {self.n_levels}-level volume"
            )
        return k

    def array(self, level: int, mode: str = "r"):
        """Open a level, caching one handle per (level, mode).

        The mode is passed through rather than ignored: a caller that asks for
        a read-only handle gets one, so a pass that is supposed to be reading
        cannot quietly write through it.
        """
        k = self._check_level(level)
        key = (k, str(mode))
        if key not in self._cache:
            self._cache[key] = self._backend.open(self.path / f"s{k}", mode=mode)
        return self._cache[key]

    # -- normalisation ------------------------------------------------------ #
    def set_normalisation(self, lo: float, hi: float) -> None:
        """Record the global intensity window for this volume.

        Normalisation is per volume and global, computed once at the coarsest
        level. Re-deriving it per chunk would make features drift across chunk
        boundaries, which the blend cannot repair.
        """
        if not hi > lo:
            raise ValueError(f"need hi > lo, got lo={lo} hi={hi}")
        self.meta["norm_lo"] = float(lo)
        self.meta["norm_hi"] = float(hi)
        _backend.write_json(self.path / _META, self.meta, self._backend)

    @property
    def normalisation(self) -> tuple[float, float] | None:
        lo, hi = self.meta.get("norm_lo"), self.meta.get("norm_hi")
        return None if lo is None or hi is None else (float(lo), float(hi))

    def compute_normalisation(
        self, level: int | None = None, percentiles=(0.5, 99.5), mask_frac: float = 0.0
    ) -> tuple[float, float]:
        """Derive and store the intensity window from a coarse level."""
        k = 0 if level is None else self._check_level(level)
        if _xp.uses_torch():
            lo, hi = self._window_on_device(k, percentiles, mask_frac)
            self.set_normalisation(lo, hi)
            return self.normalisation  # type: ignore[return-value]
        a = np.asarray(self.array(k)[:], dtype=np.float32)
        sel = a > (a.max() * mask_frac) if mask_frac > 0 else np.ones_like(a, bool)
        if not sel.any():
            sel = np.ones_like(a, bool)
        lo, hi = np.percentile(a[sel], percentiles)
        if not hi > lo:
            lo, hi = float(a.min()), float(a.max()) + 1e-6
        self.set_normalisation(float(lo), float(hi))
        return self.normalisation  # type: ignore[return-value]

    def _window_on_device(self, k: int, percentiles, mask_frac: float):
        from . import gpu_ops

        a = _xp.put(self.array(k)[:])
        sel = a > (a.max() * mask_frac) if mask_frac > 0 else None
        vals = a[sel] if sel is not None and bool(sel.any()) else a
        lo, hi = (gpu_ops.percentile(vals, q) for q in percentiles)
        if not hi > lo:
            lo, hi = float(a.min()), float(a.max()) + 1e-6
        return float(lo), float(hi)

    def normalise(self, block: np.ndarray) -> np.ndarray:
        """Apply the stored window, clipped to ``[0, 1]``."""
        norm = self.normalisation
        if norm is None:
            raise RuntimeError(
                f"{self.path} has no normalisation; "
                "call compute_normalisation() during ingest"
            )
        lo, hi = norm
        if _xp.is_tensor(block):
            return ((_xp.to_float32(block) - lo) / (hi - lo)).clamp_(0.0, 1.0)
        out = (np.asarray(block, dtype=np.float32) - lo) / (hi - lo)
        return np.clip(out, 0.0, 1.0, out=out)

    # -- reads and writes --------------------------------------------------- #
    def read_padded(
        self, level: int, origin, shape, normalise: bool = False
    ) -> np.ndarray:
        """Read a box at ``level``, zero outside the volume.

        The block comes back where this process computes: on the GPU when the
        run's device is one, so reading is the only host step.
        """
        block = _xp.put(
            read_padded(self.array(level), origin, shape, lead=0, mode="constant")
        )
        return self.normalise(block) if normalise else block

    def read_chunk(self, level: int, chunk: Chunk, normalise: bool = True) -> np.ndarray:
        """Read a chunk's padded box, the unit a register task consumes."""
        return self.read_padded(level, chunk.pad_origin, chunk.pad_shape, normalise)

    def write_block(self, level: int, origin, block: np.ndarray) -> None:
        write_block(self.array(level, "r+"), origin, block, lead=0)

    def build_pyramid(self, block: int = 256) -> None:
        """Fill levels ``K-1 .. 0`` by 2x mean pooling, coarsest last.

        Runs blockwise so a volume far larger than memory can be pooled, and
        pools from the level above rather than from native so each level costs
        an eighth of the one before it.
        """
        for k in range(self.n_levels - 2, -1, -1):
            src = self.array(k + 1)
            dst = self.array(k, "r+")
            dshape = dst.shape
            for z0 in range(0, dshape[0], block):
                for y0 in range(0, dshape[1], block):
                    for x0 in range(0, dshape[2], block):
                        o = (z0, y0, x0)
                        s = tuple(
                            min(block, dshape[d] - o[d]) for d in range(3)
                        )
                        fine = read_padded(
                            src,
                            tuple(2 * v for v in o),
                            tuple(2 * v for v in s),
                            mode="edge",
                        )
                        if _xp.uses_torch():
                            pooled = _fields.pool_field(_xp.put(fine)[None], 2)[0]
                        else:
                            pooled = fine.reshape(
                                s[0], 2, s[1], 2, s[2], 2
                            ).mean(axis=(1, 3, 5))
                        write_block(dst, o, pooled)


# --------------------------------------------------------------------------- #
# Field: a displacement for one subject at one level
# --------------------------------------------------------------------------- #
class Field:
    """A displacement field on a storage lattice, read back dense.

    Vectors are millimetres, so the lattice is purely a sampling decision:
    coarsening it loses high-frequency detail in the *field*, never the
    physical meaning of its values, and promoting between pyramid levels is
    exact.
    """

    def __init__(self, path, meta: dict, backend=None) -> None:
        self.path = Path(path)
        self.meta = meta
        self._backend = _backend.get_backend(backend)
        self._array = None

    @classmethod
    def create(
        cls,
        path,
        level: GridSpec,
        profile: Profile,
        backend=None,
        overwrite: bool = False,
        subject: str | None = None,
    ) -> "Field":
        b = _backend.get_backend(backend)
        f = profile.lattice_factor
        lat = level.coarsened(f)
        inner = max(1, profile.inner_chunk // f)
        shard = max(inner, profile.core // f)
        chunks = (3,) + tuple(min(inner, n) for n in lat.shape)
        shards = (3,) + tuple(
            int(math.ceil(min(shard, n) / c) * c)
            for n, c in zip(lat.shape, chunks[1:])
        )
        b.create(
            path,
            (3,) + lat.shape,
            _fields.DISP_DTYPE,
            chunks=chunks,
            shards=shards,
            overwrite=overwrite,
        )
        meta = {
            "kind": "field",
            "level_shape": list(level.shape),
            "level_spacing_mm": level.spacing_mm,
            "level_origin_mm": list(level.origin_mm),
            "lattice_factor": int(f),
            "subject": subject,
        }
        _backend.write_json(Path(str(path) + ".meta.json"), meta, b)
        return cls(path, meta, b)

    @classmethod
    def open(cls, path, backend=None) -> "Field":
        b = _backend.get_backend(backend)
        meta = _backend.read_json(Path(str(path) + ".meta.json"), b)
        if meta.get("kind") != "field":
            raise ValueError(f"{path} is not a field store")
        return cls(path, meta, b)

    @classmethod
    def exists(cls, path, backend=None) -> bool:
        return _backend.json_exists(
            Path(str(path) + ".meta.json"), _backend.get_backend(backend)
        )

    # -- geometry ----------------------------------------------------------- #
    @property
    def factor(self) -> int:
        return int(self.meta["lattice_factor"])

    @property
    def level_grid(self) -> GridSpec:
        return GridSpec(
            tuple(self.meta["level_shape"]),
            self.meta["level_spacing_mm"],
            tuple(self.meta["level_origin_mm"]),
        )

    @property
    def lattice_grid(self) -> GridSpec:
        return self.level_grid.coarsened(self.factor)

    @property
    def array(self):
        if self._array is None:
            self._array = self._backend.open(self.path, mode="r+")
        return self._array

    # -- reads and writes --------------------------------------------------- #
    def read_lattice(self, origin, shape) -> np.ndarray:
        """Raw lattice samples, edge-replicated outside."""
        return read_padded(self.array, origin, shape, lead=1, mode="edge")

    def read_dense(self, origin, shape, margin: int = 2) -> np.ndarray:
        """Read a level-resolution box, upsampled from the lattice.

        ``origin`` and ``shape`` are in level voxels. A margin of lattice
        samples is read around the box so the trilinear upsample has support at
        its edges.
        """
        level = self.level_grid
        lat = self.lattice_grid
        o = np.asarray(origin, dtype=np.int64)
        s = np.asarray(shape, dtype=np.int64)

        lo, ext = lattice_box(o, s, self.factor)
        lo = np.asarray(lo, dtype=np.int64) - margin
        ext = np.asarray(ext, dtype=np.int64) + 2 * margin
        block = _xp.put(self.read_lattice(lo, ext))

        sub_lat = GridSpec(
            tuple(int(v) for v in ext), lat.spacing_mm, tuple(lat.world(lo))
        )
        sub_level = GridSpec(
            tuple(int(v) for v in s), level.spacing_mm, tuple(level.world(o))
        )
        return _fields.resample_field(block, sub_lat, sub_level)

    def read_dense_on(
        self, target: GridSpec, origin, shape, margin: int = 2
    ) -> np.ndarray:
        """Read this field resampled onto a box of a *different* grid.

        Used by promote: the target is the next pyramid level, which shares a
        world frame with this one but samples it twice as finely. Since the
        stored vectors are millimetres, this only re-samples the lattice, so
        the seed the finer level receives is the coarse solution exactly.
        """
        lat = self.lattice_grid
        o = np.asarray(origin, dtype=np.int64)
        s = np.asarray(shape, dtype=np.int64)
        sub_target = GridSpec(
            tuple(int(v) for v in s), target.spacing_mm, tuple(target.world(o))
        )
        lo_world = sub_target.world(np.zeros(3))
        hi_world = sub_target.world(s - 1)
        lat_lo = np.floor(lat.voxel(lo_world)).astype(np.int64) - margin
        lat_hi = np.ceil(lat.voxel(hi_world)).astype(np.int64) + margin + 1
        ext = np.maximum(lat_hi - lat_lo, 1)
        block = _xp.put(self.read_lattice(lat_lo, ext))
        sub_lat = GridSpec(
            tuple(int(v) for v in ext), lat.spacing_mm, tuple(lat.world(lat_lo))
        )
        return _fields.resample_field(block, sub_lat, sub_target)

    def write_lattice(self, origin, block: np.ndarray) -> None:
        write_block(self.array, origin, block, lead=1)

    def write_dense(self, origin, block: np.ndarray) -> None:
        """Downsample a level-resolution block onto the lattice and write it.

        ``origin`` must be a multiple of the lattice factor, which every chunk
        core origin is by construction.
        """
        f = self.factor
        o = np.asarray(origin, dtype=np.int64)
        if np.any(o % f):
            raise ValueError(
                f"origin {tuple(int(v) for v in o)} is not aligned to the "
                f"lattice factor {f}"
            )
        # Pooled where the block lives; only the lattice goes back to the host.
        self.write_lattice(o // f, _fields.pool_field(block, f))

    def fill_identity(self) -> None:
        self.array[:] = 0


# --------------------------------------------------------------------------- #
# TaskArray: the output of one register task
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TaskEntry:
    """One (subject, chunk) result inside a task array."""

    subject: str
    chunk_id: int
    chunk_index: tuple[int, int, int]
    pad_origin: tuple[int, int, int]
    pad_shape: tuple[int, int, int]

    def lattice(self, factor: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        return lattice_box(self.pad_origin, self.pad_shape, factor)

    def to_json(self) -> dict:
        return {
            "subject": self.subject,
            "chunk_id": int(self.chunk_id),
            "chunk_index": list(self.chunk_index),
            "pad_origin": list(self.pad_origin),
            "pad_shape": list(self.pad_shape),
        }

    @classmethod
    def from_json(cls, d: dict) -> "TaskEntry":
        return cls(
            subject=d["subject"],
            chunk_id=int(d["chunk_id"]),
            chunk_index=tuple(d["chunk_index"]),
            pad_origin=tuple(d["pad_origin"]),
            pad_shape=tuple(d["pad_shape"]),
        )


class TaskArray:
    """Residual fields for a contiguous range of (subject, chunk) work.

    One task writes one *array*: one directory, one metadata document, one
    completion marker. Within it each entry is its own shard.

    Sharding per entry rather than per task is deliberate. A blend task needs
    the residuals of the chunk owning its shard plus up to 26 neighbours, and
    those neighbours are scattered across many task arrays because tasks are
    ranges over (subject, chunk) while neighbours are adjacent in 3D. With one
    shard per task a blend task would decompress a whole task array to extract
    a single entry from it, roughly fifteen-fold read amplification at the
    production profile. One shard per entry makes the read exact. Task arrays
    are transient and deleted once the blend that consumed them has verified.

    Entries are stored in a single regular array of the maximum padded lattice
    extent and trimmed on read using the recorded per-entry shape, so boundary
    chunks cost a little padding in exchange for a regular layout.

    The ``complete`` flag is the idempotency marker: a re-run of a task whose
    array is already complete does no work, which is what makes resubmitting a
    failed scheduler array element safe.
    """

    def __init__(self, path, meta: dict, backend=None) -> None:
        self.path = Path(path)
        self.meta = meta
        self._backend = _backend.get_backend(backend)
        self._array = None
        # Parsed once. Every read and write indexes into this list, so rebuilding
        # it per access made a task quadratic in the number of entries it holds.
        self._entries = tuple(TaskEntry.from_json(d) for d in meta["entries"])

    @classmethod
    def create(
        cls,
        path,
        entries: Sequence[TaskEntry],
        profile: Profile,
        backend=None,
        overwrite: bool = True,
        level: int | None = None,
        iteration: int | None = None,
    ) -> "TaskArray":
        b = _backend.get_backend(backend)
        f = profile.lattice_factor
        # Small inner chunks inside one shard per entry. A blend task reads
        # only the thin strip where a neighbour's halo overlaps its core, and
        # with the whole entry as one chunk every such read decompressed the
        # entire entry, up to 27 times per core.
        inner = max(1, profile.inner_chunk // f)
        cap = -(-int(math.ceil(profile.padded / f)) // inner) * inner
        shape = (len(entries), 3, cap, cap, cap)
        b.create(
            path,
            shape,
            _fields.DISP_DTYPE,
            chunks=(1, 3, inner, inner, inner),
            shards=(1, 3, cap, cap, cap),
            overwrite=overwrite,
        )
        meta = {
            "kind": "task",
            "lattice_factor": int(f),
            "capacity": cap,
            "level": level,
            "iteration": iteration,
            "complete": False,
            "entries": [e.to_json() for e in entries],
        }
        _backend.write_json(Path(str(path) + ".meta.json"), meta, b)
        return cls(path, meta, b)

    @classmethod
    def open(cls, path, backend=None) -> "TaskArray":
        b = _backend.get_backend(backend)
        meta = _backend.read_json(Path(str(path) + ".meta.json"), b)
        if meta.get("kind") != "task":
            raise ValueError(f"{path} is not a task array")
        return cls(path, meta, b)

    @classmethod
    def is_complete(cls, path, backend=None) -> bool:
        """True when a finished task array exists, so the task can be skipped.

        The array itself has to be there, not just the marker. Retention
        deletes finished task arrays, and a marker that outlived its data would
        otherwise let a re-run skip work whose output is gone and fail in the
        blend that went looking for it.
        """
        b = _backend.get_backend(backend)
        p = Path(str(path) + ".meta.json")
        if not _backend.json_exists(p, b) or not b.exists(path):
            return False
        try:
            return bool(_backend.read_json(p, b).get("complete", False))
        except (OSError, ValueError):
            return False

    @property
    def entries(self) -> tuple[TaskEntry, ...]:
        return self._entries

    @property
    def factor(self) -> int:
        return int(self.meta["lattice_factor"])

    @property
    def complete(self) -> bool:
        return bool(self.meta.get("complete", False))

    @property
    def array(self):
        if self._array is None:
            self._array = self._backend.open(self.path, mode="r+")
        return self._array

    def write(self, i: int, block_lattice: np.ndarray) -> None:
        """Store entry ``i``'s residual, given on its padded lattice extent."""
        e = self.entries[i]
        _, ext = e.lattice(self.factor)
        u = np.asarray(_xp.get(block_lattice), dtype=np.float32)
        if u.shape != (3,) + ext:
            raise ValueError(
                f"entry {i} expects {(3,) + ext} on the lattice, got {u.shape}"
            )
        cap = int(self.meta["capacity"])
        if any(n > cap for n in ext):
            raise ValueError(f"entry {i} extent {ext} exceeds capacity {cap}")
        self.array[i, :, : ext[0], : ext[1], : ext[2]] = u.astype(
            self.array.dtype, copy=False
        )

    def read_box(self, i: int, lo, hi) -> np.ndarray:
        """Part of entry ``i``'s lattice, ``[lo, hi)`` per axis.

        Only the inner chunks the box touches are decompressed.
        """
        _, ext = self.entries[i].lattice(self.factor)
        a = [max(0, int(v)) for v in lo]
        b = [min(int(v), int(n)) for v, n in zip(hi, ext)]
        return np.asarray(
            self.array[i, :, a[0] : b[0], a[1] : b[1], a[2] : b[2]], dtype=np.float32
        )

    def read(self, i: int) -> np.ndarray:
        e = self.entries[i]
        _, ext = e.lattice(self.factor)
        return np.asarray(
            self.array[i, :, : ext[0], : ext[1], : ext[2]], dtype=np.float32
        )

    def mark_complete(self) -> None:
        self.meta["complete"] = True
        _backend.write_json(
            Path(str(self.path) + ".meta.json"), self.meta, self._backend
        )


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #
def _output_dtype(requested: str, source_dtype) -> np.dtype:
    """Resolve ``"auto"`` to a storage dtype that loses nothing.

    Integer sources keep their type. Floating sources are stored as float32,
    which is what every pass reads anyway. A fixed request is honoured.
    """
    if str(requested) != "auto":
        return np.dtype(requested)
    src = np.dtype(source_dtype)
    if src == np.bool_:
        return np.dtype(np.uint8)
    if np.issubdtype(src, np.integer):
        return src
    return np.dtype(np.float32)


def _cast(block: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Cast for storage, clipping rather than wrapping into an integer type."""
    if np.issubdtype(dtype, np.integer) and not np.issubdtype(block.dtype, np.integer):
        info = np.iinfo(dtype)
        block = np.clip(np.rint(block), info.min, info.max)
    return block.astype(dtype, copy=False)


def ingest_source(
    source,
    path,
    spacing_mm: float,
    profile: Profile,
    origin_mm: Sequence[float] = (0.0, 0.0, 0.0),
    dtype: str = "auto",
    backend=None,
    overwrite: bool = False,
    percentiles=(0.5, 99.5),
    provenance: dict | None = None,
    median_radius: float | None = None,
) -> Volume:
    """Write a volume as a sharded multiscale store, one chunk core at a time.

    ``source`` is anything with a 3D ``shape``, a ``dtype`` and ``(z, y, x)``
    slicing: a numpy array, a zarr array, or an
    :class:`chunkreg.io_formats.VolumeSource`. Only one core-sized block, plus
    the median margin, is in memory at once, and each write fills exactly one
    shard of the native level.

    The optional median denoise is applied blockwise with a margin of half its
    width read from the neighbours, so the result is identical to filtering
    the whole volume at once.
    """
    shape = tuple(int(n) for n in source.shape)
    if len(shape) != 3:
        raise ValueError(f"expected a 3D volume, got shape {shape}")
    out_dtype = _output_dtype(dtype, source.dtype)
    native = GridSpec(shape, spacing_mm, tuple(origin_mm))
    vol = Volume.create(
        path,
        native,
        profile,
        dtype=str(out_dtype),
        backend=backend,
        overwrite=overwrite,
        provenance=provenance,
    )
    level = vol.n_levels - 1

    size = int(2 * median_radius + 1) if median_radius else 0
    margin = size // 2
    step = int(profile.core)
    for z0 in range(0, shape[0], step):
        for y0 in range(0, shape[1], step):
            for x0 in range(0, shape[2], step):
                origin = (z0, y0, x0)
                lo = [max(o - margin, 0) for o in origin]
                hi = [min(o + step + margin, n) for o, n in zip(origin, shape)]
                block = np.asarray(
                    source[tuple(slice(a, b) for a, b in zip(lo, hi))]
                )
                if size > 1:
                    if _xp.uses_torch():
                        from . import gpu_ops

                        block = _xp.get(gpu_ops.median_filter(_xp.put(block), size))
                    else:
                        from scipy import ndimage

                        block = ndimage.median_filter(block, size=size)
                core = tuple(
                    slice(o - a, min(o + step, n) - a)
                    for o, a, n in zip(origin, lo, shape)
                )
                vol.write_block(level, origin, _cast(block[core], out_dtype))

    vol.build_pyramid()
    vol.compute_normalisation(level=0, percentiles=percentiles)
    return vol


def ingest_array(
    data: np.ndarray,
    path,
    spacing_mm: float,
    profile: Profile,
    origin_mm: Sequence[float] = (0.0, 0.0, 0.0),
    dtype: str = "auto",
    backend=None,
    overwrite: bool = False,
    percentiles=(0.5, 99.5),
    provenance: dict | None = None,
) -> Volume:
    """Write an in-memory volume as a multiscale store and derive its window.

    A thin wrapper over :func:`ingest_source`, kept because tests and small
    runs have an array in hand rather than a file.
    """
    arr = _xp.get(data)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {arr.shape}")
    return ingest_source(
        arr,
        path,
        spacing_mm,
        profile,
        origin_mm=origin_mm,
        dtype=dtype,
        backend=backend,
        overwrite=overwrite,
        percentiles=percentiles,
        provenance=provenance,
    )


def check_volume(vol: Volume, profile: Profile) -> tuple[list[str], list[str]]:
    """Check a subject store against the profile a run will use.

    Returns ``(errors, warnings)``. A pyramid of the wrong depth is an error,
    because level ``k`` of the run would read the wrong resolution. Sharding
    that does not match the chunk core is a warning: the run still works, but
    tasks no longer map one to one onto files, which is what keeps IO and
    file counts bounded on a cluster.
    """
    errors: list[str] = []
    warnings: list[str] = []
    grids = pyramid(vol.native_grid, profile)
    if vol.n_levels != len(grids):
        errors.append(
            f"has {vol.n_levels} levels but this profile needs {len(grids)} "
            f"(core {profile.core}); it was ingested with a different profile"
        )
        return errors, warnings
    for k, g in enumerate(grids):
        arr = vol.array(k)
        chunks = tuple(getattr(arr, "chunks", ()) or ())
        shards = getattr(arr, "shards", None)
        want_chunk = tuple(min(profile.inner_chunk, n) for n in g.shape)
        want_shard = tuple(
            int(math.ceil(min(profile.core, n) / c) * c)
            for n, c in zip(g.shape, want_chunk)
        )
        if shards is None:
            warnings.append(f"level s{k} is not sharded")
        elif tuple(shards) != want_shard:
            warnings.append(
                f"level s{k} has shards {tuple(shards)}, expected {want_shard} "
                f"(one shard per {profile.core}-voxel chunk core)"
            )
        if chunks and chunks != want_chunk:
            warnings.append(
                f"level s{k} has inner chunks {chunks}, expected {want_chunk}"
            )
    return errors, warnings

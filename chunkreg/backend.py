"""Array backends.

Every volumetric object in ``chunkreg`` is a chunked, sharded array addressed
by a path. Zarr v3 is the production backend and the only one a real run
should use: sharding is what keeps a 12.6 Gvox volume at ~100 files instead of
~6000, and nothing else in the ecosystem reads the result.

The seam exists so that the store, pass and pipeline logic can be exercised
without zarr installed, and so that tests run in-process at small sizes. It is
deliberately thin: a backend creates and opens arrays, and the array it returns
only has to support numpy-style slicing, ``.shape``, ``.dtype`` and a
JSON-serialisable ``.attrs`` mapping. All block arithmetic lives in
:mod:`chunkreg.store`, above this line, so both backends exercise the same code.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Iterator, MutableMapping, Protocol, Sequence, runtime_checkable

import numpy as np

__all__ = [
    "ArrayLike",
    "Backend",
    "ZarrBackend",
    "MemoryBackend",
    "get_backend",
    "set_default_backend",
    "default_backend",
    "shard_layout",
]


def shard_layout(
    shape: Sequence[int], inner: int, core: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Inner chunks and shards for a spatial array, as zarr will accept them.

    A shard holds a whole number of inner chunks, so the shard edge is the core
    clipped to the array and then rounded up to a multiple of the inner chunk.
    """
    chunks = tuple(max(1, min(int(inner), int(n))) for n in shape)
    shards = tuple(
        -(-min(int(core), int(n)) // c) * c for n, c in zip(shape, chunks)
    )
    return chunks, shards


def _check_layout(shape, chunks, shards) -> None:
    """Refuse a layout zarr would refuse, whichever backend is in use.

    The memory backend enforcing the same rule is what lets the fast test
    suite catch a bad layout before a cluster run does.
    """
    if len(chunks) != len(shape):
        raise ValueError(f"chunks {tuple(chunks)} do not match shape {tuple(shape)}")
    if shards is None:
        return
    if len(shards) != len(shape):
        raise ValueError(f"shards {tuple(shards)} do not match shape {tuple(shape)}")
    for d, (c, sh) in enumerate(zip(chunks, shards)):
        if int(sh) % int(c):
            raise ValueError(
                f"Chunk edge length {sh} in dimension {d} is not divisible by "
                f"the shard's inner chunk size {c}."
            )


@runtime_checkable
class ArrayLike(Protocol):
    """The slice of the zarr array interface that :mod:`chunkreg.store` uses."""

    shape: tuple[int, ...]
    dtype: Any
    attrs: MutableMapping[str, Any]

    def __getitem__(self, key) -> np.ndarray: ...
    def __setitem__(self, key, value) -> None: ...


class Backend(Protocol):
    name: str

    def create(
        self,
        path: str | os.PathLike,
        shape: Sequence[int],
        dtype: Any,
        chunks: Sequence[int],
        shards: Sequence[int] | None = None,
        *,
        overwrite: bool = False,
        compress: bool = True,
    ) -> ArrayLike: ...

    def open(self, path: str | os.PathLike, mode: str = "r") -> ArrayLike: ...

    def exists(self, path: str | os.PathLike) -> bool: ...

    def remove(self, path: str | os.PathLike) -> None: ...


# --------------------------------------------------------------------------- #
# Zarr v3: production
# --------------------------------------------------------------------------- #
class ZarrBackend:
    """Zarr v3 with the sharding codec and blosc-zstd compression.

    ``chunks`` is the inner chunk, the unit of decompression; ``shards`` is the
    unit of storage, one file. The store chooses shards equal to a chunk core
    so that an owner-computes pass writes exactly one file per task and no two
    tasks ever touch the same object.
    """

    name = "zarr"

    def __init__(self) -> None:
        try:
            import zarr  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "The zarr backend needs zarr>=3 and numcodecs: "
                "pip install 'chunkreg[store]'"
            ) from exc

    def create(
        self,
        path,
        shape,
        dtype,
        chunks,
        shards=None,
        *,
        overwrite: bool = False,
        compress: bool = True,
    ):
        import zarr

        _check_layout(shape, chunks, shards)
        kwargs: dict[str, Any] = dict(
            shape=tuple(int(s) for s in shape),
            dtype=np.dtype(dtype),
            chunks=tuple(int(c) for c in chunks),
            overwrite=overwrite,
        )
        if shards is not None:
            kwargs["shards"] = tuple(int(s) for s in shards)
        if compress:
            try:
                from zarr.codecs import BloscCodec

                kwargs["compressors"] = [
                    BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
                ]
            except Exception:  # pragma: no cover - older/newer zarr layouts
                pass
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return zarr.create_array(store=str(path), **kwargs)

    def open(self, path, mode: str = "r"):
        import zarr

        return zarr.open_array(store=str(path), mode=mode)

    def exists(self, path) -> bool:
        return Path(path).exists()

    def remove(self, path) -> None:
        import shutil

        shutil.rmtree(path, ignore_errors=True)
        # Field and TaskArray keep their metadata in a sibling file rather than
        # inside the array directory, so rmtree does not reach it. Leaving it
        # behind leaves a completion marker with no data under it, which reads
        # back as "this task is done" for work that has just been deleted.
        sidecar = Path(str(path) + _SIDECAR)
        try:
            sidecar.unlink()
        except FileNotFoundError:
            pass
        except OSError:  # pragma: no cover - permissions, races
            pass


# --------------------------------------------------------------------------- #
# Memory: tests and small local runs
# --------------------------------------------------------------------------- #
class _MemoryArray:
    """A numpy array plus a JSON-able attrs mapping, addressed by path."""

    __slots__ = ("_a", "attrs", "_path", "_chunks", "_shards")

    def __init__(self, a: np.ndarray, path: str, chunks, shards) -> None:
        self._a = a
        self.attrs: dict[str, Any] = {}
        self._path = path
        self._chunks = tuple(chunks)
        self._shards = tuple(shards) if shards is not None else None

    @property
    def shape(self) -> tuple[int, ...]:
        return self._a.shape

    @property
    def dtype(self):
        return self._a.dtype

    @property
    def chunks(self) -> tuple[int, ...]:
        return self._chunks

    @property
    def shards(self):
        return self._shards

    def __getitem__(self, key) -> np.ndarray:
        return self._a[key]

    def __setitem__(self, key, value) -> None:
        self._a[key] = value

    def __array__(self, dtype=None, copy=None):
        return self._a if dtype is None else self._a.astype(dtype)

    def __repr__(self) -> str:
        return f"<MemoryArray {self._path} {self._a.shape} {self._a.dtype}>"


class MemoryBackend:
    """In-process arrays keyed by path. Not persistent; for tests only.

    Records the chunk and shard shapes it was asked for, so tests can assert
    the store's sharding decisions without zarr present.
    """

    name = "memory"

    def __init__(self) -> None:
        self._arrays: dict[str, _MemoryArray] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(path) -> str:
        return str(Path(path).as_posix()).rstrip("/")

    def create(
        self,
        path,
        shape,
        dtype,
        chunks,
        shards=None,
        *,
        overwrite: bool = False,
        compress: bool = True,
    ):
        _check_layout(shape, chunks, shards)
        key = self._key(path)
        with self._lock:
            if key in self._arrays and not overwrite:
                raise FileExistsError(f"array already exists: {key}")
            arr = _MemoryArray(
                np.zeros(tuple(int(s) for s in shape), dtype=np.dtype(dtype)),
                key,
                chunks,
                shards,
            )
            self._arrays[key] = arr
            return arr

    def open(self, path, mode: str = "r"):
        key = self._key(path)
        try:
            return self._arrays[key]
        except KeyError as exc:
            raise FileNotFoundError(f"no array at {key}") from exc

    def exists(self, path) -> bool:
        return self._key(path) in self._arrays

    def remove(self, path) -> None:
        prefix = self._key(path)
        with self._lock:
            for k in [
                k for k in self._arrays if k == prefix or k.startswith(prefix + "/")
            ]:
                del self._arrays[k]
        _purge_json(prefix)

    def clear(self) -> None:
        with self._lock:
            self._arrays.clear()

    def paths(self) -> list[str]:
        return sorted(self._arrays)


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
_BACKENDS: dict[str, Backend] = {}
_DEFAULT = "zarr"


def get_backend(name: str | Backend | None = None) -> Backend:
    """Return a backend by name, constructing it once per process."""
    if name is None:
        name = _DEFAULT
    if not isinstance(name, str):
        return name
    if name not in _BACKENDS:
        if name == "zarr":
            _BACKENDS[name] = ZarrBackend()
        elif name == "memory":
            _BACKENDS[name] = MemoryBackend()
        else:
            raise ValueError(f"unknown backend {name!r}; use 'zarr' or 'memory'")
    return _BACKENDS[name]


def set_default_backend(name: str) -> None:
    """Set the backend used when none is given. Tests use ``'memory'``."""
    global _DEFAULT
    if name not in ("zarr", "memory"):
        raise ValueError(f"unknown backend {name!r}")
    _DEFAULT = name


def default_backend() -> str:
    return _DEFAULT


# --------------------------------------------------------------------------- #
# Small JSON sidecars (group-level metadata)
# --------------------------------------------------------------------------- #
_MEM_JSON: dict[str, dict] = {}

_SIDECAR = ".meta.json"
"""Suffix of the metadata document a Field or TaskArray keeps beside itself."""


def _purge_json(prefix: str) -> None:
    """Drop in-memory metadata for an array that has just been removed.

    Covers both conventions: the sibling document a Field or TaskArray writes
    at ``<path>.meta.json``, and the one a Volume writes inside its own
    directory.
    """
    for key in [
        k
        for k in _MEM_JSON
        if k == prefix + _SIDECAR or k == prefix or k.startswith(prefix + "/")
    ]:
        del _MEM_JSON[key]


def write_json(path, obj: dict, backend: Backend | None = None) -> None:
    """Persist group metadata beside the arrays it describes."""
    b = get_backend(backend)
    if b.name == "memory":
        _MEM_JSON[MemoryBackend._key(path)] = json.loads(json.dumps(obj))
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def read_json(path, backend: Backend | None = None) -> dict:
    b = get_backend(backend)
    if b.name == "memory":
        try:
            return json.loads(json.dumps(_MEM_JSON[MemoryBackend._key(path)]))
        except KeyError as exc:
            raise FileNotFoundError(f"no metadata at {path}") from exc
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_exists(path, backend: Backend | None = None) -> bool:
    b = get_backend(backend)
    if b.name == "memory":
        return MemoryBackend._key(path) in _MEM_JSON
    return Path(path).exists()

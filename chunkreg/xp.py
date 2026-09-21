"""Where array work runs: NumPy on the host, or PyTorch on a device.

The pipeline has one implementation of every pass. Data enters that
implementation through :func:`put` right after it is read from a store and
leaves through :func:`get` right before it is written, and every operation in
between dispatches on the type of array it is handed. With the device set to a
GPU, a pass therefore does all of its arithmetic there: the host only reads and
writes files.

The device is a process-wide setting, chosen once per process from the run
configuration by :func:`configure`. ``"cpu"`` keeps the NumPy reference path.
``"cuda"`` or ``"cuda:N"`` runs on that GPU and fails loudly if there is none,
so a GPU job can never fall back to the CPU without anyone noticing.
``"torch-cpu"`` runs the PyTorch path on the CPU; it exists so the GPU code can
be tested on a machine without a GPU. ``"auto"`` picks ``"cuda"`` when a GPU is
visible and ``"cpu"`` otherwise.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np

__all__ = [
    "DEVICES",
    "configure",
    "tune_cuda",
    "device_name",
    "uses_torch",
    "on_gpu",
    "torch_device",
    "resolve",
    "is_tensor",
    "put",
    "get",
    "zeros",
    "zeros_like",
    "to_float32",
    "clip",
    "release",
    "using",
]

DEVICES = ("auto", "cpu", "cuda", "torch-cpu")
"""Accepted device names. ``cuda:N`` is accepted as well."""

_device = "cpu"


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def resolve(device: str | None, honour_env: bool = True) -> str:
    """Turn a configured device name into the one this process will use.

    The ``CHUNKREG_DEVICE`` environment variable, when set, wins over the
    configured name. It lets a test suite or a debugging session pin a device
    without editing the run's configuration.
    """
    override = os.environ.get("CHUNKREG_DEVICE", "").strip() if honour_env else ""
    if override:
        device = override
    d = "auto" if device is None else str(device).strip().lower()
    if d == "auto":
        return "cuda" if _cuda_available() else "cpu"
    if d in ("cpu", "torch-cpu"):
        if d == "torch-cpu":
            try:
                import torch  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("device 'torch-cpu' needs PyTorch installed") from exc
        return d
    if d == "cuda" or (d.startswith("cuda:") and d[5:].isdigit()):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                f"device {device!r} needs PyTorch with CUDA; install chunkreg[gpu]"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"device {device!r} was requested but no GPU is visible to this "
                f"process (CUDA_VISIBLE_DEVICES and the driver decide that). "
                f"Refusing to fall back to the CPU. Run this command inside a "
                f"GPU job; a scheduler driver on a login node can be moved to "
                f"the CPU deliberately with CHUNKREG_DEVICE=cpu."
            )
        if ":" in d and int(d[5:]) >= torch.cuda.device_count():
            raise RuntimeError(
                f"device {device!r} does not exist; this process sees "
                f"{torch.cuda.device_count()} GPU(s)"
            )
        return d
    raise ValueError(
        f"unknown device {device!r}; use one of {', '.join(DEVICES)} or 'cuda:N'"
    )


def configure(device: str | None, honour_env: bool = True) -> str:
    """Set this process's device. Returns the resolved name."""
    global _device
    _device = resolve(device, honour_env)
    if _device.startswith("cuda"):
        tune_cuda()
    return _device


_tuned = False


def tune_cuda(force: bool = False) -> None:
    """Turn on the CUDA settings this workload wants, once per process.

    Registration is convolution-bound: the engine's local correlation and its
    Gaussian regularisation are the inner loop, and both run in fp32 through
    cuDNN. Two defaults cost real throughput there and neither is a numerical
    risk for this problem:

    ``TF32``
        On Ampere and later, fp32 convolutions and matmuls can use tensor
        cores at roughly tenfold throughput for ten bits of mantissa. A
        displacement field is accurate to a fraction of a voxel at best, and
        the features it is solved on are normalised to order one, so the lost
        bits are far below anything the result represents. The feature network
        already runs in bfloat16, which is coarser still.

    ``cudnn.benchmark``
        Chunk geometry is fixed by the profile, so a level presents only a
        handful of distinct shapes: the padded interior and whatever the
        volume's faces, edges and corners trim it to. Autotuning each one once
        and reusing the plan for every remaining chunk of the level is exactly
        the case this flag is for.

    Both are off with ``CHUNKREG_TF32=0`` and ``CHUNKREG_CUDNN_BENCHMARK=0``,
    for bisecting a numerical difference against an earlier run.
    """
    global _tuned
    if _tuned and not force:
        return
    try:
        import torch
    except ImportError:  # pragma: no cover - resolve() already refused
        return
    if _env_flag("CHUNKREG_TF32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if _env_flag("CHUNKREG_CUDNN_BENCHMARK", True):
        torch.backends.cudnn.benchmark = True
    _tuned = True


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")


def device_name() -> str:
    return _device


def uses_torch() -> bool:
    return _device != "cpu"


def on_gpu() -> bool:
    return _device.startswith("cuda")


def torch_device():
    import torch

    return torch.device("cpu" if _device in ("cpu", "torch-cpu") else _device)


@contextmanager
def using(device: str) -> Iterator[str]:
    """Run a block on another device, restoring the previous one after."""
    global _device
    before = _device
    try:
        yield configure(device, honour_env=False)
    finally:
        _device = before


def is_tensor(x: Any) -> bool:
    return type(x).__module__.split(".", 1)[0] == "torch"


def put(a: Any, dtype=np.float32):
    """Move host data to where this process computes.

    Integer data is converted to ``dtype`` on the way, because every pass
    computes in floating point and PyTorch has no arithmetic for most unsigned
    types. Pass ``dtype=None`` to keep the host dtype on the NumPy path.
    """
    if is_tensor(a):
        import torch

        t = a.to(torch_device())
        if dtype is not None:
            t = t.to(_torch_dtype(dtype))
        return t
    if not uses_torch():
        return np.asarray(a) if dtype is None else np.asarray(a, dtype=dtype)
    import torch

    arr = np.ascontiguousarray(np.asarray(a))
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    elif arr.dtype.kind == "u" and arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
    return torch.from_numpy(arr).to(torch_device())


def get(x: Any) -> np.ndarray:
    """Bring data back to the host for writing."""
    if is_tensor(x):
        return x.detach().to("cpu").numpy()
    return np.asarray(x)


def _torch_dtype(dtype):
    import torch

    return {
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float64): torch.float64,
        np.dtype(np.int64): torch.int64,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.uint8): torch.uint8,
        np.dtype(np.bool_): torch.bool,
    }[np.dtype(dtype)]


def zeros(shape, dtype=np.float32):
    if not uses_torch():
        return np.zeros(tuple(int(n) for n in shape), dtype=dtype)
    import torch

    return torch.zeros(
        tuple(int(n) for n in shape), dtype=_torch_dtype(dtype), device=torch_device()
    )


def zeros_like(x):
    if is_tensor(x):
        import torch

        return torch.zeros_like(x)
    return np.zeros_like(x)


def to_float32(x):
    if is_tensor(x):
        import torch

        return x.to(torch.float32)
    return np.asarray(x, dtype=np.float32)


def clip(x, lo=None, hi=None):
    if is_tensor(x):
        return x.clamp(min=lo, max=hi)
    return np.clip(x, lo, hi)


def release() -> None:
    """Hand cached device memory back, so worker processes on this GPU can use it."""
    if on_gpu():
        import torch

        torch.cuda.empty_cache()

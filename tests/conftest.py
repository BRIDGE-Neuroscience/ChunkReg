"""Test fixtures.

The whole suite runs on the in-process memory backend, so it needs neither
zarr nor a filesystem. Store logic (block arithmetic, lattice conversion,
sharding decisions, idempotency markers) is backend-independent by
construction, so exercising it here exercises the zarr path too.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from chunkreg import backend as _backend
from chunkreg.grid import GridSpec, Profile


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    """Give every test its own working directory.

    Manifests are deliberately real files rather than backend objects: a
    scheduler script references them by path and a human reads them. That means
    the memory backend does not isolate them, so a config with ``root: "."``
    would write into the repository and leak between tests. Isolating the
    working directory makes that impossible rather than merely discouraged.
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def cpu_device(monkeypatch):
    """Run every test on the NumPy reference path unless it asks otherwise.

    Configs default to device 'auto', which would pick a GPU on a machine that
    has one and then refuse the CPU demons engine the tests use. A test that
    exercises the PyTorch path switches with ``xp.using`` or by setting
    ``CHUNKREG_DEVICE`` itself.
    """
    from chunkreg import xp

    monkeypatch.setenv("CHUNKREG_DEVICE", "cpu")
    xp.configure("cpu")
    yield
    xp.configure("cpu", honour_env=False)


@pytest.fixture(autouse=True)
def memory_backend():
    """Use the memory backend and start every test with an empty store."""
    _backend.set_default_backend("memory")
    b = _backend.get_backend("memory")
    b.clear()
    _backend._MEM_JSON.clear()
    yield b
    b.clear()
    _backend._MEM_JSON.clear()


@pytest.fixture
def small_profile() -> Profile:
    """A profile scaled down so a whole pyramid fits in a test.

    The halo is *not* scaled by the same ratio as the core. Kernel and
    smoothing support is absolute, not proportional, so shrinking a 256/48
    profile to 32/6 would leave a clamp of one voxel and nothing could move.
    Halo 12 against kernel 3 and sigma 0.5 leaves seven voxels of clamp, which
    is the same "most of the halo is usable" regime the production profile is
    in.
    """
    return Profile(
        core=32,
        halo=12,
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


def blob(shape=(64, 56, 48), seed=0, smooth=3.0) -> np.ndarray:
    """Band-limited noise with enough structure to register."""
    rng = np.random.default_rng(seed)
    v = rng.random(shape).astype(np.float32)
    v = ndimage.gaussian_filter(v, smooth)
    v -= v.min()
    return (v / max(v.max(), 1e-8)).astype(np.float32)


@pytest.fixture
def volume_data() -> np.ndarray:
    return (blob() * 4000.0).astype(np.uint16)

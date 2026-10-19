"""The blend smooths across core boundaries, not within each core alone."""
from __future__ import annotations

import numpy as np
import pytest

from chunkreg import backend as _backend
from chunkreg.config import load_config
from chunkreg.grid import GridSpec
from chunkreg.passes.blend import blend_box, blend_core, blend_depends, blend_margin
from chunkreg.passes.manifest import build_manifest
from chunkreg.store import TaskArray


@pytest.fixture
def mem():
    _backend.set_default_backend("memory")
    _backend.get_backend("memory").clear()
    yield
    _backend.set_default_backend("memory")


def _cfg(tmp_path):
    return load_config({
        "root": str(tmp_path), "profile": "i1", "backend": "memory",
        "profile_overrides": {"core": 16, "halo": 16, "inner_chunk": 8, "k": 3,
                              "sigma_g": 0.5, "sigma_w": 0.5},
        "subjects": [{"id": "a", "path": "a.zarr"}, {"id": "b", "path": "b.zarr"}],
        "template_subject": "a",
    })


def test_margin_and_box():
    class C:  # a chunk with a core at the volume corner and one inside
        def __init__(self, o, s):
            self.core_origin, self.core_shape = o, s
    grid = GridSpec((48, 48, 48), 1.0)
    assert blend_box(grid, C((0, 0, 0), (16, 16, 16)), 4) == ((0, 0, 0), (20, 20, 20))
    assert blend_box(grid, C((16, 16, 16), (16, 16, 16)), 4) == ((12, 12, 12), (24, 24, 24))
    assert blend_box(grid, C((32, 32, 32), (16, 16, 16)), 4) == ((28, 28, 28), (20, 20, 20))


def test_a_linear_field_blends_to_itself_across_core_boundaries(tmp_path, mem):
    """Every chunk reports the same linear residual; the blend must return it
    unchanged, boundaries included. Smoothing each core with its own edge
    replicated bent the ramp at every core boundary."""
    cfg = _cfg(tmp_path)
    grid = GridSpec((48, 48, 48), 1.0)
    manifest = build_manifest(cfg, 0, 0, grid, coarsest=0)
    assert blend_margin(cfg, 0) == 4
    f = cfg.profile.lattice_factor
    # the residual every chunk writes: u = (0.1 z, -0.05 y, 0.02 x) mm, sampled on its padded lattice
    for task in manifest.tasks:
        arr = TaskArray.create(cfg.task_path(0, 0, task.task_id), task.entries, cfg.profile,
                               backend="memory", level=0, iteration=0)
        for i, e in enumerate(task.entries):
            lo, ext = e.lattice(f)
            zz, yy, xx = np.meshgrid(*[(np.arange(n) + o + 0.5) * f - 0.5 for o, n in zip(lo, ext)],
                                     indexing="ij")
            arr.write(i, np.stack([0.1 * zz, -0.05 * yy, 0.02 * xx]).astype(np.float32))
        arr.mark_complete()
    deps = blend_depends(cfg, manifest)
    assert all(deps[c.id] for c in manifest.chunks)
    for chunk in manifest.chunks:
        u = np.asarray(blend_core(cfg, manifest, "b", chunk.id))
        assert u.shape == (3,) + tuple(chunk.core_shape)
        zz, yy, xx = np.meshgrid(*[np.arange(n) + o for o, n in zip(chunk.core_origin, chunk.core_shape)],
                                 indexing="ij")
        # lattice sampling makes the field piecewise constant at factor 2, so
        # compare the smoothed blend with the ramp to within a lattice step
        np.testing.assert_allclose(u[0], 0.1 * zz, atol=0.1 * f)
        np.testing.assert_allclose(u[1], -0.05 * yy, atol=0.05 * f)
        np.testing.assert_allclose(u[2], 0.02 * xx, atol=0.02 * f)
        # and the ramp's slope survives up to the core edge: no kink. The
        # lattice makes the field a staircase of period 2, so the slope is read
        # over two voxels; smoothing a core with its own edge replicated bent
        # this to about 0.07 at both ends.
        # (only where the core's z-edges are chunk seams, not the volume's
        # own boundary, where replicating the edge is the right thing)
        z0, z1 = chunk.core_origin[0], chunk.core_origin[0] + chunk.core_shape[0]
        if z0 > 0:
            assert abs(float((u[0][2] - u[0][0]).mean()) / 2 - 0.1) < 0.015
        if z1 < grid.shape[0]:
            assert abs(float((u[0][-1] - u[0][-3]).mean()) / 2 - 0.1) < 0.015

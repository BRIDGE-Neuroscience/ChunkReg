"""Store behaviour: block arithmetic, pyramids, lattices, idempotency."""

from __future__ import annotations

import numpy as np
import pytest

from chunkreg import backend as _backend
from chunkreg import fields
from chunkreg.grid import GridSpec, Profile, tile
from chunkreg.store import (
    Field,
    TaskArray,
    TaskEntry,
    Volume,
    ingest_array,
    lattice_box,
    read_padded,
    write_block,
)

from .conftest import blob


# --------------------------------------------------------------------------- #
# Padded block reads
# --------------------------------------------------------------------------- #
def test_read_padded_inside_is_a_plain_slice():
    a = np.arange(6 * 6 * 6, dtype=np.float32).reshape(6, 6, 6)
    out = read_padded(a, (1, 1, 1), (2, 2, 2))
    assert np.array_equal(out, a[1:3, 1:3, 1:3])


def test_read_padded_fills_zero_past_the_edge():
    a = np.ones((4, 4, 4), np.float32)
    out = read_padded(a, (-2, 0, 0), (4, 4, 4))
    assert out.shape == (4, 4, 4)
    assert np.all(out[:2] == 0.0)
    assert np.all(out[2:] == 1.0)


def test_read_padded_can_replicate_the_edge_for_fields():
    a = np.zeros((1, 4, 4, 4), np.float32)
    a[0, 0] = 7.0
    out = read_padded(a, (-2, 0, 0), (4, 4, 4), lead=1, mode="edge")
    assert np.all(out[0, :3] == 7.0)


def test_read_padded_entirely_outside_returns_fill():
    a = np.ones((4, 4, 4), np.float32)
    assert np.all(read_padded(a, (100, 0, 0), (4, 4, 4)) == 0.0)


def test_write_block_clips_at_the_boundary():
    a = np.zeros((4, 4, 4), np.float32)
    write_block(a, (3, 3, 3), np.ones((2, 2, 2), np.float32))
    assert a[3, 3, 3] == 1.0
    assert a.sum() == 1.0


def test_lattice_box_is_exact_for_aligned_chunks():
    """Core and halo are multiples of the factor, so no rounding creeps in."""
    p = Profile()
    for c in tile(GridSpec((1024, 1024, 1024), 0.05), p)[:20]:
        lo, ext = lattice_box(c.pad_origin, c.pad_shape, p.lattice_factor)
        assert all(o * p.lattice_factor == po for o, po in zip(lo, c.pad_origin))
        assert all(e * p.lattice_factor == ps for e, ps in zip(ext, c.pad_shape))


# --------------------------------------------------------------------------- #
# Volume
# --------------------------------------------------------------------------- #
def test_volume_round_trips_native_data(volume_data, small_profile):
    vol = ingest_array(volume_data, "v.zarr", 0.05, small_profile)
    back = vol.read_padded(vol.n_levels - 1, (0, 0, 0), volume_data.shape)
    assert np.array_equal(back.astype(np.uint16), volume_data)


def test_volume_levels_match_the_profile_pyramid(volume_data, small_profile):
    from chunkreg.grid import pyramid

    vol = ingest_array(volume_data, "v.zarr", 0.05, small_profile)
    expected = pyramid(GridSpec(volume_data.shape, 0.05), small_profile)
    assert vol.n_levels == len(expected)
    assert vol.grids() == expected
    assert len(tile(vol.grid(0), small_profile)) == 1


def test_pyramid_preserves_the_mean(volume_data, small_profile):
    """Mean pooling is mean-preserving, so a level-0 read is the volume mean."""
    vol = ingest_array(volume_data, "v.zarr", 0.05, small_profile)
    coarse = np.asarray(vol.array(0)[:], np.float32)
    assert coarse.mean() == pytest.approx(volume_data.mean(), rel=0.02)


def test_pyramid_of_a_constant_volume_is_constant(small_profile):
    data = np.full((64, 64, 64), 1234, np.uint16)
    vol = ingest_array(data, "v.zarr", 0.05, small_profile)
    for k in range(vol.n_levels):
        assert np.allclose(np.asarray(vol.array(k)[:], np.float32), 1234.0)


def test_normalisation_is_stored_and_reapplied(volume_data, small_profile):
    vol = ingest_array(volume_data, "v.zarr", 0.05, small_profile)
    lo, hi = vol.normalisation
    assert hi > lo
    block = vol.read_padded(vol.n_levels - 1, (0, 0, 0), (8, 8, 8), normalise=True)
    assert block.min() >= 0.0 and block.max() <= 1.0
    reopened = Volume.open("v.zarr")
    assert reopened.normalisation == (lo, hi)


def test_reading_a_chunk_without_normalisation_set_is_an_error(small_profile):
    vol = Volume.create("v.zarr", GridSpec((64, 64, 64), 0.05), small_profile)
    with pytest.raises(RuntimeError, match="normalisation"):
        vol.read_padded(0, (0, 0, 0), (4, 4, 4), normalise=True)


def test_volume_rejects_an_out_of_range_level(volume_data, small_profile):
    vol = ingest_array(volume_data, "v.zarr", 0.05, small_profile)
    with pytest.raises(IndexError):
        vol.grid(vol.n_levels)


def test_chunk_read_covers_the_padded_box(volume_data, small_profile):
    vol = ingest_array(volume_data, "v.zarr", 0.05, small_profile)
    k = vol.n_levels - 1
    for c in tile(vol.grid(k), small_profile):
        assert vol.read_chunk(k, c).shape == c.pad_shape


# --------------------------------------------------------------------------- #
# Field
# --------------------------------------------------------------------------- #
def test_field_starts_at_identity(small_profile):
    level = GridSpec((64, 64, 64), 0.05)
    f = Field.create("f.zarr", level, small_profile)
    assert np.allclose(f.read_dense((0, 0, 0), (16, 16, 16)), 0.0)


def test_field_stores_on_a_coarser_lattice(small_profile):
    level = GridSpec((64, 64, 64), 0.05)
    f = Field.create("f.zarr", level, small_profile)
    assert f.array.shape == (3, 32, 32, 32)
    assert f.lattice_grid.spacing_mm == pytest.approx(0.10)


def test_field_round_trips_a_constant_displacement(small_profile):
    level = GridSpec((64, 64, 64), 0.05)
    f = Field.create("f.zarr", level, small_profile)
    u = np.zeros((3, 32, 32, 32), np.float32)
    u[0], u[1], u[2] = 0.30, -0.10, 0.05
    f.write_dense((0, 0, 0), u)
    back = f.read_dense((4, 4, 4), (16, 16, 16))
    assert np.allclose(back[0], 0.30, atol=2e-3)
    assert np.allclose(back[1], -0.10, atol=2e-3)
    assert np.allclose(back[2], 0.05, atol=2e-3)


def test_field_dense_read_recovers_a_smooth_field(small_profile):
    from scipy import ndimage

    level = GridSpec((64, 64, 64), 0.05)
    f = Field.create("f.zarr", level, small_profile)
    rng = np.random.default_rng(3)
    raw = rng.normal(0, 0.05, (3,) + level.shape).astype(np.float32)
    u = np.stack([ndimage.gaussian_filter(raw[d], 6.0) for d in range(3)])
    f.write_dense((0, 0, 0), u)
    back = f.read_dense((0, 0, 0), level.shape)
    inner = (slice(None),) + tuple(slice(8, -8) for _ in range(3))
    assert np.abs(back[inner] - u[inner]).max() < 5e-3


def test_field_write_dense_rejects_a_misaligned_origin(small_profile):
    f = Field.create("f.zarr", GridSpec((64, 64, 64), 0.05), small_profile)
    with pytest.raises(ValueError, match="aligned"):
        f.write_dense((1, 0, 0), np.zeros((3, 4, 4, 4), np.float32))


def test_field_reopens_with_its_geometry(small_profile):
    level = GridSpec((64, 64, 64), 0.05, origin_mm=(1.0, 2.0, 3.0))
    Field.create("f.zarr", level, small_profile, subject="s01")
    f = Field.open("f.zarr")
    assert f.level_grid == level
    assert f.meta["subject"] == "s01"


def test_writing_one_chunk_core_does_not_disturb_its_neighbour(small_profile):
    """Owner-computes: a task writing its own core cannot corrupt another's."""
    level = GridSpec((96, 64, 64), 0.05)
    f = Field.create("f.zarr", level, small_profile)
    chunks = tile(level, small_profile)
    a, b = chunks[0], chunks[1]
    ua = np.full((3,) + a.core_shape, 0.2, np.float32)
    ub = np.full((3,) + b.core_shape, -0.4, np.float32)
    f.write_dense(a.core_origin, ua)
    f.write_dense(b.core_origin, ub)
    back_a = f.read_dense(a.core_origin, a.core_shape)
    back_b = f.read_dense(b.core_origin, b.core_shape)
    mid_a = tuple(slice(4, -4) for _ in range(3))
    assert np.allclose(back_a[(slice(None),) + mid_a], 0.2, atol=1e-2)
    assert np.allclose(back_b[(slice(None),) + mid_a], -0.4, atol=1e-2)


# --------------------------------------------------------------------------- #
# TaskArray
# --------------------------------------------------------------------------- #
def make_entries(level: GridSpec, profile: Profile, subject="s01", n=3):
    return [
        TaskEntry(
            subject=subject,
            chunk_id=c.id,
            chunk_index=c.index,
            pad_origin=c.pad_origin,
            pad_shape=c.pad_shape,
        )
        for c in tile(level, profile)[:n]
    ]


def test_task_array_round_trips_each_entry(small_profile):
    level = GridSpec((96, 96, 96), 0.05)
    entries = make_entries(level, small_profile)
    t = TaskArray.create("t.zarr", entries, small_profile, level=1, iteration=0)
    expected = []
    for i, e in enumerate(entries):
        _, ext = e.lattice(small_profile.lattice_factor)
        u = np.full((3,) + ext, float(i + 1) * 0.1, np.float32)
        expected.append(u)
        t.write(i, u)
    for i, u in enumerate(expected):
        assert np.allclose(t.read(i), u, atol=1e-3)


def test_task_array_is_one_array_sharded_per_entry(small_profile):
    """One array per task, but one shard per entry, so blend reads are exact.

    A blend task needs up to 27 scattered entries; sharding per task would make
    it decompress a whole task array to reach each one.
    """
    level = GridSpec((96, 96, 96), 0.05)
    entries = make_entries(level, small_profile, n=8)
    t = TaskArray.create("t.zarr", entries, small_profile)
    shapes = {e.lattice(small_profile.lattice_factor)[1] for e in entries}
    assert len(shapes) > 1, "test needs boundary chunks to be a real case"
    assert t.array.shape[0] == len(entries)
    assert t.array.shards[0] == 1
    assert t.array.shards[1:] == t.array.shape[1:]


def test_task_array_rejects_a_wrong_extent(small_profile):
    level = GridSpec((96, 96, 96), 0.05)
    t = TaskArray.create("t.zarr", make_entries(level, small_profile), small_profile)
    with pytest.raises(ValueError, match="expects"):
        t.write(0, np.zeros((3, 2, 2, 2), np.float32))


def test_completion_marker_gates_a_rerun(small_profile):
    level = GridSpec((96, 96, 96), 0.05)
    entries = make_entries(level, small_profile)
    assert not TaskArray.is_complete("t.zarr")
    t = TaskArray.create("t.zarr", entries, small_profile)
    assert not TaskArray.is_complete("t.zarr")
    t.mark_complete()
    assert TaskArray.is_complete("t.zarr")
    assert TaskArray.open("t.zarr").complete


def test_task_entries_survive_a_json_round_trip(small_profile):
    level = GridSpec((96, 96, 96), 0.05)
    entries = make_entries(level, small_profile)
    TaskArray.create("t.zarr", entries, small_profile)
    assert TaskArray.open("t.zarr").entries == entries


# --------------------------------------------------------------------------- #
# Backend seam
# --------------------------------------------------------------------------- #
def test_memory_backend_records_the_sharding_request(volume_data, small_profile):
    """Shards are chosen equal to chunk cores, which is what owner-computes needs."""
    vol = ingest_array(volume_data, "v.zarr", 0.05, small_profile)
    native = vol.array(vol.n_levels - 1)
    assert native.chunks == (8, 8, 8)
    assert all(s % c == 0 for s, c in zip(native.shards, native.chunks))
    assert max(native.shards) <= small_profile.core


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        _backend.get_backend("postgres")

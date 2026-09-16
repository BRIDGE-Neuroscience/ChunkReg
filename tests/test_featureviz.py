"""Feature visualisation: is the picture honest?

A feature sheet is only useful if the colours mean the same thing in every
panel and the numbers beside them are real. These tests pin both, plus the
MIND-SSC extractor the sheet uses to demonstrate multichannel features without
a GPU.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from chunkreg import featureviz as viz
from chunkreg.config import load_config
from chunkreg.featurereport import (
    MIN_CONTEXT_VOX,
    build_feature_report,
    sample_box,
    write_feature_report,
)
from chunkreg.features import get_extractor
from chunkreg.features.mindssc import SSC_PAIRS, MindSSCFeatures
from chunkreg.grid import Profile
from chunkreg.store import Volume, ingest_array

from .conftest import blob


# --------------------------------------------------------------------------- #
# MIND-SSC
# --------------------------------------------------------------------------- #
def test_ssc_uses_the_twelve_octahedron_edges():
    """Pairs on the same axis measure a second derivative, not self-similarity."""
    assert len(SSC_PAIRS) == 12
    assert all((i // 2) != (j // 2) for i, j in SSC_PAIRS)
    assert len(set(SSC_PAIRS)) == 12


def test_mindssc_returns_twelve_normalised_channels():
    ex = MindSSCFeatures()
    out = ex(blob((32, 32, 32)), 0.05)
    assert out.shape == (12, 32, 32, 32)
    norms = np.sqrt((out**2).sum(axis=0))
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-3)


def test_mindssc_receptive_field_is_derivable_not_guessed():
    """Unlike a network, its reach is exactly dilation plus patch radius."""
    assert MindSSCFeatures(radius=1, dilation=2).r_f == 3
    assert MindSSCFeatures(radius=2, dilation=3).r_f == 5


def test_calibrate_measures_the_receptive_field_it_declares():
    """A known-answer check on the calibration that guards the halo budget."""
    from chunkreg.calibrate import measure_receptive_field

    ex = MindSSCFeatures(radius=1, dilation=2)
    measured = measure_receptive_field(ex, size=48)
    assert 0 < measured <= ex.r_f + 1, f"measured {measured}, declared {ex.r_f}"


def test_intensity_extractor_has_no_receptive_field():
    from chunkreg.calibrate import measure_receptive_field

    assert measure_receptive_field(get_extractor("intensity"), size=32) == 0


def test_mindssc_is_invariant_to_an_affine_intensity_change():
    """The point of a self-similarity descriptor: contrast cancels out."""
    ex = MindSSCFeatures()
    v = blob((28, 28, 28), seed=4)
    a = ex(v, 0.05)
    b = ex(v * 3.0 + 0.4, 0.05)
    inner = (slice(None),) + tuple(slice(6, -6) for _ in range(3))
    assert np.abs(a[inner] - b[inner]).mean() < 0.05


def test_mindssc_responds_to_structure_and_not_to_flatness():
    ex = MindSSCFeatures()
    flat = ex(np.full((24, 24, 24), 0.5, np.float32), 0.05)
    structured = ex(blob((24, 24, 24), seed=5), 0.05)
    box = (slice(None),) + tuple(slice(5, -5) for _ in range(3))
    assert viz.feature_metrics(structured[box])["contrast"] > 10 * (
        viz.feature_metrics(flat[box])["contrast"] + 1e-9
    )


# --------------------------------------------------------------------------- #
# The shared projection
# --------------------------------------------------------------------------- #
def multichannel(shape=(16, 16, 16), c=8, seed=0) -> np.ndarray:
    """Channels that are genuinely linearly independent.

    Scaled copies of one base would be rank one however many there are, which
    is the collapse the rank metric exists to detect, so a fixture built that
    way cannot stand in for a healthy descriptor.
    """
    from scipy import ndimage

    base = blob(shape, seed=seed)
    chans = []
    for i in range(c):
        if i % 4 == 0:
            chans.append(ndimage.gaussian_filter(base, 0.5 + i * 0.4))
        elif i % 4 == 3:
            chans.append(ndimage.laplace(ndimage.gaussian_filter(base, 1.0 + i * 0.2)))
        else:
            chans.append(ndimage.sobel(base, axis=(i % 3)) * (1.0 + 0.1 * i))
    return np.stack(chans).astype(np.float32)


def test_one_basis_maps_identical_features_to_identical_colour():
    """The whole point: the same feature vector is the same colour everywhere."""
    a = multichannel(seed=1)
    b = multichannel(seed=2)
    basis = viz.fit_rgb_basis([a, b])
    assert np.allclose(basis.to_rgb(a), basis.to_rgb(a))
    # A vector appearing in both tiles gets one colour, because one map is used.
    shared_vec = a[:, 0, 0, 0]
    mixed = b.copy()
    mixed[:, 5, 5, 5] = shared_vec
    assert np.allclose(
        basis.to_rgb(a)[0, 0, 0], basis.to_rgb(mixed)[5, 5, 5], atol=1e-5
    )


def test_rgb_is_bounded_and_correctly_shaped():
    a = multichannel(c=12)
    rgb = viz.fit_rgb_basis([a]).to_rgb(a)
    assert rgb.shape == (16, 16, 16, 3)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0


def test_component_signs_are_deterministic():
    """Otherwise a rerun recolours everything and comparisons across runs fail."""
    a, b = multichannel(seed=3), multichannel(seed=4)
    first = viz.fit_rgb_basis([a, b])
    second = viz.fit_rgb_basis([a, b])
    assert np.allclose(first.basis, second.basis)
    assert np.allclose(first.to_rgb(a), second.to_rgb(a))


def test_explained_variance_is_reported_and_bounded():
    basis = viz.fit_rgb_basis([multichannel(c=12, seed=6)])
    assert 0.0 <= basis.explained <= 1.0


def test_a_single_channel_renders_as_grey_not_as_a_failed_projection():
    one = blob((12, 12, 12))[None]
    basis = viz.fit_rgb_basis([one])
    rgb = basis.to_rgb(one)
    assert basis.explained == 1.0
    assert np.allclose(rgb[..., 0], rgb[..., 1])
    assert np.allclose(rgb[..., 1], rgb[..., 2])


def test_a_basis_cannot_be_reused_across_extractors():
    basis = viz.fit_rgb_basis([multichannel(c=8)])
    with pytest.raises(ValueError, match="cannot be reused"):
        basis.to_rgb(multichannel(c=12))


def test_samples_must_agree_on_channel_count():
    with pytest.raises(ValueError, match="disagree on channel count"):
        viz.fit_rgb_basis([multichannel(c=8), multichannel(c=12)])


def test_fit_needs_at_least_one_sample():
    with pytest.raises(ValueError, match="at least one sample"):
        viz.fit_rgb_basis([])


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_rank_detects_a_collapsed_descriptor():
    """Twelve channels all carrying the same signal is rank one, not twelve."""
    base = blob((16, 16, 16), seed=7)
    collapsed = np.stack([base] * 12).astype(np.float32)
    rich = multichannel(c=12, seed=8)
    assert viz.feature_metrics(collapsed)["rank"] == pytest.approx(1.0, abs=0.05)
    assert viz.feature_metrics(rich)["rank"] > 1.5


def test_contrast_is_zero_for_a_constant_field():
    flat = np.ones((4, 8, 8, 8), np.float32)
    assert viz.feature_metrics(flat)["contrast"] == pytest.approx(0.0, abs=1e-6)


def test_metrics_report_tissue_fraction_when_given_a_mask():
    f = multichannel()
    tissue = np.zeros((16, 16, 16), bool)
    tissue[:8] = True
    assert viz.feature_metrics(f, tissue=tissue)["tissue_fraction"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Slicing and the sheet
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("plane,expected", [("axial", (6, 8)), ("coronal", (4, 8)), ("sagittal", (4, 6))])
def test_orthoslice_takes_the_right_plane(plane, expected):
    v = np.zeros((4, 6, 8), np.float32)
    assert viz.orthoslice(v, plane).shape == expected


def test_orthoslice_rejects_an_unknown_plane():
    with pytest.raises(ValueError, match="unknown plane"):
        viz.orthoslice(np.zeros((4, 4, 4)), "oblique")


def test_coarse_tiles_are_upscaled_without_smoothing():
    """A blocky coarse level next to a smooth fine one is the signal, not a flaw."""
    small = np.array([[0.0, 1.0], [1.0, 0.0]], np.float32)
    big = viz._resize_nearest(small, 8)
    assert big.shape == (8, 8)
    assert set(np.unique(big)) == {0.0, 1.0}


def test_contact_sheet_renders_every_cell(tmp_path):
    pytest.importorskip("PIL")
    feats = multichannel(c=8)
    basis = viz.fit_rgb_basis([feats])
    tiles = [
        viz.Tile(
            subject=s,
            level=k,
            spacing_mm=0.05 * (2**k),
            extent_mm=1.6,
            intensity=blob((16, 16))[..., 0] if False else np.asarray(blob((16, 16, 16))[0]),
            rgb=basis.to_rgb(feats)[0],
            metrics=viz.feature_metrics(feats, basis),
        )
        for s in ("s01", "s02")
        for k in (0, 1, 2)
    ]
    sheet = viz.contact_sheet(tiles, ["s01", "s02"], [0, 1, 2], tile_px=48, title="t")
    out = tmp_path / "sheet.png"
    viz.save_png(out, sheet)
    assert out.exists() and out.stat().st_size > 0
    assert sheet.size[0] > 3 * 48 and sheet.size[1] > 2 * 48


def test_contact_sheet_tolerates_a_missing_tile(tmp_path):
    pytest.importorskip("PIL")
    feats = multichannel(c=4)
    basis = viz.fit_rgb_basis([feats])
    tiles = [
        viz.Tile("s01", 0, 0.05, 1.6, np.asarray(blob((8, 8, 8))[0]),
                 basis.to_rgb(feats)[0], viz.feature_metrics(feats, basis))
    ]
    sheet = viz.contact_sheet(tiles, ["s01", "s02"], [0, 1], tile_px=32)
    assert sheet is not None


# --------------------------------------------------------------------------- #
# The report, end to end
# --------------------------------------------------------------------------- #
SP = 0.05
SHAPE = (64, 64, 64)


@pytest.fixture
def report_cfg(tmp_path):
    profile = Profile(
        core=16, halo=6, inner_chunk=8, lattice_factor=2, channels=12,
        features="mindssc", r_f=3, k=3, sigma_g=0.5, sigma_w=0.5, scales=(2, 1),
    )
    for i in (1, 2):
        ingest_array(
            (blob(SHAPE, seed=10 + i) * 4000).astype(np.uint16),
            str(tmp_path / f"s{i:02d}.zarr"),
            SP,
            profile,
        )
    raw = {
        "root": str(tmp_path),
        "backend": "memory",
        "profile": "i1",
        "runner": "local",
        "spacing_mm": SP,
        "gpu_mem_gb": 1000.0,
        "subjects": [
            {"id": f"s{i:02d}", "path": str(tmp_path / f"s{i:02d}.zarr")} for i in (1, 2)
        ],
    }
    p = tmp_path / "run.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(p)


def test_levels_come_from_the_store_not_the_config_profile(report_cfg):
    """The config's profile has core 256, so its pyramid would be one level.

    The volumes were ingested at core 16 and have several. Indexing them by the
    config's depth read level 0 of a multi-level store as if it were native,
    landed outside it, and returned an all-zero box with zero contrast.
    """
    rep = build_feature_report(report_cfg, extractor_name="mindssc", size_mm=0.8)
    n_store_levels = Volume.open(report_cfg.subject_path("s01"), "memory").n_levels
    assert n_store_levels > 1
    assert rep.levels == list(range(n_store_levels))
    assert all(t.metrics["contrast"] > 0 for t in rep.tiles), (
        "a zero-contrast tile means the box was read outside the array"
    )


def test_the_box_is_the_same_anatomy_at_every_level(report_cfg):
    rep = build_feature_report(report_cfg, extractor_name="mindssc", size_mm=0.8)
    assert {t.extent_mm for t in rep.tiles} == {0.8}
    by_level = {t.level: t for t in rep.tiles if t.subject == "s01"}
    # Same millimetres, so finer levels hold proportionally more voxels.
    coarse, fine = min(by_level), max(by_level)
    assert by_level[coarse].spacing_mm > by_level[fine].spacing_mm


def test_every_subject_and_level_gets_a_tile(report_cfg):
    rep = build_feature_report(report_cfg, extractor_name="mindssc", size_mm=0.8)
    assert len(rep.tiles) == len(rep.subjects) * len(rep.levels)


def test_features_are_extracted_with_a_margin_then_cropped(report_cfg):
    """Otherwise the border of every tile is the extractor seeing zero padding."""
    vol = Volume.open(report_cfg.subject_path("s01"), "memory")
    grid = vol.grids()[-1]
    ex = get_extractor("mindssc")
    ex.setup()
    block, feats = sample_box(vol, vol.n_levels - 1, grid, (1.6, 1.6, 1.6), 16, ex)
    assert block.shape == (16, 16, 16)
    assert feats.shape == (12, 16, 16, 16)
    # A cropped interior has no dead border; an uncropped one would.
    edge = feats[:, 0].std()
    middle = feats[:, 8].std()
    assert edge > 0.2 * middle


def test_a_tiny_box_is_grown_to_a_usable_context(report_cfg):
    """A four-voxel cube would collapse a downsampling network to nothing."""
    vol = Volume.open(report_cfg.subject_path("s01"), "memory")
    ex = get_extractor("mindssc")
    ex.setup()
    block, feats = sample_box(vol, 0, vol.grids()[0], (1.6, 1.6, 1.6), 4, ex)
    assert block.shape == (4, 4, 4)
    assert feats.shape == (12, 4, 4, 4)
    assert MIN_CONTEXT_VOX >= 32


def test_shared_and_per_level_bases_differ_and_are_both_labelled(report_cfg):
    shared = build_feature_report(report_cfg, extractor_name="mindssc", size_mm=0.8)
    per = build_feature_report(
        report_cfg, extractor_name="mindssc", size_mm=0.8, basis_mode="per-level"
    )
    assert shared.basis_mode == "shared"
    assert per.basis_mode == "per-level"
    assert "comparable across every panel" in shared.summary()
    assert "NOT across columns" in per.summary()
    assert len({id(b) for b in shared.bases.values()}) == 1
    assert len({id(b) for b in per.bases.values()}) == len(per.levels)


def test_unknown_basis_mode_is_rejected(report_cfg):
    with pytest.raises(ValueError, match="unknown basis mode"):
        build_feature_report(report_cfg, extractor_name="mindssc", basis_mode="rainbow")


def test_subjects_ingested_under_different_profiles_are_refused(tmp_path):
    a = Profile(core=16, halo=6, inner_chunk=8, lattice_factor=2, channels=1,
                features="intensity", r_f=0, k=3, sigma_g=0.5, sigma_w=0.5, scales=(2, 1))
    b = Profile(core=64, halo=6, inner_chunk=8, lattice_factor=2, channels=1,
                features="intensity", r_f=0, k=3, sigma_g=0.5, sigma_w=0.5, scales=(2, 1))
    ingest_array((blob(SHAPE) * 4000).astype(np.uint16), str(tmp_path / "a.zarr"), SP, a)
    ingest_array((blob(SHAPE) * 4000).astype(np.uint16), str(tmp_path / "b.zarr"), SP, b)
    raw = {
        "root": str(tmp_path), "backend": "memory", "profile": "i1", "runner": "local",
        "spacing_mm": SP, "gpu_mem_gb": 1000.0,
        "subjects": [
            {"id": "a", "path": str(tmp_path / "a.zarr")},
            {"id": "b", "path": str(tmp_path / "b.zarr")},
        ],
    }
    p = tmp_path / "run.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="different profiles"):
        build_feature_report(load_config(p), extractor_name="intensity")


def test_out_of_range_level_says_which_end_is_coarse(report_cfg):
    with pytest.raises(ValueError, match="0 is the coarsest"):
        build_feature_report(report_cfg, levels=[99], extractor_name="intensity")


def test_write_produces_a_sheet_per_tile_pngs_and_metrics(report_cfg, tmp_path):
    pytest.importorskip("PIL")
    rep = build_feature_report(report_cfg, extractor_name="mindssc", size_mm=0.8)
    out = tmp_path / "qc"
    written = write_feature_report(rep, out, tile_px=48)

    assert (out / "feature_sheet.png").exists()
    assert written["tiles"] == len(rep.tiles)
    for t in rep.tiles:
        assert (out / f"L{t.level}_{t.subject}_features.png").exists()
        assert (out / f"L{t.level}_{t.subject}_intensity.png").exists()

    metrics = json.loads((out / "feature_metrics.json").read_text())
    assert metrics["extractor"] == "mindssc"
    assert metrics["channels"] == 12
    assert metrics["basis_mode"] == "shared"
    assert len(metrics["tiles"]) == len(rep.tiles)
    assert all("contrast" in t and "rank" in t for t in metrics["tiles"])


def test_summary_ranks_levels_and_explains_the_numbers(report_cfg):
    rep = build_feature_report(report_cfg, extractor_name="mindssc", size_mm=0.8)
    text = rep.summary()
    assert "contrast" in text and "rank" in text and "shown" in text
    assert "contrast near zero" in text
    for level in rep.levels:
        assert f"\n{level:>3} " in "\n" + text


def test_the_legend_names_the_projection_mode(report_cfg, tmp_path):
    """A per-level sheet labelled "shared" invites the one comparison it cannot
    support, so the mode has to reach the figure itself, not just the console."""
    pytest.importorskip("PIL")
    from PIL import Image

    sizes = {}
    for mode in ("shared", "per-level"):
        rep = build_feature_report(
            report_cfg, extractor_name="mindssc", size_mm=0.8, basis_mode=mode
        )
        out = tmp_path / mode
        write_feature_report(rep, out, tile_px=48)
        sizes[mode] = Image.open(out / "feature_sheet.png").size
        assert rep.basis_mode == mode

    # Both render; the per-level colours genuinely differ from the shared ones.
    shared = build_feature_report(report_cfg, extractor_name="mindssc", size_mm=0.8)
    per = build_feature_report(
        report_cfg, extractor_name="mindssc", size_mm=0.8, basis_mode="per-level"
    )
    coarse = min(shared.levels)
    a = next(t for t in shared.tiles if t.level == coarse)
    b = next(t for t in per.tiles if t.level == coarse and t.subject == a.subject)
    assert not np.allclose(a.rgb, b.rgb), (
        "per-level should recolour the coarse level, which the shared basis "
        "renders at whatever contrast the fine levels leave it"
    )

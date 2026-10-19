"""Feature extractors.

An extractor turns a normalised intensity chunk into the channels an engine
registers. Its ``r_f`` is not decoration: it is the receptive-field radius that
enters the halo budget, and therefore sets how much displacement a chunk is
allowed to solve for. Getting it wrong under-sizes the halo silently, which is
why ``chunkreg setup`` measures it rather than trusting the default.
"""

from __future__ import annotations

from .base import FeatureExtractor, normalise_channels

__all__ = ["FeatureExtractor", "get_extractor", "describe", "normalise_channels"]

_REGISTRY = {
    "intensity": "chunkreg.features.intensity:IntensityFeatures",
    "anatomix": "chunkreg.features.anatomix:AnatomixFeatures",
    "mindssc": "chunkreg.features.mindssc:MindSSCFeatures",
}


def _one(spec: str, **kwargs) -> FeatureExtractor:
    import importlib

    if spec.startswith("pca:"):
        n = int(spec.split(":", 1)[1])
        from .pca import PCAFeatures

        return PCAFeatures(n_components=n, **kwargs)
    window = anatomix_window(spec)
    if window is not None:
        # ``anatomix:wN``: anatomix run in N-voxel sliding windows rather than
        # over the whole chunk in one pass (see ``features.anatomix_window``).
        from .anatomix import AnatomixFeatures

        return AnatomixFeatures(window=window, **kwargs)
    if spec not in _REGISTRY:
        raise ValueError(
            f"unknown feature spec {spec!r}; use one of "
            f"{', '.join(sorted(_REGISTRY))} or 'pca:N'"
        )
    module, _, attr = _REGISTRY[spec].partition(":")
    return getattr(importlib.import_module(module), attr)(**kwargs)


def get_extractor(spec: str, seed: int = 0, **kwargs) -> FeatureExtractor:
    """Build an extractor from a profile's ``features`` string.

    A spec is one or more extractors joined by ``+``, optionally followed by
    ``@k`` to register only ``k`` of their channels per template pass:

    ``"anatomix"``
        the sixteen anatomix channels;
    ``"anatomix+mindssc"``
        anatomix and the twelve MIND-SSC channels side by side;
    ``"anatomix+mindssc@16"``
        sixteen of those twenty-eight per pass, a new stratified draw each
        pass (see :mod:`chunkreg.features.combined`).

    Single extractors are ``intensity``, ``anatomix``, ``mindssc`` and
    ``pca:N``, where the PCA form wraps anatomix with a projection fitted once
    at the coarsest level. ``kwargs`` go to a single extractor only.
    """
    text = str(spec).strip()
    k = None
    if "@" in text:
        text, _, count = text.partition("@")
        try:
            k = int(count)
        except ValueError:
            raise ValueError(f"feature spec {spec!r}: '@' must be followed by a channel count") from None
    names = [n.strip() for n in text.split("+") if n.strip()]
    if not names:
        raise ValueError(f"empty feature spec {spec!r}")
    if len(names) == 1:
        inner = _one(names[0], **kwargs)
    else:
        if kwargs:
            raise ValueError("extractor options apply to a single extractor, not a combination")
        from .combined import CombinedFeatures

        # MIND-SSC joins a combination as raw responses, the way anatomix's
        # own anatomix+mindssc features are built.
        inner = CombinedFeatures(
            [_one(n, normalisation="none") if n == "mindssc" else _one(n) for n in names]
        )
    if k is None or k == inner.channels:
        return inner
    from .combined import ChannelSample

    return ChannelSample(inner, k, seed=seed)


def anatomix_window(name: str) -> int | None:
    """The window of an ``anatomix:wN`` spec part, or ``None`` for any other part.

    Windowed anatomix sees at most N voxels of context. On a whole brain at a
    coarse level the network otherwise sees the entire volume in one pass and
    its channels are dominated by where a voxel sits in the brain, which
    differs between subjects; limiting the context to a few centimetres makes
    them describe local tissue, which is what a deformable stage can match.
    """
    name = str(name).strip()
    if not name.startswith("anatomix:w"):
        return None
    try:
        return int(name[len("anatomix:w"):])
    except ValueError:
        raise ValueError(f"feature spec {name!r}: the window must be an integer, e.g. anatomix:w64") from None


def describe(spec: str) -> tuple[int, int]:
    """``(channels registered per pass, receptive-field radius)`` of a spec.

    Read from the spec alone, without loading any weights, so a config can be
    checked before a GPU is anywhere near it.
    """
    known_rf = {"intensity": 0, "anatomix": 24, "mindssc": 3}
    known_ch = {"intensity": 1, "anatomix": 16, "mindssc": 12}
    text, _, count = str(spec).strip().partition("@")
    names = [n.strip() for n in text.split("+") if n.strip()]
    if not names:
        raise ValueError(f"empty feature spec {spec!r}")
    channels, r_f = 0, 0
    for n in names:
        if n.startswith("pca:"):
            channels += int(n.split(":", 1)[1])
            r_f = max(r_f, known_rf["anatomix"])
        elif anatomix_window(n) is not None:
            channels += known_ch["anatomix"]
            r_f = max(r_f, known_rf["anatomix"])
        elif n in known_ch:
            channels += known_ch[n]
            r_f = max(r_f, known_rf[n])
        else:
            raise ValueError(f"unknown feature spec {n!r}")
    if count:
        k = int(count)
        if not 1 <= k <= channels:
            raise ValueError(f"feature spec {spec!r} samples {k} of {channels} channels")
        channels = k
    return channels, r_f

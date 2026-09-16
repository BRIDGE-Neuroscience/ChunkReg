"""Feature extractors.

An extractor turns a normalised intensity chunk into the channels an engine
registers. Its ``r_f`` is not decoration: it is the receptive-field radius that
enters the halo budget, and therefore sets how much displacement a chunk is
allowed to solve for. Getting it wrong under-sizes the halo silently, which is
why ``chunkreg setup`` measures it rather than trusting the default.
"""

from __future__ import annotations

from .base import FeatureExtractor, normalise_channels

__all__ = ["FeatureExtractor", "get_extractor", "normalise_channels"]

_REGISTRY = {
    "intensity": "chunkreg.features.intensity:IntensityFeatures",
    "anatomix": "chunkreg.features.anatomix:AnatomixFeatures",
    "mindssc": "chunkreg.features.mindssc:MindSSCFeatures",
}


def get_extractor(spec: str, **kwargs) -> FeatureExtractor:
    """Build an extractor from a profile's ``features`` string.

    Accepts ``"intensity"``, ``"anatomix"``, ``"mindssc"`` and ``"pca:N"``,
    where the PCA form wraps anatomix with a projection fitted once at the
    coarsest level.
    """
    import importlib

    if spec.startswith("pca:"):
        n = int(spec.split(":", 1)[1])
        from .pca import PCAFeatures

        return PCAFeatures(n_components=n, **kwargs)
    if spec not in _REGISTRY:
        raise ValueError(
            f"unknown feature spec {spec!r}; use one of "
            f"{', '.join(sorted(_REGISTRY))} or 'pca:N'"
        )
    module, _, attr = _REGISTRY[spec].partition(":")
    return getattr(importlib.import_module(module), attr)(**kwargs)

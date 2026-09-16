"""Registration engines.

An engine takes two feature stacks on a common chunk grid and returns a
displacement field. Everything above this layer is engine-agnostic, so the
pipeline can run on a CPU reference implementation anywhere and on FireANTs
where there is a GPU.
"""

from __future__ import annotations

from .base import Engine, RegResult, StageNotSupported

__all__ = ["Engine", "RegResult", "StageNotSupported", "get_engine"]

_REGISTRY: dict[str, str] = {
    "demons": "chunkreg.engines.demons:DemonsEngine",
    "fireants": "chunkreg.engines.fireants:FireAntsEngine",
    "fireants_greedy": "chunkreg.engines.fireants:FireAntsEngine",
}


def get_engine(name: str, **kwargs):
    """Construct an engine by name.

    ``demons`` is the CPU reference engine: dependency-free, deterministic and
    used by the test suite. ``fireants`` is the production engine and needs a
    GPU build of FireANTs.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown engine {name!r}; known engines: {', '.join(sorted(_REGISTRY))}"
        )
    module, _, attr = _REGISTRY[name].partition(":")
    import importlib

    return getattr(importlib.import_module(module), attr)(**kwargs)

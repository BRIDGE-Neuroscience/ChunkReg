"""Exact global percentiles from independent tasks.

The stopping rule compares percentiles of displacement magnitude across a whole
level, but no task ever sees a whole level: a register task sees its chunks and
an update task sees its shard. Percentiles do not average, so reporting one per
task and combining them afterwards is not an option, and taking the maximum
across tasks turns a percentile into an outlier detector.

Each task instead returns a fixed-size histogram over a shared logarithmic bin
edge set. Histograms add, so the pipeline sums them and reads the percentile off
the total. The cost is a few kilobytes per task regardless of volume size, and
the answer is exact to the bin width, which is 3% of the value at every scale.
"""

from __future__ import annotations

import numpy as np

__all__ = ["EDGES", "N_BINS", "histogram", "merge", "percentile", "empty"]

# 1 nanometre to 1 metre, ~3% apart. Anything a registration can produce falls
# inside, and resolution is relative so it is equally good at 0.01 mm and 10 mm.
EDGES: np.ndarray = np.concatenate(
    [[0.0], np.logspace(-6.0, 3.0, 255, dtype=np.float64)]
)
N_BINS = len(EDGES)


def empty() -> np.ndarray:
    return np.zeros(N_BINS, dtype=np.int64)


def histogram(values: np.ndarray) -> np.ndarray:
    """Bin displacement magnitudes, in millimetres.

    A device tensor is binned on its device; only the counts come back.
    """
    if type(values).__module__.split(".", 1)[0] == "torch":
        from .gpu_ops import histogram as _device_histogram

        return _device_histogram(values, EDGES)
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return empty()
    idx = np.searchsorted(EDGES, v, side="right") - 1
    np.clip(idx, 0, N_BINS - 1, out=idx)
    return np.bincount(idx, minlength=N_BINS).astype(np.int64)


def merge(hists) -> np.ndarray:
    """Sum histograms from any number of tasks."""
    total = empty()
    for h in hists:
        if h is None:
            continue
        a = np.asarray(h, dtype=np.int64)
        if a.shape != total.shape:
            raise ValueError(
                f"histogram has {a.shape} bins, expected {total.shape}; all "
                f"tasks must share the bin edges in chunkreg.stats"
            )
        total += a
    return total


def percentile(hist, q: float) -> float:
    """Read a percentile off a merged histogram, in millimetres.

    Returns the upper edge of the bin the quantile falls in, so the result is
    an upper bound on the true percentile and a stopping rule built on it never
    declares convergence early.
    """
    h = np.asarray(hist, dtype=np.int64)
    total = int(h.sum())
    if total == 0:
        return 0.0
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be a percentage in [0, 100], got {q}")
    target = q / 100.0 * total
    cum = np.cumsum(h)
    i = int(np.searchsorted(cum, target, side="left"))
    i = min(i, N_BINS - 1)
    return float(EDGES[min(i + 1, N_BINS - 1)])

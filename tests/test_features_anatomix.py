"""When a padded chunk is too big for one forward pass.

Two unrelated-looking errors mean the same thing and have the same remedy:
the device is out of memory, or a CUDA kernel refuses a tensor of more than
2**31 elements because it indexes with 32-bit arithmetic. Only the first was
recognised, so widening the profile's halo -- which is what a measured
receptive field asks for -- turned a chunk that tiles into a run that stops.

No GPU and no anatomix weights here: the classifier and the retry ladder are
the parts that were wrong.
"""

from __future__ import annotations

import pytest

from chunkreg.features.anatomix import AnatomixFeatures, too_big_for_one_pass


@pytest.mark.parametrize(
    "message",
    [
        "CUDA out of memory. Tried to allocate 20.00 GiB",
        "input tensor must fit into 32-bit index math",
        "index tensor must fit into 32-bit indexing",
        # The one that got through: no phrase in common with the others.
        "upsample_nearest3d only supports output tensors with less than "
        "INT_MAX elements, but got [4, 32, 256, 256, 256]",
    ],
)
def test_a_chunk_that_must_be_tiled_is_recognised(message):
    assert too_big_for_one_pass(RuntimeError(message))


@pytest.mark.parametrize(
    "message",
    [
        "expected 5D input, got 4D",
        "CUDA error: device-side assert triggered",
        "mat1 and mat2 shapes cannot be multiplied",
    ],
)
def test_a_real_bug_is_not_mistaken_for_a_size_problem(message):
    """Tiling a chunk that failed for another reason hides the reason."""
    assert not too_big_for_one_pass(RuntimeError(message))


def test_the_default_fallback_window_stays_under_the_32_bit_limit():
    """The limit is what forced tiling; the window must not hit it again.

    The 352-cubed chunk that ran put its widest full-resolution layer at 0.97
    of the limit, so that layer is at most 48 channels. A 256-cubed window
    leaves a factor of two and a half in hand.
    """
    widest = int(2**31 / 352**3)  # 49, from the chunk size known to work
    assert AnatomixFeatures().fallback_window ** 3 * widest < 2**31


def test_the_window_batch_stays_inside_the_element_limit():
    """The limit is on the batched tensor, so the batch has to fit as well.

    Four 256-cubed windows of 32 channels is exactly 2**31, one element over
    INT_MAX, and it failed on a device with seventy gigabytes free.
    """
    ex = AnatomixFeatures(sw_batch=4)
    for window in (384, 256, 192, 128, 64):
        batch = ex._window_batch(window)
        assert 1 <= batch <= 4
        assert batch * 32 * window**3 < 2**31, window


def test_the_window_that_failed_on_the_cluster_would_now_be_batched_smaller():
    ex = AnatomixFeatures(sw_batch=4)
    assert 4 * 32 * 256**3 >= 2**31, "the case that failed"
    assert ex._window_batch(256) < 4


def test_a_small_window_is_not_penalised():
    """Sizing the batch must not throw away parallelism it does not need."""
    ex = AnatomixFeatures(sw_batch=4)
    assert ex._window_batch(64) == 4

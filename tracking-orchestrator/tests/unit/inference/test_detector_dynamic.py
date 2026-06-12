"""Unit tests for PersonDetector dynamic-batch mode.

All Triton calls are mocked — no GPU required.  Tests cover:
  1. Dynamic mode sends exactly N frames (no padding) for N=1, 3, 8.
  2. Static mode (flag off) preserves byte-identical padded request building
     (regression golden: 3 frames → padded to 8).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import numpy.typing as npt
import pytest

from app.inference.detector import PersonDetector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockTritonClient:
    def __init__(self, batch_size: int) -> None:
        # Return a zero output tensor whose leading dim matches whatever batch
        # the caller sends.  Tests inspect .infer call args directly.
        self.infer = AsyncMock(
            side_effect=lambda **kwargs: {
                "output0": np.zeros((kwargs["inputs"][0][1].shape[0], 300, 6), dtype=np.float32)
            }
        )
        self.is_model_ready = AsyncMock(return_value=True)


def _make_images(n: int) -> list[npt.NDArray[np.uint8]]:
    rng = np.random.default_rng(42)
    return [rng.integers(0, 255, (120, 160, 3), dtype=np.uint8) for _ in range(n)]


def _sent_batch_shape(client: _MockTritonClient) -> tuple[int, ...]:
    """Return the shape of the images tensor sent in the first infer() call."""
    return tuple(client.infer.call_args_list[0].kwargs["inputs"][0][1].shape)


# ---------------------------------------------------------------------------
# Dynamic mode — no padding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("n_frames", [1, 3, 8])
async def test_dynamic_mode_sends_exact_frame_count(n_frames: int) -> None:
    client = _MockTritonClient(batch_size=n_frames)
    detector = PersonDetector(client, dynamic_batch=True)  # type: ignore[arg-type]

    results = await detector.detect_batch(_make_images(n_frames))

    assert len(results) == n_frames
    assert client.infer.await_count == 1
    batch_shape = _sent_batch_shape(client)
    assert batch_shape[0] == n_frames, f"Expected batch dim {n_frames}, got {batch_shape[0]}"
    assert batch_shape[1:] == (3, 640, 640)


@pytest.mark.asyncio
async def test_dynamic_mode_no_padding_three_frames() -> None:
    """Regression: dynamic mode must NOT pad a 3-frame batch to 8."""
    client = _MockTritonClient(batch_size=3)
    detector = PersonDetector(client, dynamic_batch=True)  # type: ignore[arg-type]

    await detector.detect_batch(_make_images(3))

    batch_shape = _sent_batch_shape(client)
    assert batch_shape[0] == 3, "Dynamic mode padded frames — must not pad"


# ---------------------------------------------------------------------------
# Static mode (flag off) — preserves existing padded behaviour (golden)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_static_mode_pads_three_frames_to_eight() -> None:
    """Flag off (default): 3 frames must be padded to static_batch_size=8."""
    client = _MockTritonClient(batch_size=8)
    # Patch return so raw_batch has 8 rows as the static model produces.
    client.infer = AsyncMock(return_value={"output0": np.zeros((8, 300, 6), dtype=np.float32)})
    detector = PersonDetector(client, static_batch_size=8, dynamic_batch=False)  # type: ignore[arg-type]

    results = await detector.detect_batch(_make_images(3))

    assert len(results) == 3
    batch_shape = _sent_batch_shape(client)
    assert batch_shape[0] == 8, (
        f"Static mode should pad 3 frames to 8, got batch dim {batch_shape[0]}"
    )


@pytest.mark.asyncio
async def test_static_mode_default_flag_off() -> None:
    """PersonDetector() with no flags must behave identically to explicit dynamic_batch=False."""
    client_default = _MockTritonClient(batch_size=8)
    client_default.infer = AsyncMock(
        return_value={"output0": np.zeros((8, 300, 6), dtype=np.float32)}
    )
    client_explicit = _MockTritonClient(batch_size=8)
    client_explicit.infer = AsyncMock(
        return_value={"output0": np.zeros((8, 300, 6), dtype=np.float32)}
    )

    detector_default = PersonDetector(client_default)  # type: ignore[arg-type]
    detector_explicit = PersonDetector(client_explicit, dynamic_batch=False)  # type: ignore[arg-type]

    images = _make_images(3)
    await detector_default.detect_batch(images)
    await detector_explicit.detect_batch(images)

    shape_default = _sent_batch_shape(client_default)
    shape_explicit = _sent_batch_shape(client_explicit)
    assert shape_default == shape_explicit

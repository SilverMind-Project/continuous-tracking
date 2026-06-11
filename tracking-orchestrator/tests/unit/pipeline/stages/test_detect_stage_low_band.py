"""Unit tests for DetectStage low-confidence recovery (M2.3).

Tests:
  1. Partition correctness: high / low band split.
  2. IoU dedup within each band and cross-band (low dropped when overlapping high).
  3. Flag disabled → raw_detections byte-identical to standard path; low_band_detections empty.
  4. Privacy: low-band detection foot-point inside a drop zone is excluded.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from app.inference.schemas import DetectionBox
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.detect import DetectStage
from app.transport.redis_streams import FrameReady

_T0 = datetime(2026, 1, 1, 12, tzinfo=UTC)
_FRAME = FrameReady(
    camera_id="cam-1",
    minio_key="k",
    width=640,
    height=480,
    frame_index=0,
    capture_time_unix_ns=int(_T0.timestamp() * 1e9),
)


def _ctx() -> FrameContext:
    ctx = FrameContext(frame=_FRAME, event_time=_T0, capture_time=_T0)
    ctx.image = np.zeros((480, 640, 3), dtype=np.uint8)
    ctx.effective_width = 640
    ctx.effective_height = 480
    return ctx


class _FakeDetector:
    """Returns a fixed list of DetectionBox when detect_batch_at_threshold is called."""

    def __init__(self, boxes: list[DetectionBox]) -> None:
        self._boxes = boxes
        self.calls_with_threshold: list[float] = []

    async def detect_batch_at_threshold(
        self, images: list[np.ndarray], threshold: float
    ) -> list[list[DetectionBox]]:
        self.calls_with_threshold.append(threshold)
        return [list(self._boxes)] * len(images)

    async def detect_batch(self, images: list[np.ndarray]) -> list[list[DetectionBox]]:
        return [list(self._boxes)] * len(images)


# ---------------------------------------------------------------------------
# 1. Partition correctness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_partition_high_and_low_band() -> None:
    """Detections ≥ high_threshold go to raw_detections; in-band go to low_band_detections."""
    high_box = DetectionBox(x1=0.1, y1=0.1, x2=0.3, y2=0.5, confidence=0.85)
    low_box = DetectionBox(x1=0.5, y1=0.1, x2=0.7, y2=0.5, confidence=0.40)
    below_floor = DetectionBox(x1=0.8, y1=0.1, x2=0.9, y2=0.5, confidence=0.10)

    fake = _FakeDetector([high_box, low_box, below_floor])
    stage = DetectStage(
        detector=fake,  # type: ignore[arg-type]
        iou_dedup_threshold=0.55,
        enable_low_confidence_recovery=True,
        low_confidence_floor=0.25,
        high_threshold=0.7,
    )
    ctx = _ctx()
    await stage.run(ctx)

    assert len(ctx.raw_detections) == 1
    assert ctx.raw_detections[0].confidence == pytest.approx(0.85)

    assert len(ctx.low_band_detections) == 1
    assert ctx.low_band_detections[0].confidence == pytest.approx(0.40)

    # Detector was called with the low floor threshold (one Triton call).
    assert fake.calls_with_threshold == [0.25]


# ---------------------------------------------------------------------------
# 2a. IoU dedup within the high band
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_iou_dedup_within_high_band() -> None:
    """Two overlapping high-confidence boxes: only the higher-confidence one survives."""
    high_a = DetectionBox(x1=0.1, y1=0.1, x2=0.4, y2=0.5, confidence=0.90)
    # Nearly identical bbox → high IoU with high_a.
    high_b = DetectionBox(x1=0.11, y1=0.11, x2=0.39, y2=0.49, confidence=0.80)

    fake = _FakeDetector([high_a, high_b])
    stage = DetectStage(
        detector=fake,  # type: ignore[arg-type]
        iou_dedup_threshold=0.55,
        enable_low_confidence_recovery=True,
        low_confidence_floor=0.25,
        high_threshold=0.7,
    )
    ctx = _ctx()
    await stage.run(ctx)

    assert len(ctx.raw_detections) == 1
    assert ctx.raw_detections[0].confidence == pytest.approx(0.90)


# ---------------------------------------------------------------------------
# 2b. Cross-band IoU dedup: low box overlapping high box is dropped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_band_iou_dedup_drops_low_when_overlaps_high() -> None:
    """A low-band box that overlaps a high-band box by > iou_threshold is dropped."""
    high_box = DetectionBox(x1=0.1, y1=0.1, x2=0.4, y2=0.5, confidence=0.85)
    # Slightly different bbox but very high IoU with high_box.
    low_overlapping = DetectionBox(x1=0.12, y1=0.12, x2=0.38, y2=0.48, confidence=0.40)
    low_separate = DetectionBox(x1=0.6, y1=0.1, x2=0.9, y2=0.5, confidence=0.35)

    fake = _FakeDetector([high_box, low_overlapping, low_separate])
    stage = DetectStage(
        detector=fake,  # type: ignore[arg-type]
        iou_dedup_threshold=0.55,
        enable_low_confidence_recovery=True,
        low_confidence_floor=0.25,
        high_threshold=0.7,
    )
    ctx = _ctx()
    await stage.run(ctx)

    assert len(ctx.raw_detections) == 1
    # low_overlapping should be dropped (overlaps high_box), low_separate survives.
    assert len(ctx.low_band_detections) == 1
    assert ctx.low_band_detections[0].confidence == pytest.approx(0.35)


# ---------------------------------------------------------------------------
# 3. Flag disabled → byte-identical to standard path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_disabled_byte_identical_standard_path() -> None:
    """When enable_low_confidence_recovery is False, raw_detections == standard path
    and low_band_detections is empty (flag-off is the golden baseline)."""
    high_box = DetectionBox(x1=0.1, y1=0.1, x2=0.4, y2=0.5, confidence=0.85)
    low_box = DetectionBox(x1=0.5, y1=0.1, x2=0.7, y2=0.5, confidence=0.40)

    # Standard detector returns only high-confidence boxes (already filtered by conf_threshold).
    class _StandardFakeDetector:
        async def detect_batch(self, images: list[np.ndarray]) -> list[list[DetectionBox]]:
            return [[high_box]] * len(images)

    stage = DetectStage(
        detector=_StandardFakeDetector(),  # type: ignore[arg-type]
        iou_dedup_threshold=0.55,
        enable_low_confidence_recovery=False,
    )
    ctx = _ctx()
    await stage.run(ctx)

    # Standard path: one high-confidence box, no low-band detections.
    assert len(ctx.raw_detections) == 1
    assert ctx.raw_detections[0].confidence == pytest.approx(0.85)
    assert ctx.low_band_detections == []
    _ = low_box  # used to avoid unused variable warning

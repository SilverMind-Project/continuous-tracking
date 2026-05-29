"""Tests for StageRunner and common stage behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.base import FrameStage, StageRunner
from app.transport.redis_streams import FrameReady

_NOW_NS = 1_700_000_000_000_000_000  # ~Nov 2023


def _make_ctx(camera_id: str = "cam-1", frame_index: int = 0) -> FrameContext:
    from datetime import UTC, datetime

    frame = FrameReady(
        camera_id=camera_id,
        minio_key=f"frames/{camera_id}/{frame_index}.jpg",
        frame_index=frame_index,
        capture_time_unix_ns=_NOW_NS,
        received_time_unix_ns=_NOW_NS + 100_000_000,
        width=640,
        height=480,
    )
    return FrameContext(
        frame=frame,
        event_time=datetime.now(UTC),
        capture_time=datetime.fromtimestamp(frame.capture_time_unix_ns / 1e9, tz=UTC),
    )


class _RecordingStage(FrameStage):
    """Stage that records call order and can optionally raise."""

    def __init__(self, name: str, record: list[str], exc: Exception | None = None) -> None:
        self.name = name
        self._record = record
        self._exc = exc

    async def run(self, ctx: FrameContext) -> None:
        self._record.append(self.name)
        if self._exc is not None:
            raise self._exc


class TestStageRunner:
    @pytest.mark.asyncio
    async def test_stage_runner_runs_in_order(self) -> None:
        """Stages must run in the order they were passed to StageRunner."""
        ctx = _make_ctx()
        order: list[str] = []

        stages: list[FrameStage] = [
            _RecordingStage("fetch", order),
            _RecordingStage("detect", order),
            _RecordingStage("privacy", order),
            _RecordingStage("publish", order),
        ]

        runner = StageRunner(stages)
        await runner.run(ctx)

        assert order == ["fetch", "detect", "privacy", "publish"]

    @pytest.mark.asyncio
    async def test_stage_runner_reraises_and_logs_stage_name(self) -> None:
        """StageRunner must re-raise stage exceptions and log the stage name."""
        ctx = _make_ctx()
        order: list[str] = []
        boom = RuntimeError("detector unavailable")

        stages: list[FrameStage] = [
            _RecordingStage("fetch", order),
            _RecordingStage("detect", order, exc=boom),
            _RecordingStage("privacy", order),
        ]

        runner = StageRunner(stages)

        with pytest.raises(RuntimeError, match="detector unavailable"):
            await runner.run(ctx)

        # privacy must not run after detect fails.
        assert order == ["fetch", "detect"]


class TestFetchStage:
    @pytest.mark.asyncio
    async def test_fetch_stage_dimension_mismatch_uses_actual_image_size(self) -> None:
        """When the fetched image shape differs from FrameReady metadata,
        effective_width/height must reflect the actual image dimensions."""
        from app.pipeline.stages.fetch import FetchStage

        ctx = _make_ctx()

        # reported: 640x480, actual image: 320x240
        async def _fake_fetch(minio_key: str) -> np.ndarray:
            return np.zeros((240, 320, 3), dtype=np.uint8)

        class FakeFetcher:
            async def fetch_rgb(self, minio_key: str) -> np.ndarray:
                return await _fake_fetch(minio_key)

        stage = FetchStage(frame_fetcher=FakeFetcher())
        await stage.run(ctx)

        assert ctx.effective_width == 320, f"expected 320, got {ctx.effective_width}"
        assert ctx.effective_height == 240, f"expected 240, got {ctx.effective_height}"
        assert ctx.image is not None
        assert ctx.image.shape == (240, 320, 3)


class TestDetectStage:
    @pytest.mark.asyncio
    async def test_detect_stage_iou_dedup_preserves_confidence_order(self) -> None:
        """IoU dedup must keep the highest-confidence box from each cluster."""
        from app.inference.schemas import DetectionBox
        from app.pipeline.stages.detect import _iou_dedup_detections

        # Three boxes: two overlapping heavily (0.4 IoU at 0.5 threshold),
        # one isolated. The lower-confidence overlapping box should be suppressed.
        boxes = [
            DetectionBox(x1=0.1, y1=0.1, x2=0.5, y2=0.5, confidence=0.95),
            DetectionBox(x1=0.15, y1=0.15, x2=0.55, y2=0.55, confidence=0.60),
            DetectionBox(x1=0.7, y1=0.7, x2=0.9, y2=0.9, confidence=0.80),
        ]

        result = _iou_dedup_detections(boxes, iou_threshold=0.5)

        assert len(result) == 2, f"expected 2 boxes after dedup, got {len(result)}"
        confidences = [b.confidence for b in result]
        assert 0.95 in confidences, "highest-confidence box must be kept"
        assert 0.80 in confidences, "isolated box must be kept"
        assert 0.60 not in confidences, "overlapping lower-confidence box must be suppressed"

    @pytest.mark.asyncio
    async def test_detect_stage_run_batch_uses_one_detector_call(self) -> None:
        """run_batch should send multiple frame images through one detector request."""
        from app.inference.schemas import DetectionBox
        from app.pipeline.stages.detect import DetectStage

        class FakeDetector:
            def __init__(self) -> None:
                self.batch_sizes: list[int] = []

            async def detect_batch(self, images: list[np.ndarray]) -> list[list[DetectionBox]]:
                self.batch_sizes.append(len(images))
                return [
                    [DetectionBox(x1=0.1, y1=0.1, x2=0.4, y2=0.8, confidence=0.9)] for _ in images
                ]

        ctx1 = _make_ctx("cam-1", 1)
        ctx2 = _make_ctx("cam-2", 1)
        ctx1.image = np.zeros((480, 640, 3), dtype=np.uint8)
        ctx2.image = np.zeros((480, 640, 3), dtype=np.uint8)
        detector = FakeDetector()
        stage = DetectStage(detector=detector)  # type: ignore[arg-type]

        await stage.run_batch([ctx1, ctx2])

        assert detector.batch_sizes == [2]
        assert len(ctx1.raw_detections) == 1
        assert len(ctx2.raw_detections) == 1
        assert len(ctx1._detection_ids) == 1
        assert len(ctx2._detection_ids) == 1

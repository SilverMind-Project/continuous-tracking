"""WT4: tests for DetectionBackfillStage stamping ph_id from det_to_ph."""

from __future__ import annotations

from datetime import UTC, datetime

from app.domain import BoundingBox, Detection, FloorPoint
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.detection_backfill import DetectionBackfillStage


def _make_ctx(
    *,
    detections: list[Detection] | None = None,
    det_to_ph: dict[str, str] | None = None,
) -> FrameContext:
    from unittest.mock import MagicMock

    frame = MagicMock()
    ctx = FrameContext(
        frame=frame,  # type: ignore[arg-type]
        event_time=datetime.now(UTC),
        capture_time=datetime.now(UTC),
    )
    ctx.domain_detections = detections or []
    ctx.det_to_ph = det_to_ph or {}
    return ctx


def _make_det(
    detection_id: str = "det-1",
    camera_id: str = "cam1",
    ph_id: str = "",
) -> Detection:
    return Detection(
        detection_id=detection_id,
        camera_id=camera_id,
        bbox=BoundingBox(0, 0, 100, 200),
        embedding=[1.0, 0.0, 0.0],
        capture_time=datetime.now(UTC),
        event_time=datetime.now(UTC),
        confidence=0.9,
        ph_id=ph_id,
        floor_point=FloorPoint(0, 0, calibrated=True),
    )


class TestDetectionBackfillPh:
    async def test_backfill_stamps_ph_id_from_det_to_ph(self) -> None:
        det_a = _make_det("det-1")
        det_b = _make_det("det-2")
        det_c = _make_det("det-3")
        ctx = _make_ctx(
            detections=[det_a, det_b, det_c],
            det_to_ph={"det-1": "ph-a", "det-2": "ph-b", "det-3": ""},
        )
        stage = DetectionBackfillStage()
        await stage.run(ctx)

        assert ctx.domain_detections[0].ph_id == "ph-a"
        assert ctx.domain_detections[1].ph_id == "ph-b"
        assert ctx.domain_detections[2].ph_id == ""

    async def test_backfill_no_op_without_det_to_ph(self) -> None:
        det_a = _make_det("det-1")
        det_b = _make_det("det-2")
        ctx = _make_ctx(
            detections=[det_a, det_b],
            det_to_ph={},  # empty
        )
        stage = DetectionBackfillStage()
        await stage.run(ctx)

        assert ctx.domain_detections[0].ph_id == ""
        assert ctx.domain_detections[1].ph_id == ""

    async def test_backfill_no_op_without_detections(self) -> None:
        ctx = _make_ctx(
            detections=[],
            det_to_ph={"det-1": "ph-a"},
        )
        stage = DetectionBackfillStage()
        await stage.run(ctx)

        assert ctx.domain_detections == []

    async def test_backfill_does_not_overwrite_existing(self) -> None:
        det_a = _make_det("det-1", ph_id="existing-gt")
        ctx = _make_ctx(
            detections=[det_a],
            det_to_ph={"det-1": "ph-a"},
        )
        stage = DetectionBackfillStage()
        await stage.run(ctx)

        # replace() overwrites ph_id from det_to_ph regardless
        assert ctx.domain_detections[0].ph_id == "ph-a"

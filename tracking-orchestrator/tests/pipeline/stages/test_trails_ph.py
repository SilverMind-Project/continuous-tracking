"""WT4: tests for TrailsStage keying on ph_id (PH id)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from app.domain import BoundingBox, Detection, FloorPoint
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.posture_trails import TrailsStage


def _make_ctx(
    *,
    detections: list[Detection] | None = None,
    effective_width: int = 1920,
    effective_height: int = 1080,
) -> FrameContext:
    frame = MagicMock()
    ctx = FrameContext(
        frame=frame,  # type: ignore[arg-type]
        event_time=datetime.now(UTC),
        capture_time=datetime.now(UTC),
    )
    ctx.domain_detections = detections or []
    ctx.effective_width = effective_width
    ctx.effective_height = effective_height
    return ctx


def _make_det(
    detection_id: str = "det-1",
    ph_id: str = "",
    x_min: int = 100,
    y_min: int = 200,
    x_max: int = 300,
    y_max: int = 600,
) -> Detection:
    return Detection(
        detection_id=detection_id,
        camera_id="cam1",
        bbox=BoundingBox(x_min, y_min, x_max, y_max),
        embedding=[1.0, 0.0, 0.0],
        capture_time=datetime.now(UTC),
        event_time=datetime.now(UTC),
        confidence=0.9,
        ph_id=ph_id,
        floor_point=FloorPoint(0, 0, calibrated=True),
    )


class TestTrailsPh:
    async def test_trails_keyed_on_ph_id(self) -> None:
        """2 detections with ph_id; 1 without → 2 trail entries."""
        det_a = _make_det("det-1", ph_id="ph-a")
        det_b = _make_det("det-2", ph_id="ph-b")
        det_c = _make_det("det-3", ph_id="")  # no PH

        stage = TrailsStage()
        ctx = _make_ctx(detections=[det_a, det_b, det_c])
        await stage.run(ctx)  # type: ignore[arg-type]

        assert len(ctx.trail_by_tracklet_snapshot) == 2
        assert "ph-a" in ctx.trail_by_tracklet_snapshot
        assert "ph-b" in ctx.trail_by_tracklet_snapshot
        # Each PH has exactly 1 point from this first frame
        assert len(ctx.trail_by_tracklet_snapshot["ph-a"]) == 1
        assert len(ctx.trail_by_tracklet_snapshot["ph-b"]) == 1

    async def test_trail_evicts_phs_with_no_current_detection(self) -> None:
        """PH present in frame 1 but absent in frame 2 → evicted."""
        stage = TrailsStage()

        # Frame 1: PH-A and PH-B both detected
        det_a1 = _make_det("det-1", ph_id="ph-a")
        det_b1 = _make_det("det-2", ph_id="ph-b")
        ctx1 = _make_ctx(detections=[det_a1, det_b1])
        await stage.run(ctx1)  # type: ignore[arg-type]
        assert "ph-b" in ctx1.trail_by_tracklet_snapshot

        # Frame 2: only PH-A detected
        det_a2 = _make_det("det-3", ph_id="ph-a")
        ctx2 = _make_ctx(detections=[det_a2])
        await stage.run(ctx2)  # type: ignore[arg-type]

        assert "ph-b" not in ctx2.trail_by_tracklet_snapshot
        assert "ph-a" in ctx2.trail_by_tracklet_snapshot
        # PH-A now has 2 points (one from each frame)
        assert len(ctx2.trail_by_tracklet_snapshot["ph-a"]) == 2

    async def test_trail_maxlen_enforced(self) -> None:
        """15 consecutive points for PH-A → trail capped at TRAIL_MAXLEN (12)."""
        stage = TrailsStage(trail_maxlen=12)

        for i in range(15):
            det = _make_det(f"det-{i}", ph_id="ph-a")
            ctx = _make_ctx(detections=[det])
            await stage.run(ctx)  # type: ignore[arg-type]

        # Last context should have the final trail
        assert len(ctx.trail_by_tracklet_snapshot["ph-a"]) == 12

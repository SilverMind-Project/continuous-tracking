"""Constructor injection for ExternalIdentityEvidenceSource and grade-aware attachment.

Verifies:
1. WorldTrackingStage receives ExternalIdentityEvidenceSource via its constructor
   and PipelineDependencies.external_evidence wires it cleanly with no private
   attribute mutation.
2. When a native face anchor and a CC assertion anchor share a detection_id,
   the native anchor wins in every cc_assertion_mode; the duplicate-detection
   defect counter does not move while cts_face_anchor_grade_overlap_total counts
   the expected cross-grade overlap.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from prometheus_client import CollectorRegistry

from app.domain import (
    AnchorOrigin,
    BoundingBox,
    Detection,
    ExternalIdentityAssertion,
    FaceAnchor,
    FloorPoint,
    WorldObservation,
)
from app.observability import metrics as metrics_pkg
from app.observability.metrics import build_metrics
from app.pipeline.frame_context import FrameContext
from app.pipeline.frame_pipeline import (
    FrameProcessingPipeline,
    PipelineConfig,
    PipelineDependencies,
)
from app.pipeline.stages.world_tracking import WorldTrackingStage
from app.pipeline.types import LiveConfigHolder
from app.services.camera_room_map import CameraRoomBinding, CameraRoomMap, RoomPolygonMap
from app.tracking.identity.evidence_grades import EvidenceGradeName
from app.tracking.world.tracker import WorldTrackerResult
from app.transport.redis_streams import FrameReady

_NOW = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)


class _FakeExternalSource:
    def __init__(self, items: Sequence[ExternalIdentityAssertion]) -> None:
        self._items = tuple(items)
        self.calls: list[float] = []

    async def recent(self, max_age_s: float = 30.0) -> Sequence[ExternalIdentityAssertion]:
        self.calls.append(max_age_s)
        return self._items


def _frame_ctx(
    *,
    face_anchors: list[FaceAnchor] | None = None,
    camera_id: str = "cam-1",
    detection_id: str = "det-1",
) -> FrameContext:
    frame = FrameReady(
        camera_id=camera_id,
        minio_key="",
        frame_index=1,
        capture_time_unix_ns=int(_NOW.timestamp() * 1e9),
        received_time_unix_ns=int(_NOW.timestamp() * 1e9),
        width=640,
        height=480,
    )
    ctx = FrameContext(
        frame=frame,
        event_time=_NOW,
        capture_time=_NOW,
        effective_width=640,
        effective_height=480,
    )
    ctx.domain_detections = [
        Detection(
            detection_id=detection_id,
            camera_id=camera_id,
            bbox=BoundingBox(x_min=100, y_min=50, x_max=250, y_max=400),
            confidence=0.9,
            floor_point=FloorPoint(x_mm=1000, y_mm=2000, calibrated=True),
            embedding=[],
            capture_time=_NOW,
            event_time=_NOW,
        )
    ]
    ctx.face_anchors = list(face_anchors or [])
    return ctx


@pytest.mark.asyncio
async def test_world_tracking_stage_uses_constructor_injected_source() -> None:
    source = _FakeExternalSource(
        [
            ExternalIdentityAssertion(
                person_id="alice",
                calibrated_confidence=0.91,
                camera_id="recamera_kitchen",
                captured_at=_NOW,
                room_name="Kitchen",
            )
        ]
    )
    captured_anchors: list[list[FaceAnchor]] = []

    async def _step(
        *,
        observations: list[WorldObservation],
        face_anchors: list[FaceAnchor],
        **_kw: Any,
    ) -> WorldTrackerResult:
        captured_anchors.append(face_anchors)
        return WorldTrackerResult(updated_phs=[], snapshots=[], continuations=[])

    tracker = MagicMock()
    tracker.step = AsyncMock(side_effect=_step)
    room_map = CameraRoomMap()
    await room_map.set_all(
        [CameraRoomBinding(camera_id="cam-1", room_id="r1", room_name="Kitchen", bound_at=_NOW)]
    )

    stage = WorldTrackingStage(
        tracker=tracker,
        live_config=LiveConfigHolder(room_map, RoomPolygonMap()),
        external_evidence=source,
        cc_assertion_mode="enabled",
        anchor_match_window_s=25.0,
    )
    await stage.run(_frame_ctx())

    assert source.calls == [25.0]
    assert len(captured_anchors) == 1
    assert len(captured_anchors[0]) == 1
    assert captured_anchors[0][0].person_id == "alice"
    assert captured_anchors[0][0].origin == AnchorOrigin.CC_ASSERTION


@pytest.mark.asyncio
async def test_pipeline_dependencies_wires_external_evidence_into_world_tracking_stage() -> None:
    source = _FakeExternalSource([])
    pipeline = FrameProcessingPipeline(PipelineConfig(allow_skeleton=True))
    await pipeline.initialize(PipelineDependencies(external_evidence=source))

    assert pipeline._world_tracking_stage is not None
    assert pipeline._world_tracking_stage._external_evidence is source


@pytest.mark.asyncio
@pytest.mark.parametrize("cc_mode", ["off", "shadow", "enabled"])
async def test_native_and_cc_anchor_overlap_keeps_native_and_counts_overlap_not_duplicate(
    monkeypatch: pytest.MonkeyPatch,
    cc_mode: str,
) -> None:
    fresh_metrics = build_metrics(registry=CollectorRegistry())
    monkeypatch.setattr(metrics_pkg, "metrics", fresh_metrics)

    source = _FakeExternalSource(
        [
            ExternalIdentityAssertion(
                person_id="bob",
                calibrated_confidence=0.95,
                camera_id="recamera_kitchen",
                captured_at=_NOW,
                room_name="Kitchen",
            )
        ]
    )
    native = FaceAnchor(
        person_id="alice",
        match_score=0.72,
        similarity=0.72,
        detection_id="det-1",
        camera_id="cam-1",
        captured_at=_NOW,
        recognition_state="recognized",
        origin=AnchorOrigin.NATIVE,
    )
    captured_obs: list[list[WorldObservation]] = []

    async def _step(
        *,
        observations: list[WorldObservation],
        face_anchors: list[FaceAnchor],
        **_kw: Any,
    ) -> WorldTrackerResult:
        captured_obs.append(observations)
        return WorldTrackerResult(updated_phs=[], snapshots=[], continuations=[])

    tracker = MagicMock()
    tracker.step = AsyncMock(side_effect=_step)
    room_map = CameraRoomMap()
    await room_map.set_all(
        [CameraRoomBinding(camera_id="cam-1", room_id="r1", room_name="Kitchen", bound_at=_NOW)]
    )
    stage = WorldTrackingStage(
        tracker=tracker,
        live_config=LiveConfigHolder(room_map, RoomPolygonMap()),
        external_evidence=source,
        cc_assertion_mode=cc_mode,
    )

    await stage.run(_frame_ctx(face_anchors=[native]))

    attached = captured_obs[0][0].face_anchor
    assert attached is not None
    assert attached.person_id == "alice"
    assert attached.origin == AnchorOrigin.NATIVE

    assert fresh_metrics.cts_face_anchor_duplicate_detection_total._value.get() == 0.0
    expected_overlap = 1.0 if cc_mode == "enabled" else 0.0
    overlap = fresh_metrics.cts_face_anchor_grade_overlap_total.labels(
        kept=str(EvidenceGradeName.NATIVE_FRAME),
        dropped=str(EvidenceGradeName.CC_ASSERTION),
    )
    assert overlap._value.get() == expected_overlap


def test_cross_grade_precedence_keeps_native_even_when_cc_arrives_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh_metrics = build_metrics(registry=CollectorRegistry())
    monkeypatch.setattr(metrics_pkg, "metrics", fresh_metrics)

    stage = WorldTrackingStage(
        tracker=MagicMock(),
        live_config=LiveConfigHolder(CameraRoomMap(), RoomPolygonMap()),
        cc_assertion_mode="enabled",
    )
    obs = WorldObservation(
        camera_id="cam-1",
        frame_index=1,
        captured_at=_NOW,
        floor_point=FloorPoint(x_mm=1000, y_mm=2000, calibrated=True),
        bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=200),
        embedding=[],
        detection_confidence=0.9,
        detection_id="det-1",
    )
    cc_anchor = FaceAnchor(
        person_id="bob",
        match_score=0.95,
        similarity=0.95,
        detection_id="det-1",
        camera_id="cam-1",
        captured_at=_NOW,
        origin=AnchorOrigin.CC_ASSERTION,
    )
    native_anchor = FaceAnchor(
        person_id="alice",
        match_score=0.65,
        similarity=0.65,
        detection_id="det-1",
        camera_id="cam-1",
        captured_at=_NOW,
        origin=AnchorOrigin.NATIVE,
    )

    attached = stage._attach_face_anchors([obs], [cc_anchor, native_anchor])
    assert attached[0].face_anchor is native_anchor
    assert fresh_metrics.cts_face_anchor_duplicate_detection_total._value.get() == 0.0
    overlap = fresh_metrics.cts_face_anchor_grade_overlap_total.labels(
        kept=str(EvidenceGradeName.NATIVE_FRAME),
        dropped=str(EvidenceGradeName.CC_ASSERTION),
    )
    assert overlap._value.get() == 1.0

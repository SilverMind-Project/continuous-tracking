"""A slow person-id never delays a tracker round; its template applies on a later round."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from app.domain import BoundingBox, Detection, FaceAnchor, FloorPoint, Identity
from app.inference.face_id_client import (
    FaceQualityReading,
    TemplateMatchResult,
    TemplateMember,
    TemplateQualityReading,
)
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.world_tracking import WorldTrackingStage
from app.pipeline.types import LiveConfigHolder
from app.services.camera_room_map import CameraRoomMap, RoomPolygonMap
from app.services.face_templates import FaceTemplateService
from app.storage.base import InMemoryPHRepository, InMemoryWorldObservationRepository
from app.storage.gallery import InMemoryGalleryRepository
from app.tracking.identity.face_template_buffer import BufferedFace, FaceTemplatePolicy
from app.tracking.identity_resolver import IdentityResolver
from app.tracking.world.tracker import WorldTracker, WorldTrackerResult
from app.transport.redis_streams import FrameReady

_T0 = datetime.now(UTC)
_QUALITY = FaceQualityReading(
    score=0.5, resolution=0.4, sharpness=0.6, detection=0.9, pitch=1.0, version="heuristic-v1"
)


class _BlockedClient:
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.calls = 0

    async def match_template(
        self, members: Sequence[TemplateMember], embedding_model_version: str
    ) -> TemplateMatchResult | None:
        self.calls += 1
        await self.release.wait()
        return TemplateMatchResult(
            person_id="alice",
            recognition_state="recognized",
            best_candidate_id="alice",
            raw_similarity=0.6,
            second_best_similarity=0.1,
            member_count=len(members),
            aggregator_version="qwmean-v1",
            quality=TemplateQualityReading(score=0.5, resolution=0.4, sharpness=0.6),
            quality_version="heuristic-v1",
            calibrated_confidence=0.9,
            calibration_profile="template",
            calibration_status="ready",
        )


class _RecordingTracker(WorldTracker):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.template_anchors_per_round: list[list[FaceAnchor]] = []

    async def step(self, *args: object, **kwargs: object) -> WorldTrackerResult:
        anchors = kwargs.get("face_anchors") or []
        self.template_anchors_per_round.append(
            [a for a in anchors if a.aggregation == "template"]  # type: ignore[attr-defined]
        )
        return await super().step(*args, **kwargs)  # type: ignore[arg-type]


def _ctx(step: int) -> FrameContext:
    captured = _T0 + timedelta(seconds=0.5 * step)
    frame = FrameReady(
        camera_id="cam-1",
        minio_key="k",
        width=640,
        height=480,
        frame_index=step,
        capture_time_unix_ns=int(captured.timestamp() * 1e9),
    )
    ctx = FrameContext(
        frame=frame,
        event_time=captured,
        capture_time=captured,
        effective_width=640,
        effective_height=480,
    )
    det_id = f"det-{step}"
    ctx.domain_detections = [
        Detection(
            detection_id=det_id,
            camera_id="cam-1",
            bbox=BoundingBox(x_min=100, y_min=50, x_max=200, y_max=400),
            embedding=[],
            capture_time=captured,
            event_time=captured,
            confidence=0.9,
            floor_point=FloorPoint(x_mm=0, y_mm=0, calibrated=False),
        )
    ]
    v = np.ones(512, dtype=np.float32) + 0.01 * step
    ctx.face_template_faces = [
        BufferedFace(
            detection_id=det_id,
            camera_id="cam-1",
            captured_at=captured,
            embedding=v / np.linalg.norm(v),
            quality=_QUALITY,
            det_score=0.8,
            yaw_deg=5.0,
            single_frame_person_id=None,
            embedding_model_version="buffalo_l",
        )
    ]
    return ctx


@pytest.mark.asyncio
async def test_a_blocked_service_does_not_delay_rounds_and_the_result_lands_later() -> None:
    client = _BlockedClient()
    gallery = InMemoryGalleryRepository()
    for identity_id in ("alice", "bob"):
        await gallery.upsert_identity(
            Identity(identity_id=identity_id, display_name=identity_id, enrolled_at=_T0)
        )
    templates = FaceTemplateService(
        FaceTemplatePolicy(mode="enabled"), client, expected_quality_version="heuristic-v1"
    )
    tracker = _RecordingTracker(
        ph_repo=InMemoryPHRepository(),
        obs_repo=InMemoryWorldObservationRepository(),
        identity_resolver=IdentityResolver(gallery_repo=gallery),
        gallery_repo=gallery,
    )
    stage = WorldTrackingStage(
        tracker=tracker,
        live_config=LiveConfigHolder(CameraRoomMap(), RoomPolygonMap()),
        cc_assertion_mode="off",
        face_templates=templates,
    )

    # Three faces trigger a request; person-id then hangs. Every round still completes promptly.
    for step in range(5):
        await asyncio.wait_for(stage.run_many([_ctx(step)]), timeout=1.0)
    await asyncio.sleep(0)  # in-memory rounds never yield; let the request start
    assert client.calls == 1
    assert templates.matcher.in_flight() == 1
    assert all(not anchors for anchors in tracker.template_anchors_per_round)

    client.release.set()
    await templates.matcher.drain()
    last = _ctx(5)
    await stage.run_many([last])

    assert [a.person_id for a in tracker.template_anchors_per_round[-1]] == ["alice"]
    assert [d.identity_id for d in last.outcome_decisions] == ["alice"]
    assert [d.authority for d in last.outcome_decisions] == ["direct_face"]
    await templates.stop()

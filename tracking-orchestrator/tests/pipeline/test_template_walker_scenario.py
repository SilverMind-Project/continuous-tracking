"""A walking person's small faces add up to an identification through a face template.

Stage level: ``FaceIdentityStage`` (with a frame image and a scripted person-id client) and
``WorldTrackingStage.run_many`` (real tracker and resolver) over ten rounds of one person walking
past one camera. Every face is 18 to 24 px between the eyes and person-id cannot recognize any of
them alone; matched together as a template, they are recognized with a template-profile calibrated
confidence above the authority threshold. With templates enabled the PH commits; off, it does
not. The integration replay helpers bypass both stages and cannot show this.

The face-id cooldown is 0 here: production's 5 s cooldown yields about three faces per camera per
15 s (see the milestone's record), which a ten-round test cannot wait for.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import numpy as np
import numpy.typing as npt
import pytest

from app.domain import BoundingBox, Detection, FloorPoint, Identity
from app.inference.face_id_client import (
    CaptureContext,
    CropQueryResult,
    CropQueryStatus,
    FaceIdentificationClient,
    FaceQualityReading,
    FaceResult,
    TemplateMatchResult,
    TemplateMember,
    TemplateQualityReading,
)
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.face_identity import FaceIdentityStage
from app.pipeline.stages.world_tracking import WorldTrackingStage
from app.pipeline.types import FaceIdCameraConfig, LiveConfigHolder
from app.services.camera_room_map import CameraRoomMap, RoomPolygonMap
from app.services.face_templates import FaceTemplateService
from app.storage.base import InMemoryPHRepository, InMemoryWorldObservationRepository
from app.storage.face_stats import InMemoryFaceObservationStatsRepository
from app.storage.gallery import InMemoryGalleryRepository
from app.tracking.identity.face_template_buffer import FaceTemplatePolicy
from app.tracking.identity_resolver import IdentityResolver
from app.tracking.world.tracker import WorldTracker
from app.transport.redis_streams import FrameReady

_CAMERA = "cam-hall"
_W, _H = 640, 480
_T0 = datetime.now(UTC)


def _unit(seed: int) -> np.ndarray:
    v = np.random.RandomState(seed).randn(512)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _walker_face(step: int) -> np.ndarray:
    """One small, noisy glimpse of the same person (about 0.5 cosine between two glimpses)."""
    v = _unit(1) + _unit(1000 + step)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _box(step: int) -> BoundingBox:
    x = 60 + 30 * step  # walking left to right
    return BoundingBox(x_min=x, y_min=60, x_max=x + 80, y_max=380)


class _WalkerClient(FaceIdentificationClient):
    """person-id for a passer-by: every single face unrecognized, the template recognized."""

    def __init__(self) -> None:
        super().__init__("http://unused.test")
        self.template_requests = 0
        self.step = 0

    async def identify_crops(
        self,
        crops: list[npt.NDArray[np.uint8]],
        crop_bboxes_norm: list[tuple[float, float, float, float]],
        context: CaptureContext,
        *,
        return_embedding: bool = False,
    ) -> list[CropQueryResult]:
        box = _box(self.step)
        cx, cy = (box.x_min + box.x_max) / 2, box.y_min + 30
        interocular = 18.0 + (self.step % 4) * 2.0  # 18 to 24 px
        face = FaceResult(
            person_id="unknown",
            name="Guest",
            bbox_normalized=[(cx - 9) / _W, (cy - 11) / _H, (cx + 9) / _W, (cy + 11) / _H],
            recognition_state="unrecognized",
            best_candidate_id="alice",
            raw_similarity=0.21,
            yaw_deg=12.0,
            det_score=0.7,
            interocular_px=interocular,
            quality=FaceQualityReading(
                score=0.3,
                resolution=0.2,
                sharpness=0.5,
                detection=0.7,
                pitch=1.0,
                version="heuristic-v1",
            ),
            embedding=_walker_face(self.step) if return_embedding else None,
            embedding_model_version="buffalo_l" if return_embedding else "",
        )
        return [
            CropQueryResult(
                crop_index=0,
                status=CropQueryStatus.FACE_FOUND,
                crop_width_px=int(crops[0].shape[1]),
                crop_height_px=int(crops[0].shape[0]),
                faces=(face,),
            )
        ]

    async def match_template(
        self, members: Sequence[TemplateMember], embedding_model_version: str
    ) -> TemplateMatchResult | None:
        self.template_requests += 1
        return TemplateMatchResult(
            person_id="alice",
            recognition_state="recognized",
            best_candidate_id="alice",
            raw_similarity=0.58,
            second_best_similarity=0.11,
            member_count=len(members),
            aggregator_version="qwmean-v1",
            quality=TemplateQualityReading(score=0.3, resolution=0.2, sharpness=0.5),
            quality_version="heuristic-v1",
            calibrated_confidence=0.9,
            calibration_profile="template",
            calibration_status="ready",
        )


def _ctx(step: int) -> FrameContext:
    captured = _T0 + timedelta(seconds=0.5 * step)
    frame = FrameReady(
        camera_id=_CAMERA,
        minio_key="k",
        width=_W,
        height=_H,
        frame_index=step,
        capture_time_unix_ns=int(captured.timestamp() * 1e9),
    )
    ctx = FrameContext(
        frame=frame,
        event_time=captured,
        capture_time=captured,
        effective_width=_W,
        effective_height=_H,
        image=np.zeros((_H, _W, 3), dtype=np.uint8),
    )
    box = _box(step)
    ctx.domain_detections = [
        Detection(
            detection_id=f"det-{step}",
            camera_id=_CAMERA,
            bbox=box,
            embedding=[],
            capture_time=captured,
            event_time=captured,
            confidence=0.9,
            floor_point=FloorPoint(x_mm=0, y_mm=0, calibrated=False),
        )
    ]
    ctx.crops = [np.zeros((box.y_max - box.y_min, box.x_max - box.x_min, 3), dtype=np.uint8)]
    return ctx


async def _walk(mode: str) -> tuple[list[str | None], list[str], _WalkerClient]:
    client = _WalkerClient()
    gallery = InMemoryGalleryRepository()
    for identity_id in ("alice", "bob"):
        await gallery.upsert_identity(
            Identity(identity_id=identity_id, display_name=identity_id, enrolled_at=_T0)
        )
    templates = (
        FaceTemplateService(
            FaceTemplatePolicy(mode=mode),  # type: ignore[arg-type]
            client,
            expected_quality_version="heuristic-v1",
        )
        if mode != "off"
        else None
    )
    face_stage = FaceIdentityStage(
        face_stats_repo=InMemoryFaceObservationStatsRepository(),
        face_id_client=client,
        face_id_cooldown_s=0.0,
        face_id_camera_configs={_CAMERA: FaceIdCameraConfig(enabled=True)},
        collect_template_faces=templates is not None,
    )
    world_stage = WorldTrackingStage(
        tracker=WorldTracker(
            ph_repo=InMemoryPHRepository(),
            obs_repo=InMemoryWorldObservationRepository(),
            identity_resolver=IdentityResolver(gallery_repo=gallery),
            gallery_repo=gallery,
        ),
        live_config=LiveConfigHolder(CameraRoomMap(), RoomPolygonMap()),
        cc_assertion_mode="off",
        face_templates=templates,
    )
    identities: list[str | None] = []
    authorities: list[str] = []
    for step in range(10):
        client.step = step
        ctx = _ctx(step)
        await face_stage.run(ctx)
        await world_stage.run_many([ctx])
        identities.extend(d.identity_id for d in ctx.outcome_decisions)
        authorities.extend(d.authority for d in ctx.outcome_decisions)
        if templates is not None:
            await templates.matcher.drain()  # deterministic: the request finishes between rounds
    if templates is not None:
        await templates.stop()
    return identities, authorities, client


@pytest.mark.asyncio
async def test_enabled_templates_identify_a_walker_no_single_face_can() -> None:
    identities, authorities, client = await _walk("enabled")

    assert client.template_requests >= 1
    assert "alice" in identities
    assert "direct_face" in authorities


@pytest.mark.asyncio
async def test_without_templates_the_walker_stays_unknown() -> None:
    identities, _, client = await _walk("off")

    assert client.template_requests == 0
    assert set(identities) <= {None}


@pytest.mark.asyncio
async def test_shadow_templates_are_requested_but_change_nothing() -> None:
    identities, _, client = await _walk("shadow")

    assert client.template_requests >= 1
    assert set(identities) <= {None}

"""The face template buffer is fed on every execution path, through the real entry points.

Feeding the buffer is a per-round side effect of ``WorldTrackingStage``; the direct, per-camera,
and cross-camera paths must each reach it. The stages before world tracking are replaced by a
runner that supplies one detection and one attributed template face per frame; world tracking,
the tracker, and the template service run for real.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import numpy as np
import pytest

from app.domain import BoundingBox, Detection, FloorPoint
from app.inference.face_id_client import FaceQualityReading, TemplateMatchResult, TemplateMember
from app.pipeline.frame_context import FrameContext
from app.pipeline.frame_pipeline import FrameProcessingPipeline, PipelineConfig
from app.pipeline.stages.world_tracking import WorldTrackingStage
from app.pipeline.types import LiveConfigHolder
from app.services.camera_room_map import CameraRoomMap, RoomPolygonMap
from app.services.face_templates import FaceTemplateService
from app.storage.base import InMemoryPHRepository, InMemoryWorldObservationRepository
from app.tracking.identity.face_template_buffer import BufferedFace, FaceTemplatePolicy
from app.tracking.world.tracker import WorldTracker
from app.transport.redis_streams import FrameReady

_QUALITY = FaceQualityReading(
    score=0.6, resolution=0.4, sharpness=0.6, detection=0.9, pitch=1.0, version="heuristic-v1"
)


class _NoMatchClient:
    async def match_template(
        self, members: Sequence[TemplateMember], embedding_model_version: str
    ) -> TemplateMatchResult | None:
        return None


def _frame(camera_id: str, index: int) -> FrameReady:
    return FrameReady(
        camera_id=camera_id,
        frame_index=index,
        minio_key=f"{camera_id}-{index}.jpg",
        width=640,
        height=480,
        capture_time_unix_ns=int(datetime.now(UTC).timestamp() * 1e9),
    )


def _supply(ctx: FrameContext) -> None:
    """What the stages before world tracking produce: one person, one attributed face."""
    det_id = f"det-{ctx.frame.camera_id}-{ctx.frame.frame_index}"
    ctx.effective_width, ctx.effective_height = 640, 480
    ctx.domain_detections = [
        Detection(
            detection_id=det_id,
            camera_id=ctx.frame.camera_id,
            bbox=BoundingBox(x_min=100, y_min=50, x_max=200, y_max=400),
            embedding=[],
            capture_time=ctx.capture_time,
            event_time=ctx.event_time,
            confidence=0.9,
            floor_point=FloorPoint(x_mm=0, y_mm=0, calibrated=False),
        )
    ]
    v = np.ones(512, dtype=np.float32)
    ctx.face_template_faces = [
        BufferedFace(
            detection_id=det_id,
            camera_id=ctx.frame.camera_id,
            captured_at=ctx.capture_time,
            embedding=v / np.linalg.norm(v),
            quality=_QUALITY,
            det_score=0.8,
            yaw_deg=10.0,
            single_frame_person_id=None,
            embedding_model_version="buffalo_l",
        )
    ]


class _SupplyingRunner:
    """Stands in for the pre-world stages (and, on the direct path, calls world tracking)."""

    def __init__(self, world_stage: WorldTrackingStage | None = None) -> None:
        self._world_stage = world_stage

    async def run(self, ctx: FrameContext) -> None:
        _supply(ctx)
        if self._world_stage is not None:
            await self._world_stage.run(ctx)


class _NoopRunner:
    async def run(self, ctx: FrameContext) -> None:
        return None


def _pipeline() -> tuple[FrameProcessingPipeline, WorldTrackingStage, FaceTemplateService]:
    templates = FaceTemplateService(
        FaceTemplatePolicy(mode="shadow"), _NoMatchClient(), expected_quality_version="heuristic-v1"
    )
    world_stage = WorldTrackingStage(
        tracker=WorldTracker(
            ph_repo=InMemoryPHRepository(), obs_repo=InMemoryWorldObservationRepository()
        ),
        live_config=LiveConfigHolder(CameraRoomMap(), RoomPolygonMap()),
        cc_assertion_mode="off",
        face_templates=templates,
    )
    pipeline = FrameProcessingPipeline(PipelineConfig(signals_enabled=False))
    pipeline._transport = AsyncMock()
    pipeline._detector = object()  # non-None so _process_frame runs the stage runner
    pipeline._world_tracking_stage = world_stage  # type: ignore[assignment]
    return pipeline, world_stage, templates


def _buffered(templates: FaceTemplateService) -> int:
    return sum(templates.buffer.size(ph_id) for ph_id in templates.buffer.ph_ids())


@pytest.mark.asyncio
async def test_direct_path_feeds_the_buffer() -> None:
    pipeline, world_stage, templates = _pipeline()
    pipeline._stage_runner = _SupplyingRunner(world_stage)  # type: ignore[assignment]

    await pipeline._process_frame(_frame("cam-a", 1))

    assert _buffered(templates) == 1


@pytest.mark.asyncio
async def test_per_camera_batch_path_feeds_the_buffer() -> None:
    pipeline, world_stage, templates = _pipeline()
    pipeline._stage_runner = _SupplyingRunner(world_stage)  # type: ignore[assignment]

    await pipeline._handle_batch("cam-a", [_frame("cam-a", 1), _frame("cam-a", 2)])

    assert _buffered(templates) == 2


@pytest.mark.asyncio
async def test_cross_camera_batch_path_feeds_the_buffer() -> None:
    pipeline, _, templates = _pipeline()
    pipeline._pre_world_runner = _SupplyingRunner()  # type: ignore[assignment]
    pipeline._post_world_runner = _NoopRunner()  # type: ignore[assignment]
    contexts = [pipeline._init_context(f) for f in (_frame("cam-a", 1), _frame("cam-b", 1))]

    await pipeline._process_cross_camera_post_detect_batch(contexts)

    assert _buffered(templates) == 2

"""FaceIdentityStage records one measurement row and one gate outcome per face.

Recording is observation only: the anchors the stage produces must be exactly the anchors it
produced before measurement existed, and a failing stats repository must not change them.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
import numpy.typing as npt
import pytest
from prometheus_client import REGISTRY

from app.domain import (
    BoundingBox,
    Detection,
    FaceGateOutcome,
    FaceObservationStat,
    FloorPoint,
)
from app.inference.face_id_client import (
    CropQueryResult,
    CropQueryStatus,
    FaceIdentificationClient,
    FaceResult,
)
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.face_identity import FaceIdentityStage
from app.pipeline.types import FaceIdCameraConfig
from app.storage.face_stats import InMemoryFaceObservationStatsRepository
from app.transport.redis_streams import FrameReady

_CAMERA = "cam-gate"
_CAPTURE_TIME = datetime(2026, 10, 20, 9, 30, tzinfo=UTC)


def _face(state: str, similarity: float, best_candidate_id: str | None) -> FaceResult:
    return FaceResult(
        person_id=best_candidate_id if state == "recognized" and best_candidate_id else "unknown",
        name="n",
        confidence=similarity,
        bbox_normalized=[0.1, 0.1, 0.2, 0.2],
        recognition_state=state,
        best_candidate_id=best_candidate_id,
        raw_similarity=similarity,
        similarity=similarity,
        yaw_deg=10.0,
        pitch_deg=-20.0,
        det_score=0.9,
        face_width_px=28.0,
        face_height_px=34.0,
        interocular_px=13.0,
    )


_FACES = (
    _face("recognized", 0.55, "resident-alpha"),
    _face("recognized", 0.65, "resident-beta"),
    _face("candidate", 0.30, None),
    _face("unrecognized", 0.10, "resident-alpha"),
)


class _ScriptedClient(FaceIdentificationClient):
    """Returns all scripted faces for the single crop, like a real service response."""

    def __init__(self) -> None:
        super().__init__("http://unused.test")

    async def identify_crops(
        self,
        crops: list[npt.NDArray[np.uint8]],
        crop_bboxes_norm: list[tuple[float, float, float, float]],
    ) -> list[CropQueryResult]:
        return [
            CropQueryResult(
                crop_index=0,
                status=CropQueryStatus.FACE_FOUND,
                crop_width_px=90,
                crop_height_px=210,
                faces=_FACES,
            )
        ]


class _FailingStatsRepo:
    async def append_many(self, rows: Sequence[FaceObservationStat]) -> None:
        raise ConnectionError("database unavailable")

    async def list_between(
        self, start: datetime, end: datetime, camera_id: str | None = None
    ) -> list[FaceObservationStat]:
        return []


def _ctx() -> FrameContext:
    frame = FrameReady(
        camera_id=_CAMERA,
        minio_key="k",
        width=640,
        height=480,
        frame_index=7,
        capture_time_unix_ns=int(_CAPTURE_TIME.timestamp() * 1e9),
    )
    ctx = FrameContext(
        frame=frame,
        event_time=datetime.now(UTC),
        capture_time=_CAPTURE_TIME,
        effective_width=640,
        effective_height=480,
    )
    ctx.domain_detections = [
        Detection(
            detection_id="det-1",
            camera_id=_CAMERA,
            bbox=BoundingBox(x_min=10, y_min=20, x_max=100, y_max=230),
            embedding=[],
            capture_time=_CAPTURE_TIME,
            event_time=_CAPTURE_TIME,
            confidence=0.9,
            floor_point=FloorPoint(x_mm=1000, y_mm=2000, calibrated=True),
        )
    ]
    ctx.crops = [np.zeros((210, 90, 3), dtype=np.uint8)]
    return ctx


def _stage(repo: object) -> FaceIdentityStage:
    return FaceIdentityStage(
        face_stats_repo=repo,  # type: ignore[arg-type]
        face_id_client=_ScriptedClient(),
        face_id_min_confidence=0.6,
        face_id_camera_configs={_CAMERA: FaceIdCameraConfig(enabled=True)},
    )


def _anchor_summary(ctx: FrameContext) -> list[tuple[str, str, float, str]]:
    return [
        (a.person_id, a.recognition_state, a.confidence, a.detection_id) for a in ctx.face_anchors
    ]


# What the stage emitted before measurement existed: the 0.55 recognized face is below the
# 0.6 floor, the candidate has no nearest identity, so two anchors remain.
_EXPECTED_ANCHORS = [
    ("resident-beta", "recognized", 0.65, "det-1"),
    ("unknown", "unrecognized", 0.9, "det-1"),
]


def _sample(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


@pytest.mark.asyncio
async def test_outcomes_recorded_per_face_and_anchors_unchanged() -> None:
    repo = InMemoryFaceObservationStatsRepository()
    dropped_before = _sample(
        "cts_face_gate_outcomes_total", camera_id=_CAMERA, outcome="dropped_min_confidence"
    )
    found_before = _sample("cts_face_queries_total", camera_id=_CAMERA, result="face_found")
    ctx = _ctx()

    await _stage(repo).run(ctx)

    assert _anchor_summary(ctx) == _EXPECTED_ANCHORS
    rows = await repo.list_between(_CAPTURE_TIME, _CAPTURE_TIME.replace(minute=31))
    assert [r.gate_outcome for r in rows] == [
        FaceGateOutcome.DROPPED_MIN_CONFIDENCE,
        FaceGateOutcome.EMITTED,
        FaceGateOutcome.DROPPED_NO_CANDIDATE,
        FaceGateOutcome.EMITTED,
    ]
    first = rows[0]
    assert first.observed_at == _CAPTURE_TIME
    assert (first.camera_id, first.detection_id) == (_CAMERA, "det-1")
    assert (first.crop_width_px, first.crop_height_px) == (90, 210)
    assert (first.interocular_px, first.raw_similarity) == (13.0, 0.55)
    assert first.best_candidate_id == "resident-alpha"
    assert (
        _sample("cts_face_gate_outcomes_total", camera_id=_CAMERA, outcome="dropped_min_confidence")
        - dropped_before
        == 1
    )
    assert (
        _sample("cts_face_queries_total", camera_id=_CAMERA, result="face_found") - (found_before)
        == 1
    )


@pytest.mark.asyncio
async def test_failing_stats_repository_does_not_change_anchors() -> None:
    failures_before = _sample("cts_face_stats_write_failures_total")
    ctx = _ctx()

    await _stage(_FailingStatsRepo()).run(ctx)

    assert _anchor_summary(ctx) == _EXPECTED_ANCHORS
    assert _sample("cts_face_stats_write_failures_total") - failures_before == 1

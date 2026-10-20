"""FaceIdentityStage records one measurement row and one gate outcome per face.

Each detection's crop holds one face inside that person's head region, so every
``_gate_outcome`` branch is exercised once per crop. Multi-face crops (attribution) are covered
in ``test_face_identity_two_faces_in_crop.py``. Recording is observation only: a failing stats
repository must not change the anchors.
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


_FRAME_W_PX = 640
_FRAME_H_PX = 480
# Four people side by side, 90 x 210 px each, no overlap (so no crossing bypass is involved).
_BOXES = [
    BoundingBox(x_min=10 + 120 * i, y_min=20, x_max=100 + 120 * i, y_max=230) for i in range(4)
]


def _face(index: int, state: str, similarity: float, best_candidate_id: str | None) -> FaceResult:
    # Face centred horizontally in detection *index*'s box, 30 px below its top: in its head region.
    cx = (_BOXES[index].x_min + _BOXES[index].x_max) / 2.0
    cy = _BOXES[index].y_min + 30.0
    return FaceResult(
        person_id=best_candidate_id if state == "recognized" and best_candidate_id else "unknown",
        name="n",
        confidence=similarity,
        bbox_normalized=[
            (cx - 14) / _FRAME_W_PX,
            (cy - 17) / _FRAME_H_PX,
            (cx + 14) / _FRAME_W_PX,
            (cy + 17) / _FRAME_H_PX,
        ],
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
    _face(0, "recognized", 0.55, "resident-alpha"),
    _face(1, "recognized", 0.65, "resident-beta"),
    _face(2, "candidate", 0.30, None),
    _face(3, "unrecognized", 0.10, "resident-alpha"),
)


class _ScriptedClient(FaceIdentificationClient):
    """Returns the scripted face of each crop, like a real service response."""

    def __init__(self) -> None:
        super().__init__("http://unused.test")

    async def identify_crops(
        self,
        crops: list[npt.NDArray[np.uint8]],
        crop_bboxes_norm: list[tuple[float, float, float, float]],
    ) -> list[CropQueryResult]:
        return [
            CropQueryResult(
                crop_index=index,
                status=CropQueryStatus.FACE_FOUND,
                crop_width_px=int(crop.shape[1]),
                crop_height_px=int(crop.shape[0]),
                faces=(_FACES[index],),
            )
            for index, crop in enumerate(crops)
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
        effective_width=_FRAME_W_PX,
        effective_height=_FRAME_H_PX,
        image=np.zeros((_FRAME_H_PX, _FRAME_W_PX, 3), dtype=np.uint8),
    )
    ctx.domain_detections = [
        Detection(
            detection_id=f"det-{index + 1}",
            camera_id=_CAMERA,
            bbox=box,
            embedding=[],
            capture_time=_CAPTURE_TIME,
            event_time=_CAPTURE_TIME,
            confidence=0.9,
            floor_point=FloorPoint(x_mm=1000 * (index + 1), y_mm=2000, calibrated=True),
        )
        for index, box in enumerate(_BOXES)
    ]
    ctx.crops = [np.zeros((210, 90, 3), dtype=np.uint8) for _ in _BOXES]
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


# The 0.55 recognized face is below the 0.6 floor and the candidate has no nearest identity,
# so two anchors remain, each on its own detection.
_EXPECTED_ANCHORS = [
    ("resident-beta", "recognized", 0.65, "det-2"),
    ("unknown", "unrecognized", 0.9, "det-4"),
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
    assigned_before = _sample("cts_face_attribution_total", camera_id=_CAMERA, result="assigned")
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
    # The crop sent is the person box padded 10% per side and 15% upward, clamped at the top.
    assert (first.crop_width_px, first.crop_height_px) == (108, 230)
    assert (first.interocular_px, first.raw_similarity) == (13.0, 0.55)
    assert first.best_candidate_id == "resident-alpha"
    assert (
        _sample("cts_face_gate_outcomes_total", camera_id=_CAMERA, outcome="dropped_min_confidence")
        - dropped_before
        == 1
    )
    assert (
        _sample("cts_face_queries_total", camera_id=_CAMERA, result="face_found") - (found_before)
        == 4
    )
    assert (
        _sample("cts_face_attribution_total", camera_id=_CAMERA, result="assigned")
        - assigned_before
        == 4
    )


@pytest.mark.asyncio
async def test_failing_stats_repository_does_not_change_anchors() -> None:
    failures_before = _sample("cts_face_stats_write_failures_total")
    ctx = _ctx()

    await _stage(_FailingStatsRepo()).run(ctx)

    assert _anchor_summary(ctx) == _EXPECTED_ANCHORS
    assert _sample("cts_face_stats_write_failures_total") - failures_before == 1

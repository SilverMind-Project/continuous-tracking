"""FaceIdentityStage records the service's face quality on anchors and measurement rows.

The stage never applies a face-quality policy: ``FaceAnchor.quality`` (the evidence-path weight)
stays 1.0 for a native anchor whatever the reading, and an older service yields no reading.
Whether the reading weighs evidence is ``resolver.face_quality_mode``
(``tests/unit/tracking/test_face_quality_mode.py``).
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import numpy.typing as npt
import pytest

from app.domain import BoundingBox, Detection, FloorPoint
from app.inference.face_id_client import (
    CropQueryResult,
    CropQueryStatus,
    FaceIdentificationClient,
    FaceQualityReading,
    FaceResult,
)
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.face_identity import FaceIdentityStage
from app.pipeline.types import FaceIdCameraConfig
from app.storage.face_stats import InMemoryFaceObservationStatsRepository
from app.transport.redis_streams import FrameReady

_CAMERA = "cam-quality"
_T = datetime(2026, 10, 20, 13, 0, tzinfo=UTC)
_W_PX, _H_PX = 640, 480
_PERSON = BoundingBox(x_min=100, y_min=60, x_max=200, y_max=360)
_READING = FaceQualityReading(
    score=0.37, resolution=0.3, sharpness=0.5, detection=0.8, pitch=1.0, version="heuristic-v1"
)


def _face(state: str, quality: FaceQualityReading | None) -> FaceResult:
    cx, cy = 150.0, 90.0  # inside the person's head region
    return FaceResult(
        person_id="resident-alpha" if state == "recognized" else "unknown",
        name="n",
        bbox_normalized=[
            (cx - 12) / _W_PX,
            (cy - 15) / _H_PX,
            (cx + 12) / _W_PX,
            (cy + 15) / _H_PX,
        ],
        recognition_state=state,
        best_candidate_id="resident-alpha",
        raw_similarity=0.8 if state != "unrecognized" else 0.1,
        det_score=0.9,
        face_width_px=24.0,
        face_height_px=30.0,
        interocular_px=11.0,
        quality=quality,
    )


class _OneFaceClient(FaceIdentificationClient):
    def __init__(self, face: FaceResult) -> None:
        super().__init__("http://unused.test")
        self._face = face

    async def identify_crops(
        self,
        crops: list[npt.NDArray[np.uint8]],
        crop_bboxes_norm: list[tuple[float, float, float, float]],
    ) -> list[CropQueryResult]:
        [crop] = crops
        return [
            CropQueryResult(
                crop_index=0,
                status=CropQueryStatus.FACE_FOUND,
                crop_width_px=int(crop.shape[1]),
                crop_height_px=int(crop.shape[0]),
                faces=(self._face,),
            )
        ]


async def _run(face: FaceResult) -> tuple[FrameContext, InMemoryFaceObservationStatsRepository]:
    repo = InMemoryFaceObservationStatsRepository()
    stage = FaceIdentityStage(
        face_stats_repo=repo,
        face_id_client=_OneFaceClient(face),
        face_id_min_raw_similarity=0.6,
        face_id_camera_configs={_CAMERA: FaceIdCameraConfig(enabled=True)},
    )
    frame = FrameReady(
        camera_id=_CAMERA,
        minio_key="k",
        width=_W_PX,
        height=_H_PX,
        frame_index=1,
        capture_time_unix_ns=int(_T.timestamp() * 1e9),
    )
    ctx = FrameContext(
        frame=frame,
        event_time=_T,
        capture_time=_T,
        effective_width=_W_PX,
        effective_height=_H_PX,
        image=np.zeros((_H_PX, _W_PX, 3), dtype=np.uint8),
    )
    ctx.domain_detections = [
        Detection(
            detection_id="det-a",
            camera_id=_CAMERA,
            bbox=_PERSON,
            embedding=[],
            capture_time=_T,
            event_time=_T,
            confidence=0.9,
            floor_point=FloorPoint(x_mm=1000, y_mm=2000, calibrated=True),
        )
    ]
    ctx.crops = [np.zeros((300, 100, 3), dtype=np.uint8)]
    await stage.run(ctx)
    return ctx, repo


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["recognized", "candidate", "unrecognized"])
async def test_anchor_carries_the_reading_and_keeps_its_evidence_quality(state: str) -> None:
    ctx, repo = await _run(_face(state, _READING))

    [anchor] = ctx.face_anchors
    assert (anchor.face_quality, anchor.face_quality_version) == (0.37, "heuristic-v1")
    assert anchor.quality == 1.0
    [row] = await repo.list_between(_T, _T.replace(minute=1))
    assert (row.quality_score, row.quality_version) == (0.37, "heuristic-v1")


@pytest.mark.asyncio
async def test_older_service_without_quality_records_none() -> None:
    ctx, repo = await _run(_face("recognized", None))

    [anchor] = ctx.face_anchors
    assert (anchor.face_quality, anchor.face_quality_version) == (None, "")
    assert anchor.quality == 1.0
    [row] = await repo.list_between(_T, _T.replace(minute=1))
    assert (row.quality_score, row.quality_version) == (None, None)

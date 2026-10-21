"""A person crop that contains two faces yields at most one anchor, never a neighbour's face.

Every face still gets a measurement row: the attributed face goes through the normal gate, a face
outside the head region is ``dropped_not_in_head_region``, and both faces of an ambiguous crop are
``dropped_ambiguous``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import numpy.typing as npt
import pytest
from prometheus_client import REGISTRY

from app.domain import BoundingBox, Detection, FaceGateOutcome, FloorPoint
from app.inference.face_id_client import (
    CaptureContext,
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

_CAMERA = "cam-pair"
_T = datetime(2026, 10, 20, 11, 0, tzinfo=UTC)
_W_PX, _H_PX = 640, 480
# Person box 100 px wide, 300 px tall; head region x in [90, 210], y in [55 (padded), 165].
_PERSON = BoundingBox(x_min=100, y_min=60, x_max=200, y_max=360)


def _face(person_id: str, cx: float, cy: float, similarity: float = 0.8) -> FaceResult:
    return FaceResult(
        person_id=person_id,
        name=person_id,
        bbox_normalized=[
            (cx - 12) / _W_PX,
            (cy - 15) / _H_PX,
            (cx + 12) / _W_PX,
            (cy + 15) / _H_PX,
        ],
        recognition_state="recognized",
        best_candidate_id=person_id,
        raw_similarity=similarity,
        det_score=0.9,
        face_width_px=24.0,
        face_height_px=30.0,
        interocular_px=11.0,
    )


class _OneCropClient(FaceIdentificationClient):
    def __init__(self, faces: tuple[FaceResult, ...]) -> None:
        super().__init__("http://unused.test")
        self._faces = faces

    async def identify_crops(
        self,
        crops: list[npt.NDArray[np.uint8]],
        crop_bboxes_norm: list[tuple[float, float, float, float]],
        context: CaptureContext,
    ) -> list[CropQueryResult]:
        [crop] = crops
        return [
            CropQueryResult(
                crop_index=0,
                status=CropQueryStatus.FACE_FOUND,
                crop_width_px=int(crop.shape[1]),
                crop_height_px=int(crop.shape[0]),
                faces=self._faces,
            )
        ]


def _ctx() -> FrameContext:
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
    return ctx


async def _run(faces: tuple[FaceResult, ...]) -> tuple[FrameContext, list[FaceGateOutcome]]:
    repo = InMemoryFaceObservationStatsRepository()
    stage = FaceIdentityStage(
        face_stats_repo=repo,
        face_id_client=_OneCropClient(faces),
        face_id_min_raw_similarity=0.6,
        face_id_camera_configs={_CAMERA: FaceIdCameraConfig(enabled=True)},
    )
    ctx = _ctx()
    await stage.run(ctx)
    rows = await repo.list_between(_T, _T.replace(minute=1))
    return ctx, [r.gate_outcome for r in rows]


def _attribution_count(result: str) -> float:
    return (
        REGISTRY.get_sample_value(
            "cts_face_attribution_total", {"camera_id": _CAMERA, "result": result}
        )
        or 0.0
    )


@pytest.mark.asyncio
async def test_neighbours_face_at_crop_side_is_not_attributed() -> None:
    # The neighbour's face is at x=230, beyond the 10 px side margin of this person's box.
    ctx, outcomes = await _run((_face("bob", 230, 100), _face("alice", 150, 90)))

    assert [(a.person_id, a.detection_id) for a in ctx.face_anchors] == [("alice", "det-a")]
    assert outcomes == [FaceGateOutcome.DROPPED_NOT_IN_HEAD_REGION, FaceGateOutcome.EMITTED]


@pytest.mark.asyncio
async def test_two_faces_in_head_region_give_no_anchor() -> None:
    before = _attribution_count("ambiguous")

    ctx, outcomes = await _run((_face("alice", 130, 90), _face("bob", 180, 100)))

    assert ctx.face_anchors == []
    assert outcomes == [FaceGateOutcome.DROPPED_AMBIGUOUS, FaceGateOutcome.DROPPED_AMBIGUOUS]
    assert _attribution_count("ambiguous") - before == 1


@pytest.mark.asyncio
async def test_face_only_at_chest_height_is_not_attributed() -> None:
    before = _attribution_count("no_face_in_head_region")

    ctx, outcomes = await _run((_face("bob", 150, 250),))

    assert ctx.face_anchors == []
    assert outcomes == [FaceGateOutcome.DROPPED_NOT_IN_HEAD_REGION]
    assert _attribution_count("no_face_in_head_region") - before == 1


class _CapturingClient(FaceIdentificationClient):
    def __init__(self) -> None:
        super().__init__("http://unused.test")
        self.crops: list[npt.NDArray[np.uint8]] = []
        self.boxes: list[tuple[float, float, float, float]] = []

    async def identify_crops(
        self,
        crops: list[npt.NDArray[np.uint8]],
        crop_bboxes_norm: list[tuple[float, float, float, float]],
        context: CaptureContext,
    ) -> list[CropQueryResult]:
        self.crops.extend(crops)
        self.boxes.extend(crop_bboxes_norm)
        return [
            CropQueryResult(
                crop_index=i,
                status=CropQueryStatus.NO_FACE,
                crop_width_px=int(c.shape[1]),
                crop_height_px=int(c.shape[0]),
            )
            for i, c in enumerate(crops)
        ]


@pytest.mark.asyncio
async def test_face_path_sends_padded_crop_from_frame_with_matching_coordinates() -> None:
    client = _CapturingClient()
    stage = FaceIdentityStage(
        face_stats_repo=InMemoryFaceObservationStatsRepository(),
        face_id_client=client,
        face_id_camera_configs={_CAMERA: FaceIdCameraConfig(enabled=True)},
    )
    ctx = _ctx()
    ctx.image = np.zeros((_H_PX, _W_PX, 3), dtype=np.uint8)
    ctx.image[15:60, 90:210] = 200  # the band above the person box, reachable only by padding

    await stage.run(ctx)

    # Padded 10 px per side and 45 px upward (15% of 300); never downward.
    [crop] = client.crops
    assert crop.shape[:2] == (345, 120)
    assert np.all(crop[:45] == 200)
    assert client.boxes == [(90 / _W_PX, 15 / _H_PX, 210 / _W_PX, 360 / _H_PX)]
    # ReID and pose keep the tight crop.
    assert ctx.crops[0].shape[:2] == (300, 100)

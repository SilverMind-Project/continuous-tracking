"""Which faces FaceIdentityStage hands to the per-PH face template buffer.

Every face attributed to the person may join, whatever the evidence policy (sub-floor and
candidate faces are exactly a template's weak members); faces of other people never do; a face
without an embedding or with a quality score from another scorer fails closed.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import numpy.typing as npt
import pytest

from app.domain import BoundingBox, Detection, FaceEvidencePolicy, FloorPoint
from app.inference.face_id_client import (
    CaptureContext,
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

_CAMERA = "cam-t"
_T = datetime(2026, 11, 12, 9, 0, tzinfo=UTC)
_W, _H = 800, 480
_BOXES = [
    BoundingBox(x_min=10 + 130 * i, y_min=20, x_max=100 + 130 * i, y_max=230) for i in range(6)
]


def _quality(version: str = "heuristic-v1") -> FaceQualityReading:
    return FaceQualityReading(
        score=0.4, resolution=0.3, sharpness=0.5, detection=0.9, pitch=1.0, version=version
    )


def _face(
    index: int, state: str, similarity: float, *, head: bool = True, **changes: object
) -> FaceResult:
    box = _BOXES[index]
    cx = (box.x_min + box.x_max) / 2.0
    cy = box.y_min + (30.0 if head else 170.0)  # outside the head region: someone else's face
    v = np.random.RandomState(index).randn(512).astype(np.float32)
    fields: dict[str, object] = {
        "person_id": "alice" if state == "recognized" else "unknown",
        "name": "n",
        "bbox_normalized": [(cx - 14) / _W, (cy - 17) / _H, (cx + 14) / _W, (cy + 17) / _H],
        "recognition_state": state,
        "best_candidate_id": "alice",
        "raw_similarity": similarity,
        "det_score": 0.9,
        "quality": _quality(),
        "embedding": v / np.linalg.norm(v),
        "embedding_model_version": "buffalo_l",
    }
    fields.update(changes)
    return FaceResult(**fields)  # type: ignore[arg-type]


_FACES = (
    _face(0, "recognized", 0.55),  # below the legacy floor: dropped as evidence, kept for templates
    _face(1, "candidate", 0.30),
    _face(2, "unrecognized", 0.10),
    _face(3, "recognized", 0.80, head=False),  # not this person's face
    _face(4, "recognized", 0.80, quality=_quality("heuristic-v2")),  # another scorer
    _face(5, "recognized", 0.80, embedding=None),  # the service sent no embedding
)


class _Client(FaceIdentificationClient):
    def __init__(self) -> None:
        super().__init__("http://unused.test")
        self.asked: list[bool] = []

    async def identify_crops(
        self,
        crops: list[npt.NDArray[np.uint8]],
        crop_bboxes_norm: list[tuple[float, float, float, float]],
        context: CaptureContext,
        *,
        return_embedding: bool = False,
    ) -> list[CropQueryResult]:
        self.asked.append(return_embedding)
        return [
            CropQueryResult(
                crop_index=i,
                status=CropQueryStatus.FACE_FOUND,
                crop_width_px=int(c.shape[1]),
                crop_height_px=int(c.shape[0]),
                faces=(_FACES[i],),
            )
            for i, c in enumerate(crops)
        ]


def _ctx() -> FrameContext:
    frame = FrameReady(
        camera_id=_CAMERA,
        minio_key="k",
        width=_W,
        height=_H,
        frame_index=1,
        capture_time_unix_ns=int(_T.timestamp() * 1e9),
    )
    ctx = FrameContext(
        frame=frame,
        event_time=_T,
        capture_time=_T,
        effective_width=_W,
        effective_height=_H,
        image=np.zeros((_H, _W, 3), dtype=np.uint8),
    )
    ctx.domain_detections = [
        Detection(
            detection_id=f"det-{i}",
            camera_id=_CAMERA,
            bbox=box,
            embedding=[],
            capture_time=_T,
            event_time=_T,
            confidence=0.9,
            floor_point=FloorPoint(x_mm=1000 * i, y_mm=0, calibrated=True),
        )
        for i, box in enumerate(_BOXES)
    ]
    ctx.crops = [np.zeros((210, 90, 3), dtype=np.uint8) for _ in _BOXES]
    return ctx


async def _run(
    collect: bool, policy: FaceEvidencePolicy = "legacy"
) -> tuple[FrameContext, _Client]:
    client = _Client()
    stage = FaceIdentityStage(
        face_stats_repo=InMemoryFaceObservationStatsRepository(),
        face_id_client=client,
        face_id_min_raw_similarity=0.6,
        face_id_camera_configs={_CAMERA: FaceIdCameraConfig(enabled=True)},
        evidence_policy=policy,
        collect_template_faces=collect,
    )
    ctx = _ctx()
    await stage.run(ctx)
    return ctx, client


@pytest.mark.parametrize("policy", ["legacy", "shadow", "calibrated"])
async def test_attributed_faces_join_whatever_the_evidence_policy(
    policy: FaceEvidencePolicy,
) -> None:
    ctx, client = await _run(collect=True, policy=policy)

    assert client.asked == [True]
    assert [f.detection_id for f in ctx.face_template_faces] == ["det-0", "det-1", "det-2"]
    first = ctx.face_template_faces[0]
    assert (first.single_frame_person_id, first.camera_id, first.captured_at) == (
        "alice",
        _CAMERA,
        _T,
    )
    assert [f.single_frame_person_id for f in ctx.face_template_faces[1:]] == [None, None]


async def test_nothing_is_collected_or_requested_while_templates_are_off() -> None:
    ctx, client = await _run(collect=False)
    assert client.asked == [False]
    assert ctx.face_template_faces == []

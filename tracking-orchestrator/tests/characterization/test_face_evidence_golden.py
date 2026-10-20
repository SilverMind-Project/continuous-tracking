"""Golden characterization of what FaceIdentityStage turns face-service responses into.

Drives the stage with a scripted client over a table of responses (recognized at several raw
similarities, candidates with and without a nearest identity, an unrecognized face, calibrated
and uncalibrated results, a per-camera similarity floor, a neighbour's face and an ambiguous
crop) and freezes the anchors and the measurement rows it produces. The golden uses semantic key
names (``score`` for the anchor's weighed score, ``outcome`` for what the gate did), so a pure
vocabulary refactor changes the accessors below but must leave the golden file byte-identical.
The anchor's ``captured_at`` is wall-clock time and is not recorded.

Regenerate only for an intended behavior change:
``CTS_REGENERATE_GOLDEN=1 pytest tests/characterization/test_face_evidence_golden.py``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from app.domain import (
    BoundingBox,
    Detection,
    FaceAnchor,
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

_GOLDEN = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "characterization"
    / "face_anchors_golden.json"
)
_T = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
_W_PX, _H_PX = 640, 480
# Person box 100 x 300 px; its head region spans roughly x in [90, 210], y in [55, 165].
_PERSON = BoundingBox(x_min=100, y_min=60, x_max=200, y_max=360)
_HEAD = (150.0, 90.0)
_GLOBAL_FLOOR = 0.6

# Behavior names for the gate outcomes, independent of the enum member names.
_OUTCOME_BEHAVIOR = {
    FaceGateOutcome.EMITTED: "emitted",
    FaceGateOutcome.DROPPED_MIN_SIMILARITY: "below_similarity_floor",
    FaceGateOutcome.DROPPED_NO_CANDIDATE: "candidate_without_identity",
    FaceGateOutcome.DROPPED_NO_DETECTION: "no_detection",
    FaceGateOutcome.DROPPED_NOT_IN_HEAD_REGION: "outside_head_region",
    FaceGateOutcome.DROPPED_AMBIGUOUS: "ambiguous_crop",
}


def _face(
    state: str,
    similarity: float,
    best_candidate_id: str | None,
    centre: tuple[float, float] = _HEAD,
    calibrated: float | None = None,
    det_score: float = 0.82,
) -> FaceResult:
    """A face as person-identification-service returns it: all three score fields agree."""
    cx, cy = centre
    return FaceResult(
        person_id=best_candidate_id if state == "recognized" and best_candidate_id else "unknown",
        name="n",
        bbox_normalized=[
            (cx - 12) / _W_PX,
            (cy - 15) / _H_PX,
            (cx + 12) / _W_PX,
            (cy + 15) / _H_PX,
        ],
        recognition_state=state,
        best_candidate_id=best_candidate_id,
        raw_similarity=similarity,
        yaw_deg=12.0,
        pitch_deg=-8.0,
        det_score=det_score,
        calibrated_confidence=calibrated,
        calibration_status="ready" if calibrated is not None else "degraded_missing",
        face_width_px=24.0,
        face_height_px=30.0,
        interocular_px=11.0,
    )


@dataclass(frozen=True)
class _Case:
    name: str
    faces: tuple[FaceResult, ...]
    camera_floor: float | None = None
    status: CropQueryStatus = CropQueryStatus.FACE_FOUND


_CASES = (
    *(
        _Case(f"recognized_{s:.2f}", (_face("recognized", s, "resident-alpha"),))
        for s in (0.35, 0.45, 0.55, 0.65, 0.85)
    ),
    _Case(
        "recognized_0.85_calibrated",
        (_face("recognized", 0.85, "resident-alpha", calibrated=0.91),),
    ),
    _Case("recognized_0.55_camera_floor_0.5", (_face("recognized", 0.55, "resident-alpha"),), 0.5),
    _Case("recognized_0.65_camera_floor_0.7", (_face("recognized", 0.65, "resident-alpha"),), 0.7),
    _Case("candidate_0.30_with_identity", (_face("candidate", 0.30, "resident-beta"),)),
    _Case("candidate_0.30_without_identity", (_face("candidate", 0.30, None),)),
    _Case(
        "candidate_0.30_calibrated", (_face("candidate", 0.30, "resident-beta", calibrated=0.4),)
    ),
    _Case("unrecognized_det_0.9", (_face("unrecognized", 0.12, "resident-alpha", det_score=0.9),)),
    _Case(
        "neighbour_face_beside_own",
        (
            _face("recognized", 0.8, "resident-beta", centre=(230.0, 100.0)),
            _face("recognized", 0.7, "resident-alpha"),
        ),
    ),
    _Case(
        "ambiguous_two_heads",
        (
            _face("recognized", 0.8, "resident-alpha", centre=(130.0, 90.0)),
            _face("recognized", 0.75, "resident-beta", centre=(180.0, 100.0)),
        ),
    ),
    _Case(
        "chest_height_face_only",
        (_face("recognized", 0.8, "resident-beta", centre=(150.0, 250.0)),),
    ),
    _Case("no_face", (), status=CropQueryStatus.NO_FACE),
)


class _ScriptedClient(FaceIdentificationClient):
    def __init__(self, case: _Case) -> None:
        super().__init__("http://unused.test")
        self._case = case

    async def identify_crops(
        self,
        crops: list[npt.NDArray[np.uint8]],
        crop_bboxes_norm: list[tuple[float, float, float, float]],
    ) -> list[CropQueryResult]:
        [crop] = crops
        return [
            CropQueryResult(
                crop_index=0,
                status=self._case.status,
                crop_width_px=int(crop.shape[1]),
                crop_height_px=int(crop.shape[0]),
                faces=self._case.faces,
            )
        ]


def _ctx(camera_id: str) -> FrameContext:
    frame = FrameReady(
        camera_id=camera_id,
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
            camera_id=camera_id,
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


def _anchor_record(anchor: FaceAnchor) -> dict[str, Any]:
    return {
        "person_id": anchor.person_id,
        "score": anchor.match_score,
        "quality": anchor.quality,
        "tracklet_id": anchor.tracklet_id,
        "detection_id": anchor.detection_id,
        "camera_id": anchor.camera_id,
        "recognition_state": anchor.recognition_state,
        "similarity": anchor.similarity,
        "yaw_deg": anchor.yaw_deg,
        "calibrated_confidence": anchor.calibrated_confidence,
        "origin": anchor.origin,
    }


def _stat_record(row: FaceObservationStat) -> dict[str, Any]:
    return {
        "observed_at": row.observed_at.isoformat(),
        "camera_id": row.camera_id,
        "detection_id": row.detection_id,
        "crop_width_px": row.crop_width_px,
        "crop_height_px": row.crop_height_px,
        "face_width_px": row.face_width_px,
        "face_height_px": row.face_height_px,
        "interocular_px": row.interocular_px,
        "det_score": row.det_score,
        "yaw_deg": row.yaw_deg,
        "pitch_deg": row.pitch_deg,
        "raw_similarity": row.raw_similarity,
        "recognition_state": row.recognition_state,
        "best_candidate_id": row.best_candidate_id,
        "outcome": _OUTCOME_BEHAVIOR[row.gate_outcome],
    }


async def _run_case(case: _Case) -> dict[str, Any]:
    camera_id = f"cam-{case.name}"
    repo = InMemoryFaceObservationStatsRepository()
    stage = FaceIdentityStage(
        face_stats_repo=repo,
        face_id_client=_ScriptedClient(case),
        face_id_min_raw_similarity=_GLOBAL_FLOOR,
        face_id_camera_configs={
            camera_id: FaceIdCameraConfig(enabled=True, min_raw_similarity=case.camera_floor)
        },
    )
    ctx = _ctx(camera_id)
    await stage.run(ctx)
    rows = await repo.list_between(_T, _T.replace(minute=1))
    return {
        "anchors": [_anchor_record(a) for a in ctx.face_anchors],
        "stats": [_stat_record(r) for r in rows],
    }


@pytest.mark.asyncio
async def test_face_stage_output_matches_golden() -> None:
    record = {case.name: await _run_case(case) for case in _CASES}
    actual = json.dumps(record, indent=1, sort_keys=True)
    if os.environ.get("CTS_REGENERATE_GOLDEN") == "1":
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(actual + "\n")
    assert actual == _GOLDEN.read_text().rstrip("\n"), (
        "face-stage output changed; a refactor must not change it"
    )


def test_every_gate_outcome_has_a_behavior_name() -> None:
    assert set(_OUTCOME_BEHAVIOR) == set(FaceGateOutcome)

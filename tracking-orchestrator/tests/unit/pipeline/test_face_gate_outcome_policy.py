"""The face evidence policy decides which recognized faces FaceIdentityStage emits.

``_gate_outcome`` is the one per-face decision; the stage builds an anchor exactly for the
outcomes ``emitted`` (live) and ``emitted_shadow_only`` (shadow evaluation only), so the outcome
recorded in ``face_observation_stats`` and the anchors always agree. The stage-level tests drive
``FaceIdentityStage.run`` with a frame image and scripted ``CropQueryResult``s.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import numpy.typing as npt
import pytest

from app.domain import (
    BoundingBox,
    Detection,
    FaceAnchor,
    FaceEvidencePolicy,
    FaceGateOutcome,
    FloorPoint,
)
from app.inference.face_id_client import (
    CaptureContext,
    CropQueryResult,
    CropQueryStatus,
    FaceIdentificationClient,
    FaceResult,
)
from app.pipeline.face_attribution import FaceAttribution
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.face_identity import FaceIdentityStage, _gate_outcome
from app.pipeline.types import FaceIdCameraConfig
from app.storage.face_stats import InMemoryFaceObservationStatsRepository
from app.transport.redis_streams import FrameReady

_GLOBAL_FLOOR = 0.6
_POLICIES: tuple[FaceEvidencePolicy, ...] = ("legacy", "shadow", "calibrated")


def _result(
    state: str, similarity: float, best_candidate_id: str | None = "resident-alpha"
) -> FaceResult:
    return FaceResult(
        person_id=best_candidate_id if state == "recognized" and best_candidate_id else "unknown",
        name="n",
        bbox_normalized=[0.1, 0.1, 0.2, 0.2],
        recognition_state=state,
        best_candidate_id=best_candidate_id,
        raw_similarity=similarity,
        det_score=0.9,
    )


def _outcome(
    policy: FaceEvidencePolicy,
    face: FaceResult,
    camera_floor: float | None = None,
    attribution: FaceAttribution = FaceAttribution.ASSIGNED,
) -> FaceGateOutcome | None:
    return _gate_outcome(
        face,
        attribution,
        policy,
        global_min_raw_similarity=_GLOBAL_FLOOR,
        camera_min_raw_similarity=camera_floor,
    )


_E = FaceGateOutcome.EMITTED
_S = FaceGateOutcome.EMITTED_SHADOW_ONLY
_D = FaceGateOutcome.DROPPED_MIN_SIMILARITY


@pytest.mark.parametrize(
    ("policy", "similarity", "camera_floor", "expected"),
    [
        # No camera floor: legacy uses the global 0.6 floor, calibrated none, shadow marks the
        # difference.
        ("legacy", 0.55, None, _D),
        ("legacy", 0.65, None, _E),
        ("shadow", 0.55, None, _S),
        ("shadow", 0.65, None, _E),
        ("calibrated", 0.55, None, _E),
        ("calibrated", 0.65, None, _E),
        ("calibrated", 0.30, None, _E),
        # A camera floor replaces the global floor under legacy and is the only floor under
        # calibrated; shadow then has nothing to add (both policies agree).
        ("legacy", 0.55, 0.5, _E),
        ("legacy", 0.65, 0.7, _D),
        ("calibrated", 0.45, 0.5, _D),
        ("calibrated", 0.55, 0.5, _E),
        ("shadow", 0.45, 0.5, _D),
        ("shadow", 0.55, 0.5, _E),
        ("shadow", 0.65, 0.7, _D),
    ],
)
def test_recognized_face_floor_by_policy_and_camera_floor(
    policy: FaceEvidencePolicy,
    similarity: float,
    camera_floor: float | None,
    expected: FaceGateOutcome,
) -> None:
    assert _outcome(policy, _result("recognized", similarity), camera_floor) is expected


@pytest.mark.parametrize("policy", _POLICIES)
@pytest.mark.parametrize("camera_floor", [None, 0.5])
def test_candidates_and_unrecognized_are_unchanged_in_every_policy(
    policy: FaceEvidencePolicy, camera_floor: float | None
) -> None:
    assert _outcome(policy, _result("candidate", 0.30), camera_floor) is _E
    assert (
        _outcome(policy, _result("candidate", 0.30, best_candidate_id=None), camera_floor)
        is FaceGateOutcome.DROPPED_NO_CANDIDATE
    )
    assert _outcome(policy, _result("unrecognized", 0.10), camera_floor) is _E
    assert _outcome(policy, _result("bogus_state", 0.9), camera_floor) is None


@pytest.mark.parametrize("policy", _POLICIES)
def test_a_face_that_is_not_the_persons_never_becomes_evidence_in_any_policy(
    policy: FaceEvidencePolicy,
) -> None:
    for similarity in (0.55, 0.95):
        face = _result("recognized", similarity)
        assert (
            _outcome(policy, face, attribution=FaceAttribution.AMBIGUOUS)
            is FaceGateOutcome.DROPPED_AMBIGUOUS
        )
        assert (
            _outcome(policy, face, attribution=FaceAttribution.NO_FACE_IN_HEAD_REGION)
            is FaceGateOutcome.DROPPED_NOT_IN_HEAD_REGION
        )


# ---------------------------------------------------------------------------
# Stage level: stats rows and anchors agree in every policy
# ---------------------------------------------------------------------------

_CAMERA = "cam-policy"
_CAPTURE_TIME = datetime(2026, 11, 11, 9, 30, tzinfo=UTC)
_FRAME_W_PX = 640
_FRAME_H_PX = 480
# Four people side by side, no overlap, so no crossing bypass is involved.
_BOXES = [
    BoundingBox(x_min=10 + 120 * i, y_min=20, x_max=100 + 120 * i, y_max=230) for i in range(4)
]


def _face_in_box(
    index: int,
    state: str,
    similarity: float,
    best_candidate_id: str | None,
    calibrated_confidence: float | None = None,
) -> FaceResult:
    # Centred horizontally, 30 px below the box top: inside the person's head region.
    cx = (_BOXES[index].x_min + _BOXES[index].x_max) / 2.0
    cy = _BOXES[index].y_min + 30.0
    return FaceResult(
        person_id=best_candidate_id if state == "recognized" and best_candidate_id else "unknown",
        name="n",
        bbox_normalized=[
            (cx - 14) / _FRAME_W_PX,
            (cy - 17) / _FRAME_H_PX,
            (cx + 14) / _FRAME_W_PX,
            (cy + 17) / _FRAME_H_PX,
        ],
        recognition_state=state,
        best_candidate_id=best_candidate_id,
        raw_similarity=similarity,
        yaw_deg=10.0,
        det_score=0.9,
        face_width_px=28.0,
        face_height_px=34.0,
        interocular_px=13.0,
        calibrated_confidence=calibrated_confidence,
    )


_FACES = (
    _face_in_box(0, "recognized", 0.55, "resident-alpha"),
    _face_in_box(1, "recognized", 0.65, "resident-beta"),
    _face_in_box(2, "candidate", 0.30, None),
    _face_in_box(3, "unrecognized", 0.10, "resident-alpha"),
)


class _ScriptedClient(FaceIdentificationClient):
    """Returns the scripted face of each crop, like a real service response."""

    def __init__(self) -> None:
        super().__init__("http://unused.test")

    async def identify_crops(
        self,
        crops: list[npt.NDArray[np.uint8]],
        crop_bboxes_norm: list[tuple[float, float, float, float]],
        context: CaptureContext,
        *,
        return_embedding: bool = False,
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


def _ctx() -> FrameContext:
    frame = FrameReady(
        camera_id=_CAMERA,
        minio_key="k",
        width=_FRAME_W_PX,
        height=_FRAME_H_PX,
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


async def _run(
    policy: FaceEvidencePolicy, camera_floor: float | None = None
) -> tuple[FrameContext, dict[str, FaceGateOutcome]]:
    repo = InMemoryFaceObservationStatsRepository()
    stage = FaceIdentityStage(
        face_stats_repo=repo,
        face_id_client=_ScriptedClient(),
        face_id_min_raw_similarity=_GLOBAL_FLOOR,
        face_id_camera_configs={
            _CAMERA: FaceIdCameraConfig(enabled=True, min_raw_similarity=camera_floor)
        },
        evidence_policy=policy,
    )
    ctx = _ctx()
    await stage.run(ctx)
    rows = await repo.list_between(_CAPTURE_TIME, _CAPTURE_TIME.replace(minute=31))
    return ctx, {row.detection_id: row.gate_outcome for row in rows}


def _summary(anchors: list[FaceAnchor]) -> list[tuple[str, str, str, bool]]:
    return [(a.detection_id, a.person_id, a.recognition_state, a.shadow_only) for a in anchors]


_LIVE_BASE = [
    ("det-2", "resident-beta", "recognized", False),
    ("det-4", "unknown", "unrecognized", False),
]


@pytest.mark.parametrize(
    ("policy", "camera_floor", "live", "shadow", "det1_outcome"),
    [
        ("legacy", None, _LIVE_BASE, [], _D),
        ("shadow", None, _LIVE_BASE, [("det-1", "resident-alpha", "recognized", True)], _S),
        (
            "calibrated",
            None,
            [("det-1", "resident-alpha", "recognized", False), *_LIVE_BASE],
            [],
            _E,
        ),
        # An operator camera floor still drops the 0.55 face under calibrated.
        ("calibrated", 0.6, _LIVE_BASE, [], _D),
    ],
)
async def test_stage_anchors_and_recorded_outcomes_agree(
    policy: FaceEvidencePolicy,
    camera_floor: float | None,
    live: list[tuple[str, str, str, bool]],
    shadow: list[tuple[str, str, str, bool]],
    det1_outcome: FaceGateOutcome,
) -> None:
    ctx, outcomes = await _run(policy, camera_floor)

    assert _summary(ctx.face_anchors) == live
    assert _summary(ctx.shadow_face_anchors) == shadow
    assert outcomes == {
        "det-1": det1_outcome,
        "det-2": _E,
        "det-3": FaceGateOutcome.DROPPED_NO_CANDIDATE,
        "det-4": _E,
    }
    # The agreement invariant: emitted rows are exactly the live anchors, shadow-only rows are
    # exactly the shadow anchors.
    assert {d for d, o in outcomes.items() if o is _E} == {a.detection_id for a in ctx.face_anchors}
    assert {d for d, o in outcomes.items() if o is _S} == {
        a.detection_id for a in ctx.shadow_face_anchors
    }


async def test_shadow_policy_leaves_every_live_output_identical_to_legacy() -> None:
    legacy, _ = await _run("legacy")
    shadow, _ = await _run("shadow")

    # Live anchors and the evidence records built from them (read by the resolver's duplicate
    # guard, the gallery-candidate stage, and ReID cadence) are byte-for-byte the legacy ones;
    # the shadow face exists only in its own list and has no evidence record.
    def comparable(ctx: FrameContext) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        anchors = [a.__dict__ | {"captured_at": None} for a in ctx.face_anchors]
        evidence = [e.__dict__ | {"captured_at": None} for e in ctx._face_evidence]
        return anchors, evidence

    assert comparable(shadow) == comparable(legacy)
    assert legacy.shadow_face_anchors == []
    assert [a.detection_id for a in shadow.shadow_face_anchors] == ["det-1"]
    assert "det-1" not in {e.detection_id for e in shadow._face_evidence}

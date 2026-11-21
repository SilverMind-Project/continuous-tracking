"""Unit tests for AppearanceSampleRecorder and AppearanceSampleRecorderStage.

Verifies governance boundaries (only direct-face authority, trajectory-attested wearables,
and verified open-set negatives are captured; posterior-only labels, room-grade wearables,
held tags, and face conflicts are rejected), capture-time deduplication, daily negative caps,
and stage wiring across all three pipeline execution paths.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from prometheus_client import CollectorRegistry

from app.domain import (
    AppearanceLabelSource,
    BoundingBox,
    Detection,
    FaceAnchor,
    FloorPoint,
    IdentityDecision,
    OrientationBin,
    PosteriorDist,
    WearableAttestationGrade,
    WearableEvidence,
    WearableEvidenceClass,
)
from app.observability import metrics as metrics_mod
from app.observability.metrics import build_metrics
from app.pipeline.frame_pipeline import (
    FrameProcessingPipeline,
    PipelineConfig,
    PipelineDependencies,
)
from app.services.appearance_sample_recorder import (
    AppearanceSampleRecorder,
    AppearanceSampleRecorderConfig,
)
from app.storage.appearance_samples import InMemoryAppearanceSampleRepository
from app.storage.wearable_teaching_holds import InMemoryWearableTeachingHoldRepository
from app.tracking.identity.household_classifier import HouseholdClassifierConfig
from app.tracking.identity.types import IdentityAuthority

T0 = datetime(2026, 11, 21, 15, 0, 0, tzinfo=UTC)


def _emb(seed_val: float, dim: int = 8) -> list[float]:
    vec = [0.0] * dim
    vec[0] = seed_val
    vec[1] = 1.0
    return vec


def _det(
    det_id: str,
    *,
    at: datetime = T0,
    quality: float = 0.7,
    embedding: list[float] | None = None,
    camera_id: str = "cam-living",
) -> Detection:
    return Detection(
        detection_id=det_id,
        camera_id=camera_id,
        bbox=BoundingBox(10, 10, 110, 310),
        embedding=_emb(1.0) if embedding is None else embedding,
        capture_time=at,
        event_time=at,
        confidence=0.95,
        floor_point=FloorPoint(1000, 1000),
        crop_quality=quality,
    )


def _decision(
    ph_id: str,
    identity_id: str | None,
    *,
    authority: IdentityAuthority,
    conflict: str = "",
) -> IdentityDecision:
    return IdentityDecision(
        ph_id=ph_id,
        identity_id=identity_id,
        posterior=PosteriorDist({identity_id or "UNKNOWN": 0.95}),
        revises_previous=False,
        authority=str(authority),
        conflict=conflict,
    )


def _wearable(
    ph_id: str,
    identity_id: str = "alice",
    *,
    grade: WearableAttestationGrade = WearableAttestationGrade.TRAJECTORY,
    tag_id: str = "tag-1",
    segment_id: str = "11111111-1111-4111-8111-111111111111",
) -> WearableEvidence:
    return WearableEvidence(
        ph_id=ph_id,
        tag_id=tag_id,
        identity_id=identity_id,
        evidence_class=(
            WearableEvidenceClass.TRAJECTORY
            if grade == WearableAttestationGrade.TRAJECTORY
            else WearableEvidenceClass.ROOM
        ),
        attestation_grade=grade,
        window_llr=18.0,
        margin_llr=10.0,
        matched_fixes=10,
        mean_residual_m=0.2,
        newest_fix_at=T0,
        segment_id=segment_id,
    )


@pytest.fixture(autouse=True)
def _fresh_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics_mod, "metrics", build_metrics(CollectorRegistry()))


async def test_records_only_trajectory_wearable_or_direct_face_authority() -> None:
    repo = InMemoryAppearanceSampleRepository()
    recorder = AppearanceSampleRecorder(
        repo=repo,
        config=AppearanceSampleRecorderConfig(enabled=True, min_quality=0.35, dedup_window_s=10.0),
    )

    dets = [
        _det("det-face"),
        _det("det-wear-traj"),
        _det("det-wear-room"),
        _det("det-posterior"),
        _det("det-prior"),
        _det("det-low-q", quality=0.2),
    ]
    recorded = await recorder.record_frame(
        detections=dets,
        det_to_ph={
            "det-face": "ph-face",
            "det-wear-traj": "ph-wear-traj",
            "det-wear-room": "ph-wear-room",
            "det-posterior": "ph-post",
            "det-prior": "ph-prior",
            "det-low-q": "ph-low-q",
        },
        orientation_by_detection={d.detection_id: (OrientationBin.FRONT, 0.9) for d in dets},
        decisions=[
            _decision("ph-face", "alice", authority=IdentityAuthority.DIRECT_FACE),
            _decision("ph-post", "bob", authority=IdentityAuthority.POSTERIOR),
            _decision("ph-prior", "bob", authority=IdentityAuthority.TEMPORAL_PRIOR),
            _decision("ph-low-q", "alice", authority=IdentityAuthority.DIRECT_FACE),
        ],
        committed_ids={
            "ph-face": "alice",
            "ph-wear-traj": "bob",
            "ph-wear-room": "bob",
            "ph-post": "bob",
            "ph-prior": "bob",
            "ph-low-q": "alice",
        },
        wearable_evidence_by_ph={
            "ph-wear-traj": _wearable(
                "ph-wear-traj", "bob", grade=WearableAttestationGrade.TRAJECTORY
            ),
            "ph-wear-room": _wearable("ph-wear-room", "bob", grade=WearableAttestationGrade.ROOM),
        },
        face_anchors=[],
    )

    assert len(recorded) == 2
    by_source = {s.label_source: s for s in recorded}
    assert set(by_source.keys()) == {
        AppearanceLabelSource.DIRECT_FACE,
        AppearanceLabelSource.WEARABLE_ATTESTED,
    }
    assert by_source[AppearanceLabelSource.DIRECT_FACE].identity_id == "alice"
    assert by_source[AppearanceLabelSource.WEARABLE_ATTESTED].identity_id == "bob"
    assert (
        by_source[AppearanceLabelSource.WEARABLE_ATTESTED].seed_association_id
        == "11111111-1111-4111-8111-111111111111"
    )


async def test_wearable_capture_respects_teaching_hold_and_face_conflict() -> None:
    repo = InMemoryAppearanceSampleRepository()
    holds = InMemoryWearableTeachingHoldRepository()
    await holds.create_hold(tag_id="tag-held", reason="swap_suspected", held_at=T0)
    recorder = AppearanceSampleRecorder(
        repo=repo,
        config=AppearanceSampleRecorderConfig(enabled=True),
        teaching_holds=holds,
    )

    dets = [_det("det-held"), _det("det-conflict")]
    conflicting_face = FaceAnchor(
        person_id="carol",
        match_score=0.75,
        tracklet_id="ph-conflict",
        detection_id="det-conflict",
        camera_id="cam-living",
        captured_at=T0,
        recognition_state="recognized",
    )
    recorded = await recorder.record_frame(
        detections=dets,
        det_to_ph={"det-held": "ph-held", "det-conflict": "ph-conflict"},
        orientation_by_detection={d.detection_id: (OrientationBin.FRONT, 0.9) for d in dets},
        decisions=[],
        committed_ids={"ph-held": "alice", "ph-conflict": "bob"},
        wearable_evidence_by_ph={
            "ph-held": _wearable("ph-held", "alice", tag_id="tag-held"),
            "ph-conflict": _wearable("ph-conflict", "bob", tag_id="tag-clean"),
        },
        face_anchors=[conflicting_face],
    )
    assert recorded == []
    assert await repo.list_samples() == []


async def test_dedup_window_is_10s_per_ph_and_orientation() -> None:
    repo = InMemoryAppearanceSampleRepository()
    recorder = AppearanceSampleRecorder(
        repo=repo,
        config=AppearanceSampleRecorderConfig(enabled=True, dedup_window_s=10.0),
    )

    async def _feed(at: datetime, orient: OrientationBin, det_id: str) -> int:
        rows = await recorder.record_frame(
            detections=[_det(det_id, at=at)],
            det_to_ph={det_id: "ph-1"},
            orientation_by_detection={det_id: (orient, 0.9)},
            decisions=[_decision("ph-1", "alice", authority=IdentityAuthority.DIRECT_FACE)],
            committed_ids={"ph-1": "alice"},
            wearable_evidence_by_ph={},
            face_anchors=[],
        )
        return len(rows)

    assert await _feed(T0, OrientationBin.FRONT, "d1") == 1
    # Same PH + same orientation within 10 s -> deduplicated.
    assert await _feed(T0 + timedelta(seconds=4), OrientationBin.FRONT, "d2") == 0
    # Same PH + different orientation within 10 s -> allowed.
    assert await _feed(T0 + timedelta(seconds=5), OrientationBin.BACK, "d3") == 1
    # Same PH + FRONT after >= 10 s -> allowed again.
    assert await _feed(T0 + timedelta(seconds=10), OrientationBin.FRONT, "d4") == 1
    assert len(await repo.list_samples()) == 3


async def test_negatives_require_unrecognized_face_or_guest_and_enforce_daily_cap() -> None:
    repo = InMemoryAppearanceSampleRepository()
    recorder = AppearanceSampleRecorder(
        repo=repo,
        config=AppearanceSampleRecorderConfig(
            enabled=True,
            dedup_window_s=1.0,
            max_negatives_per_day=2,
        ),
        guest_identity_ids=frozenset({"guest-1"}),
    )

    # Uncommitted PH WITHOUT an unrecognized face (e.g. resident seen from behind) is NOT negative.
    back_view = await recorder.record_frame(
        detections=[_det("d-back", at=T0)],
        det_to_ph={"d-back": "ph-resident-back"},
        orientation_by_detection={"d-back": (OrientationBin.BACK, 0.9)},
        decisions=[],
        committed_ids={"ph-resident-back": None},
        wearable_evidence_by_ph={},
        face_anchors=[],
    )
    assert back_view == []

    # Uncommitted PH WITH an unrecognized direct face -> negative #1.
    unrec_face = FaceAnchor(
        person_id="unknown",
        match_score=0.92,
        tracklet_id="ph-vis",
        detection_id="d-vis-1",
        camera_id="cam-living",
        captured_at=T0,
        recognition_state="unrecognized",
    )
    neg1 = await recorder.record_frame(
        detections=[_det("d-vis-1", at=T0)],
        det_to_ph={"d-vis-1": "ph-vis"},
        orientation_by_detection={"d-vis-1": (OrientationBin.FRONT, 0.9)},
        decisions=[],
        committed_ids={"ph-vis": None},
        wearable_evidence_by_ph={},
        face_anchors=[unrec_face],
    )
    assert len(neg1) == 1
    assert neg1[0].label_source == AppearanceLabelSource.NEGATIVE
    assert neg1[0].identity_id is None

    # Named guest PH -> negative #2 (hits daily cap = 2).
    neg2 = await recorder.record_frame(
        detections=[_det("d-guest", at=T0 + timedelta(seconds=2))],
        det_to_ph={"d-guest": "ph-guest"},
        orientation_by_detection={"d-guest": (OrientationBin.FRONT, 0.9)},
        decisions=[_decision("ph-guest", "guest-1", authority=IdentityAuthority.DIRECT_FACE)],
        committed_ids={"ph-guest": "guest-1"},
        wearable_evidence_by_ph={},
        face_anchors=[],
    )
    assert len(neg2) == 1
    assert neg2[0].label_source == AppearanceLabelSource.NEGATIVE

    # Third negative on the same local day is capped out.
    neg3 = await recorder.record_frame(
        detections=[_det("d-guest-2", at=T0 + timedelta(seconds=5))],
        det_to_ph={"d-guest-2": "ph-guest"},
        orientation_by_detection={"d-guest-2": (OrientationBin.BACK, 0.9)},
        decisions=[],
        committed_ids={"ph-guest": "guest-1"},
        wearable_evidence_by_ph={},
        face_anchors=[],
    )
    assert neg3 == []
    assert await repo.count_negatives_for_day(neg1[0].day) == 2


@pytest.mark.parametrize(
    ("batch_window_s", "cross_camera"),
    [
        (0.0, False),  # Direct execution path
        (0.5, False),  # Per-camera batch execution path
        (0.5, True),  # Cross-camera batch execution path
    ],
)
async def test_stage_wired_on_all_three_execution_paths(
    batch_window_s: float,
    cross_camera: bool,
) -> None:
    sample_repo = InMemoryAppearanceSampleRepository()
    cfg = PipelineConfig(
        batch_window_s=batch_window_s,
        cross_camera_detector_batching=cross_camera,
        allow_skeleton=True,
        signals_enabled=False,
        household_classifier=HouseholdClassifierConfig(capture_enabled=True),
    )
    pipeline = FrameProcessingPipeline(cfg)
    mock_transport = AsyncMock()
    deps = PipelineDependencies(
        detector=AsyncMock(),
        appearance_sample_repo=sample_repo,
    )
    # Patch transport/publisher connections so initialize builds stage runners cleanly.
    pipeline._transport = mock_transport  # type: ignore[assignment]
    from unittest.mock import patch

    with (
        patch("app.pipeline.frame_pipeline.RedisStreamsTransport", return_value=mock_transport),
        patch("app.pipeline.frame_pipeline.RevisionPublisher", return_value=AsyncMock()),
        patch("app.pipeline.frame_pipeline.PHContinuationPublisher", return_value=AsyncMock()),
        patch("app.pipeline.frame_pipeline.SceneSamplesPublisher", return_value=AsyncMock()),
    ):
        await pipeline.initialize(deps)

    assert pipeline._stage_runner is not None
    assert pipeline._post_world_runner is not None
    full_names = [s.name for s in pipeline._stage_runner._stages]
    post_names = [s.name for s in pipeline._post_world_runner._stages]
    assert "appearance_sample_recorder" in full_names
    assert "appearance_sample_recorder" in post_names
    assert post_names.index("appearance_sample_recorder") < post_names.index("publish")

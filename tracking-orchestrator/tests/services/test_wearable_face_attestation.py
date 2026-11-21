"""WearableFaceAttestationService: which faces are offered, how they batch, what is counted."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest
from prometheus_client import CollectorRegistry

from app.domain import (
    FaceAnchor,
    WearableAttestationGrade,
    WearableEvidence,
    WearableEvidenceClass,
)
from app.inference.face_id_client import FaceAttestationResult
from app.observability import metrics as metrics_mod
from app.observability.metrics import build_metrics
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.wearable_face_attestation import WearableFaceAttestationStage
from app.services.wearable_face_attestation import WearableFaceAttestationService
from app.transport.redis_streams import FrameReady

T0 = datetime(2026, 11, 20, 9, 0, tzinfo=UTC)


class _Client:
    def __init__(self, result: FaceAttestationResult | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self._result = result or FaceAttestationResult(moved=1, missed=0)

    async def attest_faces(
        self,
        *,
        association_id: str,
        identity_id: str,
        attestation_grade: str,
        detection_ids: Sequence[str],
    ) -> FaceAttestationResult | None:
        self.calls.append(
            {
                "association_id": association_id,
                "identity_id": identity_id,
                "attestation_grade": attestation_grade,
                "detection_ids": list(detection_ids),
            }
        )
        return self._result


class _FailingClient(_Client):
    async def attest_faces(self, **kwargs: object) -> FaceAttestationResult | None:  # type: ignore[override]
        return None


def _evidence(
    grade: WearableAttestationGrade = WearableAttestationGrade.TRAJECTORY,
    *,
    segment_id: str = "seg-1",
) -> WearableEvidence:
    return WearableEvidence(
        ph_id="ph-1",
        tag_id="tag-a",
        identity_id="alice",
        evidence_class=WearableEvidenceClass.TRAJECTORY,
        attestation_grade=grade,
        window_llr=18.0,
        margin_llr=8.0,
        matched_fixes=10,
        mean_residual_m=0.1,
        newest_fix_at=T0,
        segment_id=segment_id,
    )


def _face(det: str, person: str | None, state: str = "unrecognized") -> FaceAnchor:
    return FaceAnchor(person_id=person, match_score=0.4, recognition_state=state, detection_id=det)


def _count(result: str) -> float:
    counter = metrics_mod.metrics.cts_wearable_face_attestations_total
    return float(counter.labels(result=result)._value.get())


@pytest.fixture(autouse=True)
def _fresh_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics_mod, "metrics", build_metrics(CollectorRegistry()))


async def _round(
    service: WearableFaceAttestationService,
    at: datetime,
    *,
    det: str,
    evidence: WearableEvidence | None = None,
    emitted: list[FaceAnchor] | None = None,
    shadow: list[FaceAnchor] | None = None,
) -> None:
    await service.offer_round(
        evidence_by_ph={"ph-1": evidence or _evidence()},
        det_to_ph={det: "ph-1"},
        emitted_faces=emitted if emitted is not None else [_face(det, None)],
        shadow_faces=shadow or [],
        captured_at=at,
    )


async def test_batches_per_segment_and_sends_after_the_interval() -> None:
    client = _Client(FaceAttestationResult(moved=2, missed=1))
    service = WearableFaceAttestationService(client, batch_interval_s=10.0)
    await _round(service, T0, det="d1")
    await _round(service, T0 + timedelta(seconds=4), det="d2")
    await _round(service, T0 + timedelta(seconds=4), det="d2")  # same detection, once
    assert client.calls == []
    await _round(service, T0 + timedelta(seconds=10), det="d3")
    assert client.calls == [
        {
            "association_id": "seg-1",
            "identity_id": "alice",
            "attestation_grade": "trajectory",
            "detection_ids": ["d1", "d2", "d3"],
        }
    ]
    assert (_count("queued"), _count("sent"), _count("moved"), _count("buffer_miss")) == (
        3.0,
        3.0,
        2.0,
        1.0,
    )


async def test_a_grade_change_opens_a_separate_batch() -> None:
    client = _Client()
    service = WearableFaceAttestationService(client, batch_interval_s=10.0)
    await _round(service, T0, det="d1", evidence=_evidence(WearableAttestationGrade.ROOM))
    await _round(service, T0 + timedelta(seconds=11), det="d2")
    grades = [c["attestation_grade"] for c in client.calls]
    assert grades == ["room"]
    await _round(service, T0 + timedelta(seconds=25), det="d3")
    assert [c["detection_ids"] for c in client.calls] == [["d1"], ["d2", "d3"]]


async def test_only_emitted_faces_of_attested_phs_without_conflict_are_offered() -> None:
    client = _Client()
    service = WearableFaceAttestationService(client, batch_interval_s=0.0)
    # No emitted face (dropped by attribution, or none found).
    await _round(service, T0, det="d1", emitted=[])
    # Not attested yet.
    await _round(service, T0, det="d2", evidence=_evidence(WearableAttestationGrade.NONE))
    # A shadow-only recognition of someone else on the detection, at any similarity.
    await _round(service, T0, det="d3", shadow=[_face("d3", "bob", state="recognized")])
    # A live recognition of someone else.
    await _round(service, T0, det="d4", emitted=[_face("d4", "bob", state="recognized")])
    assert client.calls == []
    assert _count("excluded_conflict") == 2.0
    # A recognition of the tag's own member is fine.
    await _round(service, T0, det="d5", emitted=[_face("d5", "alice", state="recognized")])
    assert [c["detection_ids"] for c in client.calls] == [["d5"]]


async def test_failures_are_counted_and_purge_drops_pending_batches() -> None:
    failing = WearableFaceAttestationService(_FailingClient(), batch_interval_s=0.0)
    await _round(failing, T0, det="d1")
    assert _count("failed") == 1.0

    client = _Client()
    service = WearableFaceAttestationService(client, batch_interval_s=60.0)
    await _round(service, T0, det="d1")
    assert await service.purge_identity("alice") == 1
    await _round(service, T0 + timedelta(seconds=61), det="d2")
    await _round(service, T0 + timedelta(seconds=122), det="d3")
    assert [c["detection_ids"] for c in client.calls] == [["d2", "d3"]]


async def test_a_batch_goes_out_after_the_association_ends() -> None:
    client = _Client()
    service = WearableFaceAttestationService(client, batch_interval_s=10.0)
    await _round(service, T0, det="d1")
    # The association ends: later rounds carry no evidence and no faces.
    await service.offer_round(
        evidence_by_ph={},
        det_to_ph={},
        emitted_faces=[],
        shadow_faces=[],
        captured_at=T0 + timedelta(seconds=5),
    )
    assert client.calls == []
    await service.offer_round(
        evidence_by_ph={},
        det_to_ph={},
        emitted_faces=[],
        shadow_faces=[],
        captured_at=T0 + timedelta(seconds=11),
    )
    assert [c["detection_ids"] for c in client.calls] == [["d1"]]


async def test_the_stage_runs_on_rounds_without_evidence() -> None:
    client = _Client()
    service = WearableFaceAttestationService(client, batch_interval_s=10.0)
    await _round(service, T0, det="d1")
    later = T0 + timedelta(seconds=12)
    ctx = FrameContext(
        frame=FrameReady(
            camera_id="cam-1",
            minio_key="k",
            width=640,
            height=480,
            frame_index=1,
            capture_time_unix_ns=int(later.timestamp() * 1e9),
        ),
        event_time=later,
        capture_time=later,
    )
    await WearableFaceAttestationStage(service).run(ctx)
    assert [c["detection_ids"] for c in client.calls] == [["d1"]]

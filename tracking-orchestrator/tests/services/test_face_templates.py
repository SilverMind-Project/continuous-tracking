"""FaceTemplateService: template anchors, the per-round lifecycle, and the non-blocking matcher."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from prometheus_client import CollectorRegistry

from app.inference.face_id_client import (
    FaceQualityReading,
    TemplateMatchResult,
    TemplateMember,
    TemplateQualityReading,
)
from app.observability import metrics as metrics_pkg
from app.observability.metrics import Metrics, build_metrics
from app.services.face_templates import CompletedTemplate, FaceTemplateService, template_anchor
from app.tracking.identity.face_template_buffer import BufferedFace, FaceTemplatePolicy

_T0 = datetime(2026, 11, 12, 9, 0, tzinfo=UTC)
_POLICY = FaceTemplatePolicy(mode="enabled", min_faces=3, trigger_cooldown_s=0.0)
_QUALITY = FaceQualityReading(
    score=0.5, resolution=0.4, sharpness=0.6, detection=0.9, pitch=1.0, version="heuristic-v1"
)


@pytest.fixture(autouse=True)
def fresh_metrics(monkeypatch: pytest.MonkeyPatch) -> Metrics:
    fresh = build_metrics(registry=CollectorRegistry())
    monkeypatch.setattr(metrics_pkg, "metrics", fresh)
    return fresh


def _count(metric: object, **labels: str) -> float:
    total = 0.0
    for family in metric.collect():  # type: ignore[attr-defined]
        for sample in family.samples:
            if sample.name.endswith("_total") and all(
                sample.labels.get(k) == v for k, v in labels.items()
            ):
                total += sample.value
    return total


def _unit(seed: int) -> np.ndarray:
    v = np.random.RandomState(seed).randn(512)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _face(
    n: int,
    *,
    yaw: float = 10.0,
    det: float = 0.8,
    seconds: float = 0.0,
    recognized_as: str | None = None,
) -> BufferedFace:
    base = _unit(1) + _unit(100 + n)
    return BufferedFace(
        detection_id=f"det-{n}",
        camera_id="cam-1",
        captured_at=_T0 + timedelta(seconds=seconds),
        embedding=(base / np.linalg.norm(base)).astype(np.float32),
        quality=_QUALITY,
        det_score=det,
        yaw_deg=yaw,
        single_frame_person_id=recognized_as,
        embedding_model_version="buffalo_l",
    )


def _result(**changes: object) -> TemplateMatchResult:
    base = TemplateMatchResult(
        person_id="alice",
        recognition_state="recognized",
        best_candidate_id="alice",
        raw_similarity=0.72,
        second_best_similarity=0.1,
        member_count=3,
        aggregator_version="qwmean-v1",
        quality=TemplateQualityReading(score=0.55, resolution=0.4, sharpness=0.6),
        quality_version="heuristic-v1",
        calibrated_confidence=0.9,
        calibration_profile="template",
        calibration_status="ready",
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def _completed(members: tuple[BufferedFace, ...], **changes: object) -> CompletedTemplate:
    return CompletedTemplate("ph-1", 1, members, _result(**changes))


_MEMBERS = (
    _face(1, yaw=5.0, det=0.6, seconds=0.0),
    _face(2, yaw=20.0, det=0.9, seconds=2.0),
    _face(3, yaw=40.0, det=0.7, seconds=1.0),
)


# --- template_anchor --------------------------------------------------------------------------


def test_a_consistent_calibrated_template_is_an_authority_eligible_anchor() -> None:
    built = template_anchor(
        _completed(_MEMBERS), shadow_only=False, expected_quality_version="heuristic-v1"
    )

    assert built is not None
    anchor = built.anchor
    assert (anchor.person_id, anchor.tracklet_id, anchor.detection_id) == ("alice", "ph-1", "det-2")
    assert (anchor.aggregation, anchor.authority_eligible, built.authority_blocked) == (
        "template",
        True,
        None,
    )
    assert anchor.calibrated_confidence == pytest.approx(0.9)
    assert (anchor.match_score, anchor.quality, anchor.face_quality) == (0.72, 1.0, 0.55)
    # Medians of the members: never better than the typical member.
    assert (anchor.yaw_deg, anchor.det_score) == (20.0, 0.7)
    assert built.evidence is not None
    assert (built.evidence.source, built.evidence.tracklet_id) == ("template", "ph-1")


def test_a_member_recognized_as_someone_else_blocks_authority_but_still_votes() -> None:
    members = (*_MEMBERS[:2], _face(3, recognized_as="bob"))
    built = template_anchor(
        _completed(members), shadow_only=False, expected_quality_version="heuristic-v1"
    )

    assert built is not None
    assert built.anchor.authority_eligible is False
    assert built.anchor.calibrated_confidence == pytest.approx(0.9)  # still weighs as calibrated
    assert built.authority_blocked == "member_disagreement"


@pytest.mark.parametrize(
    ("changes", "expected_version"),
    [
        ({"calibration_profile": ""}, "heuristic-v1"),  # no template artifact loaded
        ({"calibration_profile": "single_frame"}, "heuristic-v1"),
        ({}, "heuristic-v2"),  # fitted on another quality scorer
    ],
)
def test_calibration_is_kept_only_from_a_matching_template_artifact(
    changes: dict[str, object], expected_version: str
) -> None:
    built = template_anchor(
        _completed(_MEMBERS, **changes),
        shadow_only=False,
        expected_quality_version=expected_version,
    )
    assert built is not None
    assert built.anchor.calibrated_confidence is None
    assert built.authority_blocked == "uncalibrated"


def test_a_candidate_template_names_its_best_candidate_and_an_unrecognized_one_nothing() -> None:
    candidate = template_anchor(
        _completed(_MEMBERS, recognition_state="candidate", person_id="unknown"),
        shadow_only=False,
        expected_quality_version="heuristic-v1",
    )
    assert candidate is not None
    assert (candidate.anchor.person_id, candidate.anchor.calibrated_confidence) == ("alice", None)
    assert candidate.authority_blocked is None
    assert (
        template_anchor(
            _completed(
                _MEMBERS,
                recognition_state="unrecognized",
                person_id="unknown",
                best_candidate_id=None,
            ),
            shadow_only=False,
            expected_quality_version="heuristic-v1",
        )
        is None
    )


def test_a_shadow_template_has_no_evidence_record() -> None:
    built = template_anchor(
        _completed(_MEMBERS), shadow_only=True, expected_quality_version="heuristic-v1"
    )
    assert built is not None
    assert built.anchor.shadow_only is True
    assert built.evidence is None


# --- the round lifecycle ----------------------------------------------------------------------


class _Client:
    def __init__(self, result: TemplateMatchResult | None = None) -> None:
        self.result = result or _result()
        self.calls = 0
        self.release = asyncio.Event()
        self.release.set()

    async def match_template(
        self, members: Sequence[TemplateMember], embedding_model_version: str
    ) -> TemplateMatchResult | None:
        self.calls += 1
        await self.release.wait()
        return self.result


def _service(client: _Client, policy: FaceTemplatePolicy = _POLICY) -> FaceTemplateService:
    return FaceTemplateService(policy, client, expected_quality_version="heuristic-v1")


def _feed(
    service: FaceTemplateService,
    faces: Sequence[BufferedFace],
    now: datetime,
    committed: dict[str, str | None] | None = None,
) -> None:
    service.after_round(
        faces=faces,
        det_to_ph={f.detection_id: "ph-1" for f in faces},
        committed=committed or {},
        now=now,
    )


async def test_a_template_is_requested_at_min_faces_and_offered_until_its_ph_is_resolved() -> None:
    client = _Client()
    service = _service(client)
    _feed(service, _MEMBERS[:2], _T0)
    await service.matcher.drain()
    assert client.calls == 0  # two faces are not enough

    _feed(service, _MEMBERS[2:], _T0 + timedelta(seconds=1))
    await service.matcher.drain()
    assert client.calls == 1

    # The PH was not observed in the next round: the template stays pending.
    anchors, evidence = service.pending_anchors(_T0 + timedelta(seconds=2))
    assert [a.tracklet_id for a in anchors] == ["ph-1"]
    assert len(evidence) == 1
    _feed(service, [], _T0 + timedelta(seconds=2), committed={"ph-other": None})
    anchors, _ = service.pending_anchors(_T0 + timedelta(seconds=3))
    assert [a.tracklet_id for a in anchors] == ["ph-1"]

    # Resolved this round: consumed.
    _feed(service, [], _T0 + timedelta(seconds=3), committed={"ph-1": "alice"})
    assert service.pending_anchors(_T0 + timedelta(seconds=4)) == ([], [])


async def test_a_reset_while_a_request_is_in_flight_discards_its_result(
    fresh_metrics: Metrics,
) -> None:
    client = _Client()
    client.release.clear()
    service = _service(client)
    _feed(service, _MEMBERS, _T0)
    await asyncio.sleep(0)
    assert service.matcher.in_flight() == 1

    service.reset_ph("ph-1")  # an operator split the PH meanwhile
    client.release.set()
    await service.matcher.drain()

    assert service.pending_anchors(_T0 + timedelta(seconds=1)) == ([], [])
    assert (
        _count(fresh_metrics.cts_face_template_results_discarded_total, reason="stale_after_reset")
        == 1
    )


async def test_an_expired_result_is_discarded(fresh_metrics: Metrics) -> None:
    service = _service(_Client())
    _feed(service, _MEMBERS, _T0)
    await service.matcher.drain()

    assert service.pending_anchors(_T0 + timedelta(seconds=30)) == ([], [])
    assert _count(fresh_metrics.cts_face_template_results_discarded_total, reason="expired") == 1


async def test_a_slow_service_never_blocks_a_round() -> None:
    client = _Client()
    client.release.clear()
    service = _service(client)

    _feed(service, _MEMBERS, _T0)  # returns at once: the request runs in the background
    assert service.pending_anchors(_T0 + timedelta(seconds=1)) == ([], [])

    client.release.set()
    await service.matcher.drain()
    anchors, _ = service.pending_anchors(_T0 + timedelta(seconds=2))
    assert [a.person_id for a in anchors] == ["alice"]


async def test_one_request_per_ph_and_the_in_flight_limit(fresh_metrics: Metrics) -> None:
    client = _Client()
    client.release.clear()
    policy = FaceTemplatePolicy(
        mode="enabled",
        min_faces=2,
        top_k=2,
        trigger_cooldown_s=0.0,
        max_concurrency=1,
        max_in_flight=1,
    )
    service = _service(client, policy)

    _feed(service, _MEMBERS[:2], _T0)
    _feed(service, [_face(4, seconds=1.0)], _T0 + timedelta(seconds=1))  # same PH, still in flight
    service.after_round(
        faces=[_face(5), _face(6)],
        det_to_ph={"det-5": "ph-2", "det-6": "ph-2"},
        committed={},
        now=_T0,
    )
    await asyncio.sleep(0)

    assert service.matcher.in_flight() == 1
    assert (
        _count(fresh_metrics.cts_face_template_requests_total, result="dropped_in_flight_limit")
        == 1
    )
    await service.stop()
    assert service.matcher.in_flight() == 0


async def test_an_identity_change_resets_the_buffer_and_drops_the_template() -> None:
    service = _service(_Client())
    _feed(service, _MEMBERS, _T0, committed={"ph-1": "alice"})
    await service.matcher.drain()
    assert service.buffer.size("ph-1") == 3

    _feed(service, [], _T0 + timedelta(seconds=1), committed={"ph-1": "bob"})

    assert service.buffer.size("ph-1") == 0
    assert service.pending_anchors(_T0 + timedelta(seconds=2)) == ([], [])


async def test_authority_blocks_are_counted_once_when_consumed(fresh_metrics: Metrics) -> None:
    client = _Client(_result(calibration_profile=""))
    service = _service(client)
    _feed(service, _MEMBERS, _T0)
    await service.matcher.drain()

    service.pending_anchors(_T0 + timedelta(seconds=1))
    _feed(service, [], _T0 + timedelta(seconds=1), committed={"ph-other": None})  # not resolved
    service.pending_anchors(_T0 + timedelta(seconds=2))
    _feed(service, [], _T0 + timedelta(seconds=2), committed={"ph-1": None})  # resolved

    assert (
        _count(fresh_metrics.cts_face_template_authority_blocked_total, reason="uncalibrated") == 1
    )


def test_the_service_is_not_built_while_templates_are_off() -> None:
    with pytest.raises(ValueError, match="off"):
        FaceTemplateService(FaceTemplatePolicy(), _Client(), expected_quality_version="v")

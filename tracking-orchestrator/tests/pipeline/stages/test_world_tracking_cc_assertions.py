"""WorldTrackingStage CC assertion mode-gating tests.

Covers the resolver.cc_assertion_mode contract: off (matcher not consulted),
shadow (matched but not injected; outcome metrics recorded), enabled
(matched anchors injected as evidence), and the room-lookup/scale wiring
using constructor-injected ExternalIdentityEvidenceSource.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from prometheus_client import CollectorRegistry

from app.domain import (
    AnchorOrigin,
    BoundingBox,
    Detection,
    ExternalIdentityAssertion,
    FloorPoint,
    PersonHypothesis,
)
from app.observability import metrics as metrics_pkg
from app.observability.metrics import build_metrics
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.world_tracking import WorldTrackingStage
from app.pipeline.types import LiveConfigHolder
from app.services.camera_room_map import CameraRoomBinding, CameraRoomMap, RoomPolygonMap
from app.tracking.world.tracker import WorldTrackerResult
from app.transport.redis_streams import FrameReady


class _FakeExternalEvidenceSource:
    def __init__(self, assertions: Sequence[ExternalIdentityAssertion]) -> None:
        self._assertions = tuple(assertions)
        self.recent_calls = 0

    async def recent(self, max_age_s: float = 30.0) -> Sequence[ExternalIdentityAssertion]:
        self.recent_calls += 1
        return self._assertions


def _make_ctx(now: datetime, camera_id: str = "cam-1") -> FrameContext:
    frame = FrameReady(
        camera_id=camera_id,
        minio_key="",
        frame_index=1,
        capture_time_unix_ns=int(now.timestamp() * 1e9),
        received_time_unix_ns=int(now.timestamp() * 1e9),
        width=640,
        height=480,
    )
    ctx = FrameContext(
        frame=frame,
        event_time=now,
        capture_time=now,
        effective_width=640,
        effective_height=480,
    )
    ctx.domain_detections = [
        Detection(
            detection_id="det-1",
            camera_id=camera_id,
            bbox=BoundingBox(x_min=100, y_min=50, x_max=250, y_max=400),
            confidence=0.9,
            floor_point=FloorPoint(0, 0, calibrated=False),
            embedding=[],
            capture_time=now,
            event_time=now,
        )
    ]
    return ctx


def _make_stage(
    *,
    mode: str,
    external_evidence: _FakeExternalEvidenceSource,
    tracker_mock: MagicMock,
    camera_room_map: CameraRoomMap,
    room_match_confidence_scale: float = 0.8,
) -> WorldTrackingStage:
    return WorldTrackingStage(
        tracker=tracker_mock,
        live_config=LiveConfigHolder(camera_room_map, RoomPolygonMap()),
        external_evidence=external_evidence,
        cc_assertion_mode=mode,
        room_match_confidence_scale=room_match_confidence_scale,
    )


def _mock_tracker(result: WorldTrackerResult) -> tuple[MagicMock, list[Any]]:
    captured_face_anchors: list[Any] = []
    tracker_mock = MagicMock()

    async def _step(*, observations, face_anchors, **kw):
        captured_face_anchors.append(face_anchors)
        return result

    tracker_mock.step = AsyncMock(side_effect=_step)
    return tracker_mock, captured_face_anchors


@pytest.mark.asyncio
async def test_off_mode_matcher_not_consulted() -> None:
    """off skips external assertion matching entirely."""
    now = datetime.now(UTC)
    source = _FakeExternalEvidenceSource(
        [
            ExternalIdentityAssertion(
                person_id="alice",
                calibrated_confidence=0.9,
                camera_id="recamera_kitchen",
                captured_at=now,
                room_name="Kitchen",
            )
        ]
    )
    result = WorldTrackerResult(updated_phs=[], snapshots=[], continuations=[])
    tracker_mock, captured = _mock_tracker(result)
    room_map = CameraRoomMap()
    await room_map.set_all(
        [CameraRoomBinding(camera_id="cam-1", room_id="r1", room_name="Kitchen", bound_at=now)]
    )
    stage = _make_stage(
        mode="off",
        external_evidence=source,
        tracker_mock=tracker_mock,
        camera_room_map=room_map,
    )

    await stage.run_many([_make_ctx(now)])

    assert source.recent_calls == 0
    assert captured[0] == []


@pytest.mark.asyncio
async def test_enabled_mode_injects_matched_anchors() -> None:
    """enabled mode passes matched CC anchors to WorldTracker.step()."""
    now = datetime.now(UTC)
    source = _FakeExternalEvidenceSource(
        [
            ExternalIdentityAssertion(
                person_id="alice",
                calibrated_confidence=0.9,
                camera_id="recamera_kitchen",
                captured_at=now,
                room_name="Kitchen",
            )
        ]
    )
    result = WorldTrackerResult(updated_phs=[], snapshots=[], continuations=[])
    tracker_mock, captured = _mock_tracker(result)
    room_map = CameraRoomMap()
    await room_map.set_all(
        [CameraRoomBinding(camera_id="cam-1", room_id="r1", room_name="Kitchen", bound_at=now)]
    )
    stage = _make_stage(
        mode="enabled",
        external_evidence=source,
        tracker_mock=tracker_mock,
        camera_room_map=room_map,
    )

    await stage.run_many([_make_ctx(now)])

    assert source.recent_calls == 1
    injected = captured[0]
    assert len(injected) == 1
    assert injected[0].person_id == "alice"
    assert injected[0].origin == AnchorOrigin.CC_ASSERTION


@pytest.mark.asyncio
async def test_stage_passes_room_lookup_and_configured_scale_to_matcher() -> None:
    """Wiring assertion: the stage resolves the camera->room lookup itself and
    threads its configured room_match_confidence_scale into the matcher."""
    now = datetime.now(UTC)
    source = _FakeExternalEvidenceSource(
        [
            ExternalIdentityAssertion(
                person_id="alice",
                calibrated_confidence=0.9,
                camera_id="recamera_kitchen",
                captured_at=now,
                room_name="Kitchen",
            )
        ]
    )
    result = WorldTrackerResult(updated_phs=[], snapshots=[], continuations=[])
    tracker_mock, captured = _mock_tracker(result)
    room_map = CameraRoomMap()
    await room_map.set_all(
        [CameraRoomBinding(camera_id="cam-1", room_id="r1", room_name="Kitchen", bound_at=now)]
    )
    stage = _make_stage(
        mode="enabled",
        external_evidence=source,
        tracker_mock=tracker_mock,
        camera_room_map=room_map,
        room_match_confidence_scale=0.4,
    )

    await stage.run_many([_make_ctx(now)])

    injected = captured[0]
    assert len(injected) == 1
    assert injected[0].match_score == pytest.approx(0.9 * 0.4)


@pytest.mark.asyncio
async def test_shadow_mode_does_not_inject_but_records_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """shadow mode matches (metrics fire) but withholds injection; the
    resolved PH identity vs the assertion's person_id drives the outcome label."""
    fresh_metrics = build_metrics(registry=CollectorRegistry())
    monkeypatch.setattr(metrics_pkg, "metrics", fresh_metrics)

    now = datetime.now(UTC)
    source = _FakeExternalEvidenceSource(
        [
            ExternalIdentityAssertion(
                person_id="alice",
                calibrated_confidence=0.9,
                camera_id="recamera_kitchen",
                captured_at=now,
                room_name="Kitchen",
            )
        ]
    )
    ph = PersonHypothesis(
        ph_id="ph-1",
        state_mean=(0.0, 0.0, 0.0, 0.0),
        state_cov=(0.0,) * 16,
        born_at=now,
        last_seen_at=now,
        last_seen_camera="cam-1",
        observation_count=1,
        current_identity_id=None,
    )
    result = WorldTrackerResult(
        updated_phs=[ph],
        snapshots=[],
        continuations=[],
        det_to_ph={"det-1": "ph-1"},
    )
    tracker_mock, captured = _mock_tracker(result)
    room_map = CameraRoomMap()
    await room_map.set_all(
        [CameraRoomBinding(camera_id="cam-1", room_id="r1", room_name="Kitchen", bound_at=now)]
    )
    stage = _make_stage(
        mode="shadow",
        external_evidence=source,
        tracker_mock=tracker_mock,
        camera_room_map=room_map,
    )

    await stage.run_many([_make_ctx(now)])

    assert source.recent_calls == 1
    assert captured[0] == []
    assert fresh_metrics.cc_assertions_matched_total.labels(gate="room")._value.get() == 1.0
    assert (
        fresh_metrics.cc_assertions_shadow_total.labels(outcome="would_name_unknown")._value.get()
        == 1.0
    )


@pytest.mark.asyncio
async def test_shadow_mode_disagrees_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    """A PH resolved to a different identity than the assertion records 'disagrees'."""
    fresh_metrics = build_metrics(registry=CollectorRegistry())
    monkeypatch.setattr(metrics_pkg, "metrics", fresh_metrics)

    now = datetime.now(UTC)
    source = _FakeExternalEvidenceSource(
        [
            ExternalIdentityAssertion(
                person_id="alice",
                calibrated_confidence=0.9,
                camera_id="recamera_kitchen",
                captured_at=now,
                room_name="Kitchen",
            )
        ]
    )
    ph = PersonHypothesis(
        ph_id="ph-1",
        state_mean=(0.0, 0.0, 0.0, 0.0),
        state_cov=(0.0,) * 16,
        born_at=now,
        last_seen_at=now,
        last_seen_camera="cam-1",
        observation_count=1,
        current_identity_id="bob",
    )
    result = WorldTrackerResult(
        updated_phs=[ph], snapshots=[], continuations=[], det_to_ph={"det-1": "ph-1"}
    )
    tracker_mock, _captured = _mock_tracker(result)
    room_map = CameraRoomMap()
    await room_map.set_all(
        [CameraRoomBinding(camera_id="cam-1", room_id="r1", room_name="Kitchen", bound_at=now)]
    )
    stage = _make_stage(
        mode="shadow",
        external_evidence=source,
        tracker_mock=tracker_mock,
        camera_room_map=room_map,
    )

    await stage.run_many([_make_ctx(now)])

    assert fresh_metrics.cc_assertions_shadow_total.labels(outcome="disagrees")._value.get() == 1.0


@pytest.mark.asyncio
async def test_uncalibrated_assertion_rejected_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rejection metrics fire in shadow/enabled mode; silence is not success."""
    fresh_metrics = build_metrics(registry=CollectorRegistry())
    monkeypatch.setattr(metrics_pkg, "metrics", fresh_metrics)

    now = datetime.now(UTC)
    source = _FakeExternalEvidenceSource(
        [
            ExternalIdentityAssertion(
                person_id="alice",
                calibrated_confidence=None,
                camera_id="recamera_kitchen",
                captured_at=now,
                room_name="Kitchen",
            )
        ]
    )
    result = WorldTrackerResult(updated_phs=[], snapshots=[], continuations=[])
    tracker_mock, captured = _mock_tracker(result)
    room_map = CameraRoomMap()
    await room_map.set_all(
        [CameraRoomBinding(camera_id="cam-1", room_id="r1", room_name="Kitchen", bound_at=now)]
    )
    stage = _make_stage(
        mode="shadow",
        external_evidence=source,
        tracker_mock=tracker_mock,
        camera_room_map=room_map,
    )

    await stage.run_many([_make_ctx(now)])

    assert captured[0] == []
    rejected = fresh_metrics.cc_assertions_rejected_total.labels(reason="uncalibrated")
    assert rejected._value.get() == 1.0

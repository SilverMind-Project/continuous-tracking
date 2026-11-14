"""An identity commit stores the capture time of its newest evidence, not the round time.

A face template is matched in the background and applied up to ``face_templates.max_age_s`` after
its newest member face. Its commit must start the 30 s maintenance window at that member's capture
time, so a late template never makes an identity look freshly observed. A frame face is this
round's evidence and is unchanged. The clock never moves backwards for an identity the PH holds.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.config import Settings
from app.domain import (
    BoundingBox,
    FaceAnchor,
    FloorPoint,
    Identity,
    PersonHypothesis,
    WorldObservation,
)
from app.main import _build_resolver_config, _build_world_tracker_config
from app.storage.base import InMemoryPHRepository, InMemoryWorldObservationRepository
from app.storage.gallery import InMemoryGalleryRepository
from app.tracking.identity_resolver import IdentityResolver
from app.tracking.world.tracker import WorldTracker, WorldTrackerResult

_SETTINGS_PATH = Path(__file__).resolve().parents[4] / "config" / "settings.yaml"
_T0 = datetime(2026, 11, 14, 9, 0, 0, tzinfo=UTC)
_ROOM = {"living_room": [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]}


async def _tracker(ph_repo: InMemoryPHRepository) -> WorldTracker:
    os.environ.setdefault("PERSON_ID_SERVICE_URL", "http://person-id.test")
    settings = Settings(_SETTINGS_PATH)
    settings.reload()
    gallery = InMemoryGalleryRepository()
    for identity in ("alice", "bob"):
        await gallery.upsert_identity(
            Identity(identity_id=identity, display_name=identity, enrolled_at=_T0)
        )
    return WorldTracker(
        ph_repo=ph_repo,
        obs_repo=InMemoryWorldObservationRepository(),
        config=_build_world_tracker_config(settings),
        identity_resolver=IdentityResolver(
            gallery_repo=gallery, config=_build_resolver_config(settings)
        ),
        gallery_repo=gallery,
    )


def _obs(at: datetime, face: FaceAnchor | None = None) -> WorldObservation:
    det_id = f"d-{(at - _T0).total_seconds():.1f}"
    return WorldObservation(
        camera_id="cam-1",
        frame_index=int((at - _T0).total_seconds() * 10),
        captured_at=at,
        floor_point=FloorPoint(x_mm=5000, y_mm=5000, calibrated=True),
        bbox=BoundingBox(x_min=100, y_min=100, x_max=150, y_max=300),
        embedding=[1.0, 0.0, 0.0],
        detection_confidence=0.9,
        detection_id=det_id,
        quality=0.85,
        face_anchor=None if face is None else _with_detection(face, det_id),
    )


def _with_detection(face: FaceAnchor, det_id: str) -> FaceAnchor:
    return FaceAnchor(
        person_id=face.person_id,
        match_score=face.match_score,
        quality=face.quality,
        detection_id=det_id,
        camera_id=face.camera_id,
        captured_at=face.captured_at,
        recognition_state=face.recognition_state,
        similarity=face.similarity,
        yaw_deg=face.yaw_deg,
        calibrated_confidence=face.calibrated_confidence,
    )


def _frame_face(at: datetime) -> FaceAnchor:
    """A calibrated single-frame face: ArcFace authority under every evidence policy."""
    return FaceAnchor(
        person_id="alice",
        match_score=0.92,
        quality=1.0,
        camera_id="cam-1",
        captured_at=at,
        recognition_state="recognized",
        similarity=0.92,
        yaw_deg=5.0,
        calibrated_confidence=0.9,
    )


def _template(ph_id: str, newest_member_at: datetime) -> FaceAnchor:
    """A calibrated, member-consistent template; its ``captured_at`` is its newest member's."""
    return FaceAnchor(
        person_id="alice",
        match_score=0.62,
        quality=1.0,
        tracklet_id=ph_id,
        detection_id="d-member",
        camera_id="cam-1",
        captured_at=newest_member_at,
        recognition_state="recognized",
        similarity=0.62,
        yaw_deg=10.0,
        calibrated_confidence=0.88,
        face_quality=0.5,
        aggregation="template",
        authority_eligible=True,
    )


async def _step(
    tracker: WorldTracker, obs: WorldObservation, extra: list[FaceAnchor] | None = None
) -> WorldTrackerResult:
    faces = [obs.face_anchor] if obs.face_anchor is not None else []
    return await tracker.step(
        observations=[obs],
        now=obs.captured_at,
        room_polygons=_ROOM,
        face_anchors=faces + (extra or []),
    )


async def _the_ph(repo: InMemoryPHRepository) -> PersonHypothesis:
    (ph,) = await repo.list_open()
    return ph


async def _faceless_walk(tracker: WorldTracker, start: datetime, until: datetime) -> None:
    at = start
    while at <= until:
        await _step(tracker, _obs(at))
        at += timedelta(seconds=0.5)


async def test_template_commit_stores_its_newest_member_time() -> None:
    repo = InMemoryPHRepository()
    tracker = await _tracker(repo)
    await _faceless_walk(tracker, _T0, _T0 + timedelta(seconds=11.5))
    ph = await _the_ph(repo)
    assert ph.current_identity_id is None

    applied_at = _T0 + timedelta(seconds=12)
    member_at = _T0
    result = await _step(tracker, _obs(applied_at), [_template(ph.ph_id, member_at)])

    (decision,) = result.identity_decisions
    assert (decision.identity_id, decision.evidence_backed) == ("alice", True)
    assert decision.evidence_at == member_at
    ph = await _the_ph(repo)
    assert ph.last_independent_identity_evidence_at == member_at
    assert ph.current_identity_committed_at == applied_at


async def test_maintenance_window_runs_from_the_member_not_the_round() -> None:
    repo = InMemoryPHRepository()
    tracker = await _tracker(repo)
    await _faceless_walk(tracker, _T0, _T0 + timedelta(seconds=11.5))
    ph = await _the_ph(repo)
    await _step(tracker, _obs(_T0 + timedelta(seconds=12)), [_template(ph.ph_id, _T0)])

    # Prior-only maintenance until 30 s after the member face, then Unknown: not 30 s after the
    # round that applied the template (which would hold the name until 42 s).
    await _faceless_walk(tracker, _T0 + timedelta(seconds=12.5), _T0 + timedelta(seconds=29.5))
    assert (await _the_ph(repo)).current_identity_id == "alice"
    await _faceless_walk(tracker, _T0 + timedelta(seconds=30.5), _T0 + timedelta(seconds=31.0))
    assert (await _the_ph(repo)).current_identity_id is None


async def test_frame_face_commit_is_stamped_with_its_round() -> None:
    repo = InMemoryPHRepository()
    tracker = await _tracker(repo)
    await _faceless_walk(tracker, _T0, _T0 + timedelta(seconds=11.5))
    at = _T0 + timedelta(seconds=12)
    result = await _step(tracker, _obs(at, _frame_face(at)))

    (decision,) = result.identity_decisions
    assert (decision.identity_id, decision.evidence_at) == ("alice", at)
    assert (await _the_ph(repo)).last_independent_identity_evidence_at == at


async def test_an_older_template_never_moves_the_clock_backwards() -> None:
    repo = InMemoryPHRepository()
    tracker = await _tracker(repo)
    await _faceless_walk(tracker, _T0, _T0 + timedelta(seconds=9.5))
    face_at = _T0 + timedelta(seconds=10)
    await _step(tracker, _obs(face_at, _frame_face(face_at)))
    ph = await _the_ph(repo)
    assert ph.last_independent_identity_evidence_at == face_at

    result = await _step(
        tracker,
        _obs(_T0 + timedelta(seconds=10.5)),
        [_template(ph.ph_id, _T0 + timedelta(seconds=2))],
    )

    (decision,) = result.identity_decisions
    assert decision.evidence_at == face_at
    assert (await _the_ph(repo)).last_independent_identity_evidence_at == face_at

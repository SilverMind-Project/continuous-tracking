"""A Person Hypothesis is born Unknown; only the identity resolver writes its identity.

A spawned PH used to take its first face's ``person_id`` (even a grey-zone candidate) as
``current_identity_id`` before the resolver ran. That gave an identity nobody committed a 0.6
prior, made the newborn an incumbent holder in the duplicate-active-identity guard, and produced
a "demoted" revision for an identity that was never assigned. These tests pin the rule with the
production configuration from ``config/settings.yaml``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.config import Settings
from app.domain import (
    BoundingBox,
    FaceAnchor,
    FloorPoint,
    Identity,
    IdentityDecision,
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
_IDENTITIES = ("amma", "grandma")


class _SaveRecordingPHRepository(InMemoryPHRepository):
    """Records every PH passed to ``save`` so a test can see the row written at spawn."""

    def __init__(self) -> None:
        super().__init__()
        self.saved: list[PersonHypothesis] = []

    async def save(self, ph: PersonHypothesis) -> None:
        self.saved.append(ph)
        await super().save(ph)


@dataclass(frozen=True)
class _Harness:
    tracker: WorldTracker
    ph_repo: _SaveRecordingPHRepository


async def _harness() -> _Harness:
    os.environ.setdefault("PERSON_ID_SERVICE_URL", "http://person-id.test")
    settings = Settings(_SETTINGS_PATH)
    settings.reload()
    gallery = InMemoryGalleryRepository()
    for identity in _IDENTITIES:
        await gallery.upsert_identity(
            Identity(identity_id=identity, display_name=identity, enrolled_at=_T0)
        )
    ph_repo = _SaveRecordingPHRepository()
    tracker = WorldTracker(
        ph_repo=ph_repo,
        obs_repo=InMemoryWorldObservationRepository(),
        config=_build_world_tracker_config(settings),
        identity_resolver=IdentityResolver(
            gallery_repo=gallery, config=_build_resolver_config(settings)
        ),
        gallery_repo=gallery,
    )
    return _Harness(tracker=tracker, ph_repo=ph_repo)


def _anchor(
    state: str, similarity: float, det_id: str, at: datetime, calibrated: float | None = None
) -> FaceAnchor:
    return FaceAnchor(
        person_id="amma",
        match_score=similarity,
        quality=1.0,
        detection_id=det_id,
        camera_id="cam-1",
        captured_at=at,
        recognition_state=state,
        similarity=similarity,
        yaw_deg=5.0,
        calibrated_confidence=calibrated,
    )


def _obs(det_id: str, at: datetime, anchor: FaceAnchor | None) -> WorldObservation:
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
        face_anchor=anchor,
    )


async def _step(tracker: WorldTracker, obs: WorldObservation) -> WorldTrackerResult:
    return await tracker.step(
        observations=[obs],
        now=obs.captured_at,
        room_polygons=_ROOM,
        face_anchors=[obs.face_anchor] if obs.face_anchor is not None else [],
    )


def _only(decisions: list[IdentityDecision]) -> IdentityDecision:
    assert len(decisions) == 1
    return decisions[0]


_FIRST_FACES = [
    pytest.param("candidate", 0.32, None, id="candidate"),
    pytest.param("recognized", 0.45, None, id="recognized_weak"),
    pytest.param("recognized", 0.90, None, id="recognized_strong"),
    pytest.param("recognized", 0.95, 0.93, id="recognized_calibrated"),
]


@pytest.mark.parametrize(("state", "similarity", "calibrated"), _FIRST_FACES)
async def test_ph_is_saved_unknown_at_spawn(
    state: str, similarity: float, calibrated: float | None
) -> None:
    h = await _harness()
    await _step(h.tracker, _obs("d0", _T0, _anchor(state, similarity, "d0", _T0, calibrated)))

    assert h.ph_repo.saved, "the spawn must persist the new PH"
    assert h.ph_repo.saved[0].current_identity_id is None


@pytest.mark.parametrize(("state", "similarity", "calibrated"), _FIRST_FACES)
async def test_first_face_is_decided_like_any_unknown_ph(
    state: str, similarity: float, calibrated: float | None
) -> None:
    newborn = await _harness()
    born = _only(
        (
            await _step(
                newborn.tracker,
                _obs("d0", _T0, _anchor(state, similarity, "d0", _T0, calibrated)),
            )
        ).identity_decisions
    )

    existing = await _harness()
    await _step(existing.tracker, _obs("d0", _T0, None))
    at = _T0 + timedelta(seconds=0.5)
    later = _only(
        (
            await _step(
                existing.tracker, _obs("d1", at, _anchor(state, similarity, "d1", at, calibrated))
            )
        ).identity_decisions
    )

    assert (born.identity_id, born.authority, born.evidence_backed) == (
        later.identity_id,
        later.authority,
        later.evidence_backed,
    )
    assert born.previous_identity_id is None


async def test_candidate_first_face_neither_commits_nor_revises() -> None:
    """A grey-zone first face no longer becomes a prior that is then demoted.

    (A recognized face below the emission floor never reaches the resolver live; when one does,
    it is weighed as for an existing Unknown PH, which the test above pins.)
    """
    h = await _harness()
    result = await _step(h.tracker, _obs("d0", _T0, _anchor("candidate", 0.32, "d0", _T0)))

    assert _only(result.identity_decisions).identity_id is None
    assert result.revisions == []
    (ph,) = await h.ph_repo.list_open()
    assert ph.current_identity_id is None


async def test_strong_first_face_still_commits_on_frame_one() -> None:
    """Uniform prior plus one strong face still commits without the seeded prior."""
    h = await _harness()
    result = await _step(h.tracker, _obs("d0", _T0, _anchor("recognized", 0.90, "d0", _T0)))

    assert _only(result.identity_decisions).identity_id == "amma"
    (ph,) = await h.ph_repo.list_open()
    assert ph.current_identity_id == "amma"
    # A first commit is published as a revision from Unknown, as for any Unknown PH.
    assert [(r.previous_identity_id, r.new_identity_id) for r in result.revisions] == [
        (None, "amma")
    ]

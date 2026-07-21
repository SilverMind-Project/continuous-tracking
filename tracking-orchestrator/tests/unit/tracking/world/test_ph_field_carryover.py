"""M08 (F10): structural proof that every ``PersonHypothesis`` mutation site uses
``dataclasses.replace`` and therefore carries unlisted fields automatically.

Each site below declares its changed-field set as an explicit ``frozenset[str]``
and this file asserts, for every current dataclass field:

- fields IN the changed set actually differ after the operation (the site does
  what it claims);
- fields NOT in the changed set are carried through unchanged (nothing leaks).

``test_known_ph_fields_matches_dataclass`` is the guard that makes this test
suite fail-by-design when a field is added to or removed from ``PersonHypothesis``:
it is a hardcoded snapshot of field names, not derived from the dataclass, so a
new field appears in neither a site's changed set nor its carried set until a
human classifies it -- the review moment the M02 incident lacked.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from app.domain import (
    BoundingBox,
    FloorPoint,
    OrientationBin,
    PersonHypothesis,
    ViewPrototype,
    WorldObservation,
)
from app.storage.base import InMemoryPHRepository, InMemoryWorldObservationRepository
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.kalman import initialize
from app.tracking.world.tracker import WorldTracker, _advance_unmatched_ph

_T0 = datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC)
_CAM = "cam-sentinel"

# Hardcoded snapshot of PersonHypothesis field names. Do not derive this from
# dataclasses.fields(PersonHypothesis) -- the whole point is that a field
# missing from this set (added or removed) fails test_known_ph_fields_matches_dataclass
# below, forcing a conscious carried/changed classification at every site.
_KNOWN_PH_FIELDS = frozenset(
    {
        "ph_id",
        "state_mean",
        "state_cov",
        "born_at",
        "last_seen_at",
        "last_seen_camera",
        "observation_count",
        "current_identity_id",
        "current_identity_committed_at",
        "last_independent_identity_evidence_at",
        "gallery_mean",
        "height_estimate_m",
        "active_cameras",
        "closed_at",
        "last_floor_speed_m_s",
        "last_posture",
        "metadata",
        "mean_quality",
        "view_prototypes",
    }
)


def test_known_ph_fields_matches_dataclass() -> None:
    """Guard: adding/removing a PersonHypothesis field must update this file.

    Every site's changed-field frozenset below is checked against
    _KNOWN_PH_FIELDS, not against dataclasses.fields() directly, so a new field
    is invisible to every site's classification until _KNOWN_PH_FIELDS is
    updated -- at which point each site's changed-field set must be reviewed and
    the new field consciously added to the changed set (if that site updates it)
    or left out (if it should be carried).
    """
    actual = {f.name for f in dataclasses.fields(PersonHypothesis)}
    assert actual == _KNOWN_PH_FIELDS, (
        "PersonHypothesis fields changed -- classify the new/removed field as "
        "changed or carried at every site in this file, then update _KNOWN_PH_FIELDS"
    )


def _sentinel_ph(ph_id: str = "ph-sentinel") -> PersonHypothesis:
    """A PersonHypothesis with every field holding a distinct, non-default value."""
    return PersonHypothesis(
        ph_id=ph_id,
        state_mean=(1.1, 2.2, 0.3, 0.4),
        # A valid (diagonal, positive-definite) covariance with four distinct
        # sentinel values -- a flat range of distinct floats is not a valid
        # covariance matrix and produces NaN/negative-sqrt warnings downstream.
        state_cov=(
            1.1,
            0.0,
            0.0,
            0.0,
            0.0,
            2.2,
            0.0,
            0.0,
            0.0,
            0.0,
            0.33,
            0.0,
            0.0,
            0.0,
            0.0,
            0.44,
        ),
        born_at=_T0 - timedelta(minutes=5),
        last_seen_at=_T0,
        last_seen_camera=_CAM,
        observation_count=7,
        current_identity_id="sentinel-identity",
        current_identity_committed_at=_T0 - timedelta(seconds=30),
        last_independent_identity_evidence_at=_T0 - timedelta(seconds=20),
        gallery_mean=[0.9, 0.1, 0.2, 0.3],
        height_estimate_m=1.65,
        active_cameras=frozenset([_CAM, "cam-other"]),
        closed_at=None,
        last_floor_speed_m_s=0.42,
        last_posture="sitting",
        metadata={"sentinel_key": "sentinel_value"},
        mean_quality=0.77,
        view_prototypes=(
            ViewPrototype(orientation=OrientationBin.FRONT, embedding=(0.1, 0.2, 0.3), count=3),
        ),
    )


def _assert_carryover(
    before: PersonHypothesis, after: PersonHypothesis, changed: frozenset[str]
) -> None:
    unknown = changed - _KNOWN_PH_FIELDS
    assert changed <= _KNOWN_PH_FIELDS, f"unknown field(s) in changed set: {unknown}"
    for f in dataclasses.fields(PersonHypothesis):
        name = f.name
        before_val = getattr(before, name)
        after_val = getattr(after, name)
        if name in changed:
            assert before_val != after_val, (
                f"field {name!r} is declared changed but has the same value "
                f"before and after -- the changed-field set is wrong, not the code"
            )
        else:
            assert before_val == after_val, (
                f"field {name!r} was not declared changed but differs after the "
                f"operation -- either the code changed it silently or the "
                f"changed-field set needs updating"
            )


# ---------------------------------------------------------------------------
# Pure helper: _advance_unmatched_ph
# ---------------------------------------------------------------------------

_ADVANCE_CLOSE_CHANGED = frozenset({"closed_at"})
_ADVANCE_KEEP_OPEN_CHANGED = frozenset({"state_mean", "state_cov"})


def test_advance_unmatched_ph_close_branch_carries_everything_but_closed_at() -> None:
    ph = _sentinel_ph()
    predicted = initialize(5.0, 5.0, 1.0, 0.5, _T0)
    after = _advance_unmatched_ph(
        ph, predicted, now=_T0 + timedelta(seconds=100), ph_close_grace_s=5.0
    )
    assert after.closed_at == _T0 + timedelta(seconds=100)
    _assert_carryover(ph, after, _ADVANCE_CLOSE_CHANGED)


def test_advance_unmatched_ph_keep_open_branch_carries_everything_but_kalman_state() -> None:
    ph = _sentinel_ph()
    predicted = initialize(5.0, 5.0, 1.0, 0.5, _T0)
    after = _advance_unmatched_ph(
        ph, predicted, now=_T0 + timedelta(seconds=1), ph_close_grace_s=5.0
    )
    assert after.closed_at is None
    _assert_carryover(ph, after, _ADVANCE_KEEP_OPEN_CHANGED)


# ---------------------------------------------------------------------------
# InMemoryPHRepository admin operations
# ---------------------------------------------------------------------------

_CORRECT_IDENTITY_CHANGED = frozenset({"current_identity_id", "current_identity_committed_at"})


@pytest.mark.asyncio
async def test_correct_identity_carries_everything_but_identity_fields() -> None:
    repo = InMemoryPHRepository()
    ph = _sentinel_ph("ph-1")
    await repo.save(ph)

    await repo.correct_identity(
        "ph-1", new_identity_id="corrected-identity", reason="test", actor="admin"
    )
    after = await repo.get("ph-1")
    assert after is not None
    assert after.current_identity_id == "corrected-identity"
    _assert_carryover(ph, after, _CORRECT_IDENTITY_CHANGED)


@pytest.mark.asyncio
async def test_batch_correct_carries_everything_but_identity_fields() -> None:
    repo = InMemoryPHRepository()
    ph = _sentinel_ph("ph-1")
    await repo.save(ph)

    await repo.batch_correct(["ph-1"], ["corrected-identity"], "admin", ["test"])
    after = await repo.get("ph-1")
    assert after is not None
    assert after.current_identity_id == "corrected-identity"
    _assert_carryover(ph, after, _CORRECT_IDENTITY_CHANGED)


_MERGE_SOURCE_CHANGED = frozenset({"closed_at", "metadata"})


@pytest.mark.asyncio
async def test_merge_source_closure_carries_everything_but_closed_at_and_metadata() -> None:
    repo = InMemoryPHRepository()
    source = _sentinel_ph("ph-source")
    target = _sentinel_ph("ph-target")
    target = dataclasses.replace(target, active_cameras=frozenset(["cam-target-only"]))
    await repo.save(source)
    await repo.save(target)

    await repo.merge(
        source_ph_id="ph-source", target_ph_id="ph-target", actor="admin", reason="test"
    )
    after = await repo.get("ph-source")
    assert after is not None
    assert after.closed_at is not None
    assert after.metadata["merged_into_ph_id"] == "ph-target"
    _assert_carryover(source, after, _MERGE_SOURCE_CHANGED)


_SPLIT_EARLIER_CHANGED = frozenset({"last_seen_at", "observation_count", "closed_at"})
# closed_at is NOT in the later half's changed set: the site passes
# closed_at=None, which equals the original (open) ph.closed_at -- a
# no-op reassertion of "still open", not an actual change from the source PH.
_SPLIT_LATER_CHANGED = frozenset({"ph_id", "born_at", "last_seen_at", "observation_count"})


def _floor(x_m: float, y_m: float) -> FloorPoint:
    return FloorPoint(x_mm=int(x_m * 1000), y_mm=int(y_m * 1000), calibrated=True)


def _obs_for_split(captured_at: datetime) -> WorldObservation:
    return WorldObservation(
        camera_id=_CAM,
        frame_index=0,
        captured_at=captured_at,
        floor_point=_floor(1.0, 1.0),
        bbox=BoundingBox(x_min=10, y_min=20, x_max=110, y_max=220),
        embedding=[],
        detection_confidence=0.9,
    )


@pytest.mark.asyncio
async def test_split_both_halves_carry_ph_state_correctly() -> None:
    repo = InMemoryPHRepository()
    ph = _sentinel_ph("ph-1")
    await repo.save(ph)

    obs_repo = InMemoryWorldObservationRepository()
    t1 = _T0 - timedelta(seconds=10)
    t2 = _T0 - timedelta(seconds=5)
    await obs_repo.save(_obs_for_split(t1), "ph-1")
    oid2 = await obs_repo.save(_obs_for_split(t2), "ph-1")
    # Inject the persisted observations into the PH repo's internal list --
    # split reads its own repository-local observation index, not the
    # separate WorldObservationRepository (matches the existing test convention
    # in tests/storage/postgres/test_ph_merge_split_validation.py).
    repo._observations["ph-1"] = await obs_repo.list_by_ph("ph-1", limit=10)

    earlier_id, later_id = await repo.split(
        "ph-1", at_observation_id=oid2, actor="admin", reason="test"
    )
    assert earlier_id == "ph-1"
    earlier = await repo.get(earlier_id)
    later = await repo.get(later_id)
    assert earlier is not None
    assert later is not None

    assert earlier.observation_count == 1
    assert earlier.closed_at is not None
    _assert_carryover(ph, earlier, _SPLIT_EARLIER_CHANGED)

    assert later.ph_id == later_id
    assert later.observation_count == 1
    assert later.closed_at is None
    _assert_carryover(ph, later, _SPLIT_LATER_CHANGED)


# ---------------------------------------------------------------------------
# Tracker-level sentinel survival: the matched-PH update site (site 1), the
# hottest path and the one that previously silently reset last_posture and
# closed_at to their dataclass defaults on every successful match.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_matched_update_survives_sentinel_metadata_last_posture_and_evidence_clock() -> None:
    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()
    tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo, config=WorldTrackerConfig())

    ph = _sentinel_ph("ph-1")
    await ph_repo.save(ph)
    assert ph.gallery_mean is not None

    # Observation at the PH's exact position with a matching embedding so the
    # geometric + appearance gate passes cleanly and this update is a MATCH,
    # not a spawn (a spawn would prove nothing about this site).
    obs = WorldObservation(
        camera_id=_CAM,
        frame_index=1,
        captured_at=_T0 + timedelta(seconds=1),
        floor_point=_floor(ph.state_mean[0], ph.state_mean[1]),
        bbox=BoundingBox(x_min=10, y_min=20, x_max=110, y_max=220),
        embedding=ph.gallery_mean,
        detection_confidence=0.9,
        detection_id="det-1",
        quality=0.8,
    )

    await tracker.step(observations=[obs], now=obs.captured_at, low_band_observations=[])

    open_phs = await ph_repo.list_open()
    assert len(open_phs) == 1, "observation must match the existing PH, not spawn a new one"
    matched = open_phs[0]
    assert matched.ph_id == "ph-1"

    # Fields the matched-update site does not own must survive unchanged.
    assert matched.last_posture == "sitting"
    assert matched.metadata.get("sentinel_key") == "sentinel_value"
    assert matched.current_identity_id == "sentinel-identity"
    assert matched.current_identity_committed_at == ph.current_identity_committed_at
    assert matched.last_independent_identity_evidence_at == ph.last_independent_identity_evidence_at
    assert matched.closed_at is None
    # Sanity: the site does own these -- prove it actually advanced them.
    assert matched.last_seen_at == obs.captured_at
    assert matched.observation_count == ph.observation_count + 1

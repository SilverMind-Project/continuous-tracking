"""Unit tests for the low-confidence detection recovery second association pass (M2.3).

Tests:
  2. open PH + low-band obs inside tight gate → PH updated, observation_count unchanged,
     gallery_mean unchanged.
  3. low-band obs outside gate → dropped.
  4. low-band obs with no open PH nearby → never spawns (counter increments).
  5. guardrail: a stale PH (ghost) is NOT immortalised — it closes within
     ph_close_grace_s when low-band obs fall outside the recovery gate.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.domain import BoundingBox, FloorPoint, PersonHypothesis, WorldObservation
from app.storage.base import InMemoryPHRepository, InMemoryWorldObservationRepository
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.tracker import WorldTracker

_T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
_CAM = "cam-1"


def _floor(x_m: float, y_m: float, *, calibrated: bool = True) -> FloorPoint:
    return FloorPoint(x_mm=int(x_m * 1000), y_mm=int(y_m * 1000), calibrated=calibrated)


def _obs(
    *,
    x_m: float = 1.0,
    y_m: float = 1.0,
    confidence: float = 0.9,
    captured_at: datetime | None = None,
    embedding: list[float] | None = None,
) -> WorldObservation:
    return WorldObservation(
        camera_id=_CAM,
        frame_index=0,
        captured_at=captured_at or _T0,
        floor_point=_floor(x_m, y_m),
        bbox=BoundingBox(x_min=100, y_min=50, x_max=300, y_max=400),
        embedding=embedding or [0.1] * 128,
        detection_confidence=confidence,
        detection_id="",
        quality=0.7,
    )


def _low_obs(
    x_m: float = 1.0,
    y_m: float = 1.0,
    *,
    captured_at: datetime | None = None,
) -> WorldObservation:
    """Minimal low-band observation (no embedding, no face)."""
    return WorldObservation(
        camera_id=_CAM,
        frame_index=0,
        captured_at=captured_at or _T0,
        floor_point=_floor(x_m, y_m),
        bbox=BoundingBox(x_min=100, y_min=50, x_max=300, y_max=400),
        embedding=[],
        detection_confidence=0.4,
        detection_id="",
        quality=0.0,
    )


def _ph(
    ph_id: str = "ph-1",
    *,
    x_m: float = 1.0,
    y_m: float = 1.0,
    observation_count: int = 5,
    gallery_mean: list[float] | None = None,
) -> PersonHypothesis:
    return PersonHypothesis(
        ph_id=ph_id,
        state_mean=(x_m, y_m, 0.0, 0.0),
        state_cov=(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.01,
            0.0,
            0.0,
            0.0,
            0.0,
            0.01,
        ),
        born_at=_T0 - timedelta(seconds=10),
        last_seen_at=_T0,
        last_seen_camera=_CAM,
        observation_count=observation_count,
        current_identity_id=None,
        gallery_mean=gallery_mean or [0.2] * 128,
        height_estimate_m=1.7,
        active_cameras=frozenset([_CAM]),
        last_floor_speed_m_s=0.0,
        mean_quality=0.7,
    )


def _make_tracker(cfg: WorldTrackerConfig) -> tuple[WorldTracker, InMemoryPHRepository]:
    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()
    tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo, config=cfg)
    return tracker, ph_repo


_RECOVERY_CFG = WorldTrackerConfig(
    gate_chi2=9.21,
    ph_close_grace_s=5.0,
    min_observations_to_publish=1,
    enable_low_confidence_recovery=True,
    low_confidence_floor=0.25,
    recovery_gate_chi2=5.99,
)

_NO_RECOVERY_CFG = WorldTrackerConfig(
    gate_chi2=9.21,
    ph_close_grace_s=5.0,
    min_observations_to_publish=1,
    enable_low_confidence_recovery=False,
)


# ---------------------------------------------------------------------------
# Test 2: PH inside tight gate → Kalman+last_seen_at update; count unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_low_band_match_updates_state_not_evidence() -> None:
    """A low-band obs close to an open PH updates Kalman state and last_seen_at
    but does NOT increment observation_count, gallery_mean, or view_prototypes."""
    tracker, ph_repo = _make_tracker(_RECOVERY_CFG)

    ph = _ph("ph-1", x_m=1.0, y_m=1.0, observation_count=5, gallery_mean=[0.2] * 128)
    await ph_repo.save(ph)

    # No high-band obs for this frame (simulates person lying down = low confidence).
    t1 = _T0 + timedelta(seconds=1)
    lb = _low_obs(x_m=1.05, y_m=1.05, captured_at=t1)

    await tracker.step(
        observations=[],
        now=t1,
        low_band_observations=[lb],
    )

    open_phs = await ph_repo.list_open()
    assert len(open_phs) == 1, "PH must stay open after low-band recovery"
    survived = open_phs[0]

    assert survived.observation_count == 5, "observation_count must NOT increment in low-band pass"
    assert survived.gallery_mean == [0.2] * 128, "gallery_mean must NOT change in low-band pass"
    assert survived.last_seen_at == t1, "last_seen_at must be updated to obs capture time"


# ---------------------------------------------------------------------------
# Test 3: obs outside tight gate → PH unmatched, closes on grace
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_low_band_obs_outside_gate_not_matched() -> None:
    """A low-band obs geometrically far from the PH (Mahalanobis² > recovery_gate_chi2)
    is not matched in the second pass.  We use a very short elapsed time (0.2s) so
    Kalman covariance stays small and the gate remains tight."""
    tracker, ph_repo = _make_tracker(_RECOVERY_CFG)

    ph = _ph("ph-1", x_m=1.0, y_m=1.0)
    await ph_repo.save(ph)

    # 0.2 s elapsed → covariance barely grows; obs 5 m away → Maha² >> recovery_gate_chi2.
    t1 = _T0 + timedelta(seconds=0.2)
    lb_far = _low_obs(x_m=5.0, y_m=5.0, captured_at=t1)

    await tracker.step(
        observations=[],
        now=t1,
        low_band_observations=[lb_far],
    )

    open_phs = await ph_repo.list_open()
    # PH is still within grace period (0.2s < 5.0s grace); it should be open but UNMATCHED.
    assert len(open_phs) == 1
    survived = open_phs[0]
    # observation_count unchanged because the obs was outside the gate.
    assert survived.observation_count == 5
    # last_seen_at NOT updated (obs was not matched).
    assert survived.last_seen_at == _T0


# ---------------------------------------------------------------------------
# Test 4: no open PH nearby → low-band obs never spawns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_low_band_obs_with_no_nearby_ph_does_not_spawn() -> None:
    """Unmatched low-band observations must NEVER spawn a new PH."""
    tracker, ph_repo = _make_tracker(_RECOVERY_CFG)
    # No PHs exist.

    lb = _low_obs(x_m=5.0, y_m=5.0)
    await tracker.step(
        observations=[],
        now=_T0,
        low_band_observations=[lb],
    )

    open_phs = await ph_repo.list_open()
    assert len(open_phs) == 0, "low-band obs must never spawn a new PH"


# ---------------------------------------------------------------------------
# Test 5: guardrail — ghost PH closes on time when no recovery match
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ghost_ph_closes_within_grace_without_match() -> None:
    """A PH with no matching obs (high or low) closes once ph_close_grace_s expires.

    The low-band recovery must not immortalise a ghost PH whose low-band obs
    is spatially outside the tight gate.
    """
    tracker, ph_repo = _make_tracker(_RECOVERY_CFG)

    ph = _ph("ph-ghost", x_m=1.0, y_m=1.0)
    await ph_repo.save(ph)

    # Low-band obs in a completely different location — should never match.
    grace = _RECOVERY_CFG.ph_close_grace_s
    t_close = _T0 + timedelta(seconds=grace + 1)
    lb_far = _low_obs(x_m=50.0, y_m=50.0, captured_at=t_close)

    await tracker.step(
        observations=[],
        now=t_close,
        low_band_observations=[lb_far],
    )

    open_phs = await ph_repo.list_open()
    assert len(open_phs) == 0, (
        "Ghost PH must close within ph_close_grace_s even with low-band obs present"
    )


# ---------------------------------------------------------------------------
# Test: flag off → low_band_observations ignored (no second pass)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_off_low_band_obs_ignored() -> None:
    """When enable_low_confidence_recovery=False, passing low_band_observations
    has no effect — unmatched PH closes as normal."""
    tracker, ph_repo = _make_tracker(_NO_RECOVERY_CFG)

    ph = _ph("ph-1", x_m=1.0, y_m=1.0)
    await ph_repo.save(ph)

    t_close = _T0 + timedelta(seconds=_NO_RECOVERY_CFG.ph_close_grace_s + 1)
    lb = _low_obs(x_m=1.05, y_m=1.05, captured_at=t_close)

    await tracker.step(
        observations=[],
        now=t_close,
        low_band_observations=[lb],
    )

    # Flag is off: low-band obs is not used, so PH closes normally.
    open_phs = await ph_repo.list_open()
    assert len(open_phs) == 0, "PH must close when flag is off even if low-band obs is nearby"

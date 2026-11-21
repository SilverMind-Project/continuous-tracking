"""Unit tests for HeightProfileLearner and CCConfigSyncService operator height sync."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from app.calibration.state import CalibrationState
from app.domain import IdentityHeightProfile
from app.services.camera_room_map import CameraRoomMap, RoomPolygonMap
from app.services.cc_config_sync import PERSON_HEIGHTS_PATH, CCConfigSyncService
from app.services.height_profile_learner import (
    AttestedHeightSample,
    HeightProfileLearner,
    HeightProfileLearnerConfig,
    InMemoryHeightSampleSource,
    compute_robust_height_stats,
)
from app.services.transit_zone_map import TransitZoneMap
from app.storage.height_profiles import InMemoryIdentityHeightProfileRepository


def _make_samples(
    identity_id: str,
    *,
    count: int,
    days: int,
    cameras: list[str],
    base_time: datetime,
    heights: list[float] | None = None,
) -> list[AttestedHeightSample]:
    rows: list[AttestedHeightSample] = []
    for i in range(count):
        day_offset = i % days
        cam = cameras[i % len(cameras)]
        h = heights[i] if heights is not None else 1.62 + ((i % 5) - 2) * 0.005
        rows.append(
            AttestedHeightSample(
                identity_id=identity_id,
                camera_id=cam,
                observed_at=base_time - timedelta(days=day_offset, seconds=i + 1),
                height_m=h,
                height_sigma_m=0.05,
            )
        )
    return rows


def test_compute_robust_height_stats_uses_mad_and_floor() -> None:
    # Tight cluster -> floored at 0.03 m.
    med, sigma = compute_robust_height_stats([1.60] * 100 + [1.605] * 100, min_sigma_m=0.03)
    assert med == pytest.approx(1.6025)
    assert sigma == pytest.approx(0.03)

    # Wider spread + gross outliers -> median & MAD stay robust.
    vals = [1.55] * 50 + [1.60] * 100 + [1.65] * 50 + [2.15] * 5
    med2, sigma2 = compute_robust_height_stats(vals, min_sigma_m=0.03)
    assert med2 == pytest.approx(1.60)
    # 105 deviations are >= 0.05 out of 205 -> MAD = 0.05 -> 1.4826 * 0.05 = 0.07413
    assert sigma2 == pytest.approx(1.4826 * 0.05, rel=0.01)


@pytest.mark.asyncio
async def test_learner_enforces_min_samples_days_and_cameras() -> None:
    now = datetime(2026, 11, 21, 12, 0, 0, tzinfo=UTC)
    samples = [
        *_make_samples("few-samples", count=199, days=2, cameras=["c1", "c2"], base_time=now),
        *_make_samples("one-day", count=200, days=1, cameras=["c1", "c2"], base_time=now),
        *_make_samples("one-cam", count=200, days=2, cameras=["c1"], base_time=now),
        *_make_samples("qualified", count=200, days=2, cameras=["c1", "c2"], base_time=now),
    ]
    repo = InMemoryIdentityHeightProfileRepository()
    invalidated = 0

    def _on_update() -> None:
        nonlocal invalidated
        invalidated += 1

    learner = HeightProfileLearner(
        sample_source=InMemoryHeightSampleSource(samples),
        profile_repo=repo,
        config=HeightProfileLearnerConfig(min_samples=200, min_days=2, min_cameras=2),
        on_profiles_updated=_on_update,
    )

    result = await learner.run_once(now=now)
    assert result.updated_identities == ("qualified",)
    assert result.skipped_identities == {
        "few-samples": "insufficient_samples",
        "one-day": "insufficient_days",
        "one-cam": "insufficient_cameras",
    }
    assert invalidated == 1

    profile = await repo.get("qualified")
    assert profile is not None
    assert profile.source == "learned"
    assert profile.samples == 200
    assert profile.cameras == 2
    assert profile.height_m == pytest.approx(1.62, abs=0.005)
    assert profile.height_sigma_m == pytest.approx(0.03)
    assert await repo.get("few-samples") is None


@pytest.mark.asyncio
async def test_learned_profile_replaces_operator_profile_and_sync_respects_precedence() -> None:
    now = datetime(2026, 11, 21, 12, 0, 0, tzinfo=UTC)
    repo = InMemoryIdentityHeightProfileRepository()

    class _FakeCCClient:
        def __init__(self, payload: list[dict[str, Any]]) -> None:
            self.payload = payload

        async def get(self, path: str) -> Any:
            if path == PERSON_HEIGHTS_PATH:
                return self.payload
            return None

    cc = _FakeCCClient(
        [
            {"person_id": "alice", "height_cm": 158.0},
            {"person_id": "bob", "height_cm": 182.0},
            {"person_id": "carol", "height_cm": None},
        ]
    )
    invalidations = 0

    def _bump() -> None:
        nonlocal invalidations
        invalidations += 1

    sync = CCConfigSyncService(
        client=cc,  # type: ignore[arg-type]
        calibration_state=CalibrationState(),
        camera_room_map=CameraRoomMap(),
        room_polygon_map=RoomPolygonMap(),
        transit_zone_map=TransitZoneMap(),
        height_profile_repo=repo,
        on_height_profiles_updated=_bump,
    )

    # First sync writes operator profiles for alice and bob; carol has none.
    await sync._sync_person_heights(now=now)
    alice = await repo.get("alice")
    bob = await repo.get("bob")
    assert (
        alice is not None and alice.source == "operator" and alice.height_m == pytest.approx(1.58)
    )
    assert bob is not None and bob.source == "operator" and bob.height_m == pytest.approx(1.82)
    assert await repo.get("carol") is None
    assert invalidations == 1

    # Learner promotes alice to a learned profile (1.54 m).
    learner = HeightProfileLearner(
        sample_source=InMemoryHeightSampleSource(
            _make_samples(
                "alice",
                count=220,
                days=3,
                cameras=["c1", "c2"],
                base_time=now,
                heights=[1.54] * 220,
            )
        ),
        profile_repo=repo,
        config=HeightProfileLearnerConfig(min_samples=200, min_days=2, min_cameras=2),
    )
    res = await learner.run_once(now=now)
    assert res.updated_identities == ("alice",)
    alice_learned = await repo.get("alice")
    assert alice_learned is not None
    assert alice_learned.source == "learned"
    assert alice_learned.height_m == pytest.approx(1.54)

    # Subsequent CC sync with edited alice operator height & cleared bob height:
    # - alice MUST keep her learned profile (1.54 m).
    # - bob's operator profile MUST be deleted because CC cleared height_cm.
    cc.payload = [
        {"person_id": "alice", "height_cm": 165.0},
        {"person_id": "bob", "height_cm": None},
    ]
    await sync._sync_person_heights(now=now + timedelta(minutes=1))
    alice_after = await repo.get("alice")
    assert alice_after == IdentityHeightProfile(
        identity_id="alice",
        height_m=1.54,
        height_sigma_m=0.03,
        source="learned",
        samples=220,
        cameras=2,
        updated_at=now,
    )
    assert await repo.get("bob") is None

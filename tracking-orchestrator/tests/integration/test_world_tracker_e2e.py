"""WorldTracker end-to-end replay tests (C1, C2) + hallway-bathroom proof.

Replays synthetic WorldObservation fixtures through the real WorldTracker
pipeline (Postgres-backed repos) and asserts correct PH lifecycle.

Fixture format: length-prefixed JSON binary. Each chunk = one frame step.
See scripts/synthesize_replay_fixture.py for generation.

Marked @pytest.mark.integration; CI selects this marker. Testcontainer
Postgres is started by the session fixture in tests/conftest.py.

 C1 (test_single_ph_covers_both_cameras) is now a normal passing test.
The cross-camera dedup pass lands in U1 and collapses the two simultaneous
observations from overlapping cameras into one PH before the association step.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from prometheus_client import CollectorRegistry, Counter

from app.domain import BoundingBox, FloorPoint, WorldObservation
from app.observability.metrics import build_metrics
from tests.integration._replay import _ROOM_POLYGONS, FIXTURES_DIR, load_fixture


def _counter_total(counter: Counter) -> float:
    return sum(
        sample.value
        for metric in counter.collect()
        for sample in metric.samples
        if sample.name.endswith("_total")
    )


@pytest.mark.integration
class TestWorldTrackerE2EOnePersonTwoCameras:
    """C1: one person under two overlapping cameras produces exactly one PH."""

    @pytest.mark.asyncio
    async def test_single_ph_covers_both_cameras(self, db_pool: Any) -> None:
        """After replaying the two-camera fixture, exactly one PH exists."""
        fixture = FIXTURES_DIR / "two_cameras_one_room.bin"
        assert fixture.exists(), "Fixture missing: run scripts/synthesize_replay_fixture.py"

        from app.storage.postgres.ph_repo import (
            PostgresPHRepository,
            PostgresWorldObservationRepository,
        )
        from app.tracking.world.tracker import WorldTracker

        ph_repo = PostgresPHRepository(db_pool)
        obs_repo = PostgresWorldObservationRepository(db_pool)
        tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo)

        steps = load_fixture(fixture)
        assert len(steps) >= 2, "Fixture must have at least 2 steps"

        # Verify fixture spans at least 2 cameras (fixture integrity check T6).
        camera_ids_in_fixture: set[str] = {obs.camera_id for step in steps for obs in step}
        assert len(camera_ids_in_fixture) >= 2, (
            f"Fixture must span >=2 cameras, got {camera_ids_in_fixture}"
        )

        base_time = datetime(2026, 5, 28, 9, 0, 0, tzinfo=UTC)
        for i, frame_obs in enumerate(steps):
            now = base_time + timedelta(seconds=i * 0.5)
            await tracker.step(observations=frame_obs, now=now, room_polygons=_ROOM_POLYGONS)

        phs, total = await ph_repo.list_active(include_transient=True)
        assert total == 1, (
            f"C1: expected exactly 1 PH after one-person two-camera replay, got {total}"
        )

        ph = phs[0]
        observations, obs_total = await ph_repo.get_observations(ph.ph_id, limit=1000)
        assert obs_total > 0, "PH must have at least one observation"

        obs_camera_ids = {o.camera_id for o in observations}
        assert len(obs_camera_ids) >= 2, (
            f"C1: PH must have observations from both cameras, got {obs_camera_ids}"
        )

    @pytest.mark.asyncio
    async def test_ph_lifecycle_not_interrupted_by_camera_handoff(self, db_pool: Any) -> None:
        """The original PH accumulates observations from both cameras.

        Even with the architectural gap (two simultaneous observations spawn a
        second PH), the FIRST PH does receive observations from both cam-1 and
        cam-2 during the overlap phase, because one of the two cam-2 observations
        is matched to the existing PH rather than spawning a new one.
        """
        fixture = FIXTURES_DIR / "two_cameras_one_room.bin"
        assert fixture.exists(), "Fixture missing: run scripts/synthesize_replay_fixture.py"

        from app.storage.postgres.ph_repo import (
            PostgresPHRepository,
            PostgresWorldObservationRepository,
        )
        from app.tracking.world.tracker import WorldTracker

        ph_repo = PostgresPHRepository(db_pool)
        obs_repo = PostgresWorldObservationRepository(db_pool)
        tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo)

        steps = load_fixture(fixture)
        base_time = datetime(2026, 5, 28, 9, 0, 0, tzinfo=UTC)
        for i, frame_obs in enumerate(steps):
            now = base_time + timedelta(seconds=i * 0.5)
            await tracker.step(observations=frame_obs, now=now, room_polygons=_ROOM_POLYGONS)

        phs, _ = await ph_repo.list_active(include_transient=True)
        assert len(phs) >= 1, "At least one PH must exist after replay"

        ph = phs[0]
        observations, _ = await ph_repo.get_observations(ph.ph_id, limit=1000)

        # Observations from cam-1 and cam-2 must appear in one PH.
        obs_camera_ids = {o.camera_id for o in observations}
        assert "cam-1" in obs_camera_ids, "PH must include cam-1 observations"
        assert "cam-2" in obs_camera_ids, "PH must include cam-2 observations"
        assert ph.closed_at is None, "PH must still be open at end of replay"


@pytest.mark.integration
class TestWorldTrackerE2ETwoPeopleTwoRooms:
    """C2: two people in separate rooms produce two distinct PHs that never merge."""

    @pytest.mark.asyncio
    async def test_two_phs_do_not_merge(self, db_pool: Any) -> None:
        """After replaying the two-rooms fixture, exactly two distinct PHs exist."""
        fixture = FIXTURES_DIR / "two_rooms_two_people.bin"
        assert fixture.exists(), "Fixture missing: run scripts/synthesize_replay_fixture.py"

        from app.storage.postgres.ph_repo import (
            PostgresPHRepository,
            PostgresWorldObservationRepository,
        )
        from app.tracking.world.tracker import WorldTracker

        ph_repo = PostgresPHRepository(db_pool)
        obs_repo = PostgresWorldObservationRepository(db_pool)
        tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo)

        steps = load_fixture(fixture)
        assert len(steps) >= 2, "Fixture must have at least 2 steps"

        camera_ids_in_fixture = {obs.camera_id for step in steps for obs in step}
        assert len(camera_ids_in_fixture) >= 2, (
            f"Fixture must span >=2 cameras, got {camera_ids_in_fixture}"
        )

        base_time = datetime(2026, 5, 28, 9, 0, 0, tzinfo=UTC)
        for i, frame_obs in enumerate(steps):
            now = base_time + timedelta(seconds=i * 0.5)
            await tracker.step(observations=frame_obs, now=now, room_polygons=_ROOM_POLYGONS)

        phs, total = await ph_repo.list_active(include_transient=True)
        assert total == 2, f"C2: expected exactly 2 PHs for two people in two rooms, got {total}"

        ph_ids = {ph.ph_id for ph in phs}
        assert len(ph_ids) == 2, "C2: the two PHs must have distinct IDs"

        # Each PH's observations must come from only one camera.
        for ph in phs:
            obs, _ = await ph_repo.get_observations(ph.ph_id, limit=1000)
            cam_ids = {o.camera_id for o in obs}
            assert len(cam_ids) == 1, (
                f"C2: PH {ph.ph_id} should observe only one camera, got {cam_ids}"
            )


@pytest.mark.integration
class TestHallwayBathroomDoor:
    """senior-safety proof — hallway + doorway camera at a bathroom door."""

    @pytest.mark.asyncio
    async def test_hallway_bathroom_one_person(self, db_pool: Any) -> None:
        """Exactly one PH throughout the visible phases of the hallway-bathroom fixture.

        The hallway camera and the doorway camera both see one senior at the bathroom
        door simultaneously (steps 10-19). The cross-camera dedup pass (U1) must
        collapse those observations to a single PH. The bathroom-blind interval
        (steps 20-49, empty frames) does not spawn a phantom second PH.
        """
        fixture = FIXTURES_DIR / "hallway_bathroom_door.bin"
        assert fixture.exists(), "Fixture missing: run scripts/synthesize_replay_fixture.py"

        from app.storage.postgres.ph_repo import (
            PostgresPHRepository,
            PostgresWorldObservationRepository,
        )
        from app.tracking.world.tracker import WorldTracker

        ph_repo = PostgresPHRepository(db_pool)
        obs_repo = PostgresWorldObservationRepository(db_pool)
        tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo)

        steps = load_fixture(fixture)
        base_time = datetime(2026, 5, 28, 9, 0, 0, tzinfo=UTC)
        for i, frame_obs in enumerate(steps):
            now = base_time + timedelta(seconds=i * 0.5)
            await tracker.step(observations=frame_obs, now=now, room_polygons=_ROOM_POLYGONS)

        phs, total = await ph_repo.list_active(include_transient=True)
        assert total == 1, (
            f"U1 hallway-bathroom: expected exactly 1 PH, got {total}. "
            "The dedup pass must collapse overlapping camera observations."
        )

        # The single PH must have observations from both cameras.
        ph = phs[0]
        obs_list, _ = await ph_repo.get_observations(ph.ph_id, limit=1000)
        cam_ids = {o.camera_id for o in obs_list}
        assert "cam-hall" in cam_ids, "hallway camera observations must be on the PH"
        assert "cam-door" in cam_ids, "doorway camera observations must be on the PH"

        # T9 Postgres: quality field round-trips through the DB (all fixture observations
        # have quality=0.5; they must not silently revert to 0.0 after save+load).
        assert all(o.quality > 0.0 for o in obs_list), (
            "T9-Postgres: quality must survive a Postgres save+load (fixture sets quality=0.5)"
        )


@pytest.mark.integration
class TestUncalibratedSpawnWithRoomPolygons:
    """uncalibrated cameras spawn PHs even when room polygons exist."""

    @pytest.mark.asyncio
    async def test_uncalibrated_camera_spawns_phs_with_room_polygons(
        self,
        db_pool: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from app.observability import metrics as metrics_pkg
        from app.storage.postgres.ph_repo import (
            PostgresPHRepository,
            PostgresWorldObservationRepository,
        )
        from app.tracking.world import tracker as tracker_mod
        from app.tracking.world.tracker import WorldTracker

        fresh_metrics = build_metrics(registry=CollectorRegistry())
        monkeypatch.setattr(metrics_pkg, "metrics", fresh_metrics)
        monkeypatch.setattr(tracker_mod._metrics, "metrics", fresh_metrics)

        ph_repo = PostgresPHRepository(db_pool)
        obs_repo = PostgresWorldObservationRepository(db_pool)
        tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo)
        now = datetime(2026, 5, 28, 10, 0, 0, tzinfo=UTC)

        obs = WorldObservation(
            camera_id="uncalibrated-cam",
            frame_index=1,
            captured_at=now,
            floor_point=FloorPoint(x_mm=200_000, y_mm=200_000, calibrated=False),
            bbox=BoundingBox(x_min=10, y_min=20, x_max=110, y_max=220),
            embedding=[1.0, 0.0, 0.0, 0.0],
            detection_confidence=0.90,
            detection_id="det-uncal-1",
            quality=0.50,
        )

        await tracker.step(observations=[obs], now=now, room_polygons=_ROOM_POLYGONS)

        phs, total = await ph_repo.list_active(include_transient=True)
        assert total == 1
        assert phs[0].last_seen_camera == "uncalibrated-cam"
        assert _counter_total(fresh_metrics.world_tracker_ph_spawned_total) == 1.0
        assert _counter_total(fresh_metrics.world_tracker_spawn_rejected_out_of_room_total) == 0.0
        shadow_sample = next(
            sample
            for metric in fresh_metrics.identity_shadow_mismatch_total.collect()
            for sample in metric.samples
            if sample.labels.get("feature") == "uncalibrated_spawn"
            and sample.name.endswith("_total")
        )
        assert shadow_sample.value == 1.0

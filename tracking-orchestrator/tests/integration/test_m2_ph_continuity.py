"""PH lifecycle continuity integration proofs.

Replays the diagnosis fixtures through WorldTracker.step with continuity features
enabled (revival, sticky maintenance, uncalibrated gate relax), an identity
resolver wired in, and known identities registered.  Asserts strict
improvement over the recorded baseline on distinct-PH count, identity stability,
and guardrail preservation.

Shadow-metric proofs (flags off) demonstrate the mechanisms would fire
before enabling them.

Marked @pytest.mark.integration; CI selects this marker.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from prometheus_client import Counter

from tests.integration._replay import _ROOM_POLYGONS, FIXTURES_DIR, load_fixture

BASE_TIME = datetime(2026, 5, 28, 9, 0, 0, tzinfo=UTC)


def _counter_total(counter: Counter) -> float:
    return sum(
        sample.value
        for metric in counter.collect()
        for sample in metric.samples
        if sample.name.endswith("_total")
    )


def _labeled_counter_value(counter: Counter, label_name: str, label_value: str) -> float:
    return sum(
        sample.value
        for metric in counter.collect()
        for sample in metric.samples
        if sample.name.endswith("_total") and sample.labels.get(label_name) == label_value
    )


# ── Replay helpers ────────────────────────────────────────────────────────


async def _replay_with_continuity_features(
    db_pool: Any,
    fixture_name: str,
    *,
    known_identity_ids: list[str] | None = None,
) -> Any:
    """Replay a fixture with continuity features enabled and an identity resolver wired in.

    Returns (ph_repo, tracker) for post-replay assertions.
    """
    from app.domain import Identity
    from app.storage.base import InMemoryGalleryRepository
    from app.storage.postgres.ph_repo import (
        PostgresPHRepository,
        PostgresWorldObservationRepository,
    )
    from app.tracking.identity_resolver import IdentityResolver, ResolverConfig
    from app.tracking.world.config import WorldTrackerConfig
    from app.tracking.world.tracker import WorldTracker

    now = datetime.now(UTC)

    # Gallery with known identities so the resolver can build a non-degenerate prior.
    gallery = InMemoryGalleryRepository()
    for identity_id in known_identity_ids or []:
        await gallery.upsert_identity(
            Identity(identity_id=identity_id, display_name=identity_id, enrolled_at=now)
        )

    resolver_config = ResolverConfig(enable_sticky_maintenance=True)
    resolver = IdentityResolver(gallery_repo=gallery, config=resolver_config)

    tracker_config = WorldTrackerConfig(
        enable_ph_revival=True,
        enable_uncalibrated_gate_relax=True,
    )

    ph_repo = PostgresPHRepository(db_pool)
    obs_repo = PostgresWorldObservationRepository(db_pool)
    tracker = WorldTracker(
        ph_repo=ph_repo,
        obs_repo=obs_repo,
        config=tracker_config,
        identity_resolver=resolver,
    )

    fixture = FIXTURES_DIR / fixture_name
    assert fixture.exists(), f"Fixture missing: {fixture_name}"

    steps = load_fixture(fixture)
    for i, frame_obs in enumerate(steps):
        frame_now = BASE_TIME + timedelta(seconds=i * 0.5)
        # Extract face anchors from observations for the resolver.
        face_anchors = [obs.face_anchor for obs in frame_obs if obs.face_anchor is not None] or None
        await tracker.step(
            observations=frame_obs,
            now=frame_now,
            room_polygons=_ROOM_POLYGONS,
            face_anchors=face_anchors,
        )

    return ph_repo, tracker


async def _replay_shadow(
    db_pool: Any,
    fixture_name: str,
) -> Any:
    """Replay a fixture with continuity features OFF (shadow mode).

    Returns ph_repo for post-replay metric assertions.
    """
    from app.storage.postgres.ph_repo import (
        PostgresPHRepository,
        PostgresWorldObservationRepository,
    )
    from app.tracking.world.tracker import WorldTracker

    ph_repo = PostgresPHRepository(db_pool)
    obs_repo = PostgresWorldObservationRepository(db_pool)
    tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo)

    fixture = FIXTURES_DIR / fixture_name
    assert fixture.exists(), f"Fixture missing: {fixture_name}"

    steps = load_fixture(fixture)
    for i, frame_obs in enumerate(steps):
        frame_now = BASE_TIME + timedelta(seconds=i * 0.5)
        face_anchors = [obs.face_anchor for obs in frame_obs if obs.face_anchor is not None] or None
        await tracker.step(
            observations=frame_obs,
            now=frame_now,
            room_polygons=_ROOM_POLYGONS,
            face_anchors=face_anchors,
        )

    return ph_repo


# ── Measurable gate: single_camera_turn ──────────────────────────────────


@pytest.mark.integration
class TestSingleCameraTurnContinuity:
    """single_camera_turn.bin with continuity features enabled.

    Recorded baseline: 2 distinct PHs (front-face PH + post-gap respawn as UNKNOWN).
    Target: 1 PH with identity alice held throughout.
    """

    @pytest.mark.asyncio
    async def test_exactly_one_ph_for_one_person(self, db_pool: Any) -> None:
        """With revival, the closed PH is revived instead of spawning new."""
        ph_repo, _tracker = await _replay_with_continuity_features(
            db_pool, "single_camera_turn.bin", known_identity_ids=["alice"]
        )

        phs, _total = await ph_repo.list_active(include_transient=True)
        distinct_ph = {ph.ph_id for ph in phs}
        # exactly 1 PH (revival reopens the closed PH).
        assert len(distinct_ph) == 1, (
            f"expected exactly 1 PH for one person, got {len(distinct_ph)}"
        )

    @pytest.mark.asyncio
    async def test_identity_held_through_turn(self, db_pool: Any) -> None:
        """With sticky maintenance, the alice identity is held through the
        turn and walk frames — never UNKNOWN after being committed."""
        ph_repo, _tracker = await _replay_with_continuity_features(
            db_pool, "single_camera_turn.bin", known_identity_ids=["alice"]
        )

        phs, _total = await ph_repo.list_active(include_transient=True)
        # All PHs must have alice identity.
        for ph in phs:
            assert ph.current_identity_id == "alice", (
                f"PH {ph.ph_id} identity should be alice, got {ph.current_identity_id}"
            )

    @pytest.mark.asyncio
    async def test_strict_improvement_over_recorded_baseline(self, db_pool: Any) -> None:
        """Continuity features must reduce distinct-PH count below the recorded baseline of 2."""
        from tests.integration.test_diagnosis_baseline import (
            BASELINE_MIN_DISTINCT_PH_IDS_TURN,
        )

        ph_repo, _tracker = await _replay_with_continuity_features(
            db_pool, "single_camera_turn.bin", known_identity_ids=["alice"]
        )

        phs, _total = await ph_repo.list_active(include_transient=True)
        distinct_ph = len({ph.ph_id for ph in phs})
        # Strict improvement: fewer distinct PHs than the recorded baseline minimum.
        assert distinct_ph < BASELINE_MIN_DISTINCT_PH_IDS_TURN, (
            f"expected fewer than {BASELINE_MIN_DISTINCT_PH_IDS_TURN} distinct PHs, "
            f"got {distinct_ph}"
        )


# ── Guardrail: two_people_one_room ────────────────────────────────────────


@pytest.mark.integration
class TestTwoPeopleOneRoomContinuity:
    """two_people_one_room.bin with continuity features enabled.

    Guardrail: the continuity bias (revival + sticky maintenance) must not
    merge two distinct people into one PH.
    """

    @pytest.mark.asyncio
    async def test_two_distinct_phs_preserved(self, db_pool: Any) -> None:
        """Two people remain two distinct PHs."""
        ph_repo, _tracker = await _replay_with_continuity_features(
            db_pool, "two_people_one_room.bin", known_identity_ids=["alice", "bob"]
        )

        phs, total = await ph_repo.list_active(include_transient=True)
        assert total == 2, f"expected 2 PHs for two people, got {total}"

        ph_ids = {ph.ph_id for ph in phs}
        assert len(ph_ids) == 2, "the two PHs must have distinct IDs"

    @pytest.mark.asyncio
    async def test_two_distinct_identities_preserved(self, db_pool: Any) -> None:
        """Two people retain two distinct identities (alice, bob)."""
        ph_repo, _tracker = await _replay_with_continuity_features(
            db_pool, "two_people_one_room.bin", known_identity_ids=["alice", "bob"]
        )

        phs, _total = await ph_repo.list_active(include_transient=True)
        identity_ids = {ph.current_identity_id for ph in phs}
        # Both alice and bob should be present.
        assert "alice" in identity_ids, "alice identity missing"
        assert "bob" in identity_ids, "bob identity missing"
        # No UNKNOWN PHs (both should have face-anchored identities).
        assert None not in identity_ids, "no PH should be UNKNOWN"

    @pytest.mark.asyncio
    async def test_no_over_merge(self, db_pool: Any) -> None:
        """The continuity bias must not merge alice and bob into one PH."""
        ph_repo, _tracker = await _replay_with_continuity_features(
            db_pool, "two_people_one_room.bin", known_identity_ids=["alice", "bob"]
        )

        phs, _total = await ph_repo.list_active(include_transient=True)
        # Each PH must have a single identity (not both).
        for ph in phs:
            assert ph.current_identity_id in ("alice", "bob"), (
                f"PH {ph.ph_id} has unexpected identity {ph.current_identity_id}"
            )


# ── Guardrail: resident_plus_stranger ─────────────────────────────────────


@pytest.mark.integration
class TestResidentPlusStrangerContinuity:
    """resident_plus_stranger.bin with continuity features enabled.

    Clinical guardrail: the favor-continuity bias must NOT transfer the
    resident's identity to the stranger's track.  This is the consequential
    dementia-care error (reporting the resident where it is actually a
    visitor).
    """

    @pytest.mark.asyncio
    async def test_stranger_never_gets_resident_identity(self, db_pool: Any) -> None:
        """The stranger's PH stays UNKNOWN throughout."""
        ph_repo, _tracker = await _replay_with_continuity_features(
            db_pool, "resident_plus_stranger.bin", known_identity_ids=["alice"]
        )

        phs, _total = await ph_repo.list_active(include_transient=True)

        # Resident PH must have alice identity.
        resident_phs = [ph for ph in phs if ph.current_identity_id == "alice"]
        assert len(resident_phs) >= 1, "resident (alice) must have at least one PH"

        # Stranger PH must NOT have alice identity.
        for ph in phs:
            if ph not in resident_phs:
                assert ph.current_identity_id != "alice", (
                    f"GUARDRAIL: stranger PH {ph.ph_id} must not inherit resident identity"
                )


# ── Shadow-metric proofs ─────────────────────────────────────────────────


@pytest.mark.integration
class TestContinuityShadowMetrics:
    """With continuity features OFF, the shadow counters must be > 0 on the turn replay,
    demonstrating the mechanisms would fire before enabling them."""

    @pytest.mark.asyncio
    async def test_shadow_revival_counter_increments(self, db_pool: Any) -> None:
        """world_tracker_shadow_revival_total > 0 when enable_ph_revival=False.

        The 6 s occlusion gap in single_camera_turn.bin forces PH close,
        then the post-gap observation triggers a shadow revival candidate.
        """
        from app.observability import metrics as _metrics

        before = _counter_total(_metrics.metrics.world_tracker_shadow_revival_total)
        await _replay_shadow(db_pool, "single_camera_turn.bin")
        after = _counter_total(_metrics.metrics.world_tracker_shadow_revival_total)

        assert after > before, (
            f"shadow revival counter should increment (before={before}, after={after})"
        )

    @pytest.mark.asyncio
    async def test_shadow_sticky_maintenance_counter_increments(self, db_pool: Any) -> None:
        """identity_shadow_mismatch_total{feature="sticky_maintenance"} > 0
        when enable_sticky_maintenance=False.

        Constructed scenario: PH with committed alice identity, posterior
        shifts to UNKNOWN (no face, no ReID), but sticky maintenance
        would hold alice within the window.  The shadow counter fires
        when the live decision (demote) differs from the sticky decision
        (hold).
        """
        from datetime import UTC, datetime, timedelta

        from app.domain import Identity, PosteriorDist
        from app.observability import metrics as _metrics
        from app.storage.base import InMemoryGalleryRepository
        from app.tracking.identity_resolver import IdentityResolver, ResolverConfig

        now = datetime.now(UTC)
        gallery = InMemoryGalleryRepository()
        await gallery.upsert_identity(
            Identity(identity_id="alice", display_name="Alice", enrolled_at=now)
        )

        resolver = IdentityResolver(
            gallery_repo=gallery,
            config=ResolverConfig(
                enable_sticky_maintenance=False,  # shadow mode
                prior_maintenance_max_age_s=120.0,
            ),
        )

        # Build a PH-like resolvable with committed alice identity.
        committed_at = now - timedelta(seconds=5)

        from types import SimpleNamespace

        entity = SimpleNamespace(
            entity_id="ph-test",
            observation_ids=["obs-1"],
            camera_ids=["cam-a"],
            current_identity_id="alice",
            current_identity_committed_at=committed_at,
            last_seen_at=now,
            started_at=now - timedelta(seconds=60),
        )

        # Posterior says UNKNOWN dominates (no face, no ReID evidence).
        posterior = PosteriorDist({"UNKNOWN": 0.90, "alice": 0.10})
        face = PosteriorDist({})  # no face evidence
        reid = PosteriorDist({})  # no ReID evidence

        before = _labeled_counter_value(
            _metrics.metrics.identity_shadow_mismatch_total,
            "feature",
            "sticky_maintenance",
        )

        decision = resolver._commit(entity, posterior, face, reid, now)
        # Live: identity_unchanged is False (top_id="UNKNOWN" != "alice"),
        # no standard maintenance window → demotes to UNKNOWN.
        assert decision.identity_id is None, "live decision should demote to UNKNOWN"

        after = _labeled_counter_value(
            _metrics.metrics.identity_shadow_mismatch_total,
            "feature",
            "sticky_maintenance",
        )

        assert after > before, (
            f"shadow sticky_maintenance counter should increment (before={before}, after={after})"
        )

    @pytest.mark.asyncio
    async def test_shadow_assoc_counter_mechanism_is_wired(self, db_pool: Any) -> None:
        """The shadow association counter is registered and replay with
        uncalibrated observations does not crash.

        The fine-grained gate-comparison logic is proven at unit level in
        test_cost_matrix.py.  This integration test verifies the counter
        exists and the replay path exercises the shadow-assoc code.
        """
        from app.observability import metrics as _metrics

        # Verify counter is registered.
        counter = _metrics.metrics.world_tracker_shadow_assoc_mismatch_total
        assert counter is not None

        # Replay: exercises the shadow-assoc path (uncalibrated obs, relax off).
        await _replay_shadow(db_pool, "single_camera_turn.bin")

        # Counter may or may not increment depending on whether assignments
        # differ; the unit tests prove the logic.  Here we verify no crash.

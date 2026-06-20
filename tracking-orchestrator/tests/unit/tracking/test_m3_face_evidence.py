"""Rich face evidence contract — CTS unit tests.

Tests for three-valued face evidence in the identity resolver:
candidate corroboration, unknown face no-penalty, frontality weighting,
and candidate-not-contradiction guardrail.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from app.domain import (
    FaceAnchor,
    GlobalTrack,
    Identity,
    PosteriorDist,
)
from app.storage.base import InMemoryGalleryRepository
from app.tracking.identity.commit_policy import compute_contradiction
from app.tracking.identity.policy import CommitPolicy
from app.tracking.identity_resolver import (
    IdentityResolver,
    ResolverConfig,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gt(
    ph_id: str = "ph-1",
    current_identity_id: str | None = None,
    observation_ids: list[str] | None = None,
    camera_ids: list[str] | None = None,
    current_identity_committed_at: datetime | None = None,
) -> GlobalTrack:
    now = datetime.now(UTC)
    return GlobalTrack(
        global_track_id=ph_id,
        camera_ids=camera_ids or ["cam_a"],
        tracklet_ids=observation_ids or ["obs-1"],
        started_at=now,
        last_seen_at=now,
        current_identity_id=current_identity_id,
        current_identity_committed_at=current_identity_committed_at,
        state="active",
    )


def _make_identity(identity_id: str, display_name: str = "") -> Identity:
    return Identity(
        identity_id=identity_id,
        display_name=display_name or identity_id,
        enrolled_at=datetime.now(UTC),
    )


def _make_resolver(
    identities: list[Identity] | None = None,
    config: ResolverConfig | None = None,
    gallery_repo: InMemoryGalleryRepository | None = None,
) -> IdentityResolver:
    repo = gallery_repo or InMemoryGalleryRepository()
    return IdentityResolver(
        gallery_repo=repo,
        identities=identities or [],
        config=config or ResolverConfig(),
    )


# ---------------------------------------------------------------------------
# Frontality factor
# ---------------------------------------------------------------------------


class TestFrontalityFactor:
    def test_full_frontal(self):
        """Yaw within frontality_full_yaw_deg → factor = 1.0."""
        config = ResolverConfig(frontality_full_yaw_deg=15.0)
        resolver = _make_resolver(config=config)
        assert resolver._frontality_factor(0.0) == 1.0
        assert resolver._frontality_factor(10.0) == 1.0
        assert resolver._frontality_factor(15.0) == 1.0

    def test_full_profile(self):
        """Yaw at or beyond frontality_zero_yaw_deg → factor = frontality_min_factor."""
        config = ResolverConfig(frontality_zero_yaw_deg=60.0, frontality_min_factor=0.3)
        resolver = _make_resolver(config=config)
        assert resolver._frontality_factor(60.0) == 0.3
        assert resolver._frontality_factor(90.0) == 0.3

    def test_mid_range_linear(self):
        """Yaw halfway between full and zero thresholds → factor halfway."""
        config = ResolverConfig(
            frontality_full_yaw_deg=15.0,
            frontality_zero_yaw_deg=60.0,
            frontality_min_factor=0.3,
        )
        resolver = _make_resolver(config=config)
        # At 37.5° (halfway): factor = 1.0 - 0.5*(1.0-0.3) = 0.65
        factor = resolver._frontality_factor(37.5)
        assert factor == pytest.approx(0.65)

    def test_negative_yaw_treated_as_absolute(self):
        """Negative yaw (looking left) is treated the same as positive yaw."""
        config = ResolverConfig(frontality_full_yaw_deg=15.0, frontality_min_factor=0.3)
        resolver = _make_resolver(config=config)
        f_pos = resolver._frontality_factor(40.0)
        f_neg = resolver._frontality_factor(-40.0)
        assert f_pos == pytest.approx(f_neg)


# ---------------------------------------------------------------------------
# Candidate corroboration (B4)
# ---------------------------------------------------------------------------


class TestCandidateCorroboration:
    """A candidate face for the held identity raises its posterior vs no candidate."""

    @pytest.mark.asyncio
    async def test_candidate_face_corroborates_with_sticky_maintenance(self):
        """PH committed to alice; candidate face for alice at similarity 0.33, yaw=35°.
        With sticky maintenance on, the candidate helps hold the identity through
        the grey-zone frames where no recognized face exists."""
        identities = [_make_identity("alice"), _make_identity("bob")]
        repo = InMemoryGalleryRepository()
        for ident in identities:
            await repo.upsert_identity(ident)
        config = ResolverConfig(
            commit_prob=0.65,
            commit_margin=0.15,
            enable_sticky_maintenance=True,
        )
        resolver = _make_resolver(identities=identities, config=config, gallery_repo=repo)

        gt = _make_gt(current_identity_id="alice")

        # With candidate: a weak face anchor for alice at similarity 0.33, yaw=35°.
        candidate_anchor = FaceAnchor(
            person_id="alice",
            confidence=0.33,  # raw similarity
            recognition_state="candidate",
            similarity=0.33,
            yaw_deg=35.0,
            tracklet_id="obs-1",
        )

        outcome = await resolver.resolve(
            hypotheses=[gt],
            new_face_anchors=[candidate_anchor],
            captured_at=datetime.now(UTC),
        )
        decision = outcome.decisions[0]

        # With sticky maintenance and a candidate face for the held identity,
        # alice should NOT be demoted (the candidate corroborates continuity).
        # The candidate face for alice prevents contradiction even when the
        # posterior argmax is weak.
        assert decision.identity_id == "alice", (
            f"Expected alice held by sticky maintenance + candidate, got {decision.identity_id}"
        )


# ---------------------------------------------------------------------------
# Unknown face no-penalty (B4)
# ---------------------------------------------------------------------------


class TestUnknownFaceNoPenalty:
    """A face_present_unknown marker adds mass to UNKNOWN but does not demote a held identity."""

    @pytest.mark.asyncio
    async def test_unknown_face_does_not_demote_held_identity(self):
        """PH committed to alice; unrecognized face marker present. Alice stays alice."""
        identities = [_make_identity("alice"), _make_identity("bob")]
        repo = InMemoryGalleryRepository()
        for ident in identities:
            await repo.upsert_identity(ident)
        config = ResolverConfig(
            commit_prob=0.65,
            commit_margin=0.15,
            face_present_unknown_unknown_mass=0.10,
            enable_sticky_maintenance=True,
        )
        resolver = _make_resolver(identities=identities, config=config, gallery_repo=repo)

        gt = _make_gt(current_identity_id="alice")

        unrecognized_anchor = FaceAnchor(
            person_id="unknown",
            confidence=0.70,  # det_score
            recognition_state="unrecognized",
            similarity=0.12,
            yaw_deg=10.0,
            tracklet_id="obs-1",
        )

        outcome = await resolver.resolve(
            hypotheses=[gt],
            new_face_anchors=[unrecognized_anchor],
            captured_at=datetime.now(UTC),
        )

        decision = outcome.decisions[0]
        # The held identity should still be alice.
        assert decision.identity_id == "alice", (
            f"Expected identity to stay alice, got {decision.identity_id}"
        )
        # UNKNOWN mass should be present in the posterior.
        unknown_prob = decision.posterior.distribution.get("UNKNOWN", 0.0)
        assert unknown_prob > 0.0, "Expected UNKNOWN mass > 0 in posterior"

    @pytest.mark.asyncio
    async def test_new_ph_with_only_unknown_face_resolves_to_unknown(self):
        """A brand-new PH with only face_present_unknown resolves to UNKNOWN."""
        ident = _make_identity("alice")
        repo = InMemoryGalleryRepository()
        await repo.upsert_identity(ident)
        resolver = _make_resolver(identities=[ident], gallery_repo=repo)

        gt = _make_gt(current_identity_id=None)

        unrecognized_anchor = FaceAnchor(
            person_id="unknown",
            confidence=0.70,
            recognition_state="unrecognized",
            similarity=0.10,
            yaw_deg=5.0,
            tracklet_id="obs-1",
        )

        outcome = await resolver.resolve(
            hypotheses=[gt],
            new_face_anchors=[unrecognized_anchor],
            captured_at=datetime.now(UTC),
        )

        decision = outcome.decisions[0]
        assert decision.identity_id is None, (
            f"Expected UNKNOWN (None) for new PH, got {decision.identity_id}"
        )


# ---------------------------------------------------------------------------
# Frontality weighting (B4)
# ---------------------------------------------------------------------------


class TestFrontalityWeighting:
    """Frontality factor down-weights off-axis faces."""

    @pytest.mark.asyncio
    async def test_frontal_face_produces_higher_posterior_than_profile(self):
        """Same face confidence at yaw=5° vs yaw=70°: frontal yields higher posterior."""
        identities = [_make_identity("alice"), _make_identity("bob")]
        repo = InMemoryGalleryRepository()
        for ident in identities:
            await repo.upsert_identity(ident)
        resolver = _make_resolver(identities=identities, gallery_repo=repo)

        entity_frontal = _make_gt(ph_id="ph-frontal")
        entity_profile = _make_gt(ph_id="ph-profile")

        frontal_anchor = FaceAnchor(
            person_id="alice",
            confidence=0.70,
            recognition_state="recognized",
            similarity=0.70,
            yaw_deg=5.0,
            tracklet_id="obs-1",
        )
        profile_anchor = FaceAnchor(
            person_id="alice",
            confidence=0.70,
            recognition_state="recognized",
            similarity=0.70,
            yaw_deg=70.0,
            tracklet_id="obs-1",
        )

        outcome_frontal = await resolver.resolve(
            hypotheses=[entity_frontal],
            new_face_anchors=[frontal_anchor],
            captured_at=datetime.now(UTC),
        )
        outcome_profile = await resolver.resolve(
            hypotheses=[entity_profile],
            new_face_anchors=[profile_anchor],
            captured_at=datetime.now(UTC),
        )

        prob_frontal = outcome_frontal.decisions[0].posterior.distribution.get("alice", 0.0)
        prob_profile = outcome_profile.decisions[0].posterior.distribution.get("alice", 0.0)

        assert prob_frontal > prob_profile, (
            f"Frontal ({prob_frontal:.4f}) should beat profile ({prob_profile:.4f})"
        )

    @pytest.mark.asyncio
    async def test_frontality_reduces_posterior(self):
        """The frontality factor reduces posterior for off-axis faces vs frontal faces."""
        identities = [_make_identity("alice")]
        repo = InMemoryGalleryRepository()
        for ident in identities:
            await repo.upsert_identity(ident)
        config = ResolverConfig(commit_prob=0.60, commit_margin=0.10)
        resolver = _make_resolver(identities=identities, config=config, gallery_repo=repo)

        # Frontal face with same confidence.
        frontal_anchor = FaceAnchor(
            person_id="alice",
            confidence=0.65,
            recognition_state="recognized",
            similarity=0.65,
            yaw_deg=5.0,
            tracklet_id="obs-1",
        )
        # Profile face at yaw=70° (frontality_factor should be min_factor=0.3).
        profile_anchor = FaceAnchor(
            person_id="alice",
            confidence=0.65,
            recognition_state="recognized",
            similarity=0.65,
            yaw_deg=70.0,
            tracklet_id="obs-2",
        )

        outcome_frontal = await resolver.resolve(
            hypotheses=[_make_gt(ph_id="ph-frontal")],
            new_face_anchors=[frontal_anchor],
            captured_at=datetime.now(UTC),
        )
        outcome_profile = await resolver.resolve(
            hypotheses=[_make_gt(ph_id="ph-profile")],
            new_face_anchors=[profile_anchor],
            captured_at=datetime.now(UTC),
        )

        prob_frontal = outcome_frontal.decisions[0].posterior.distribution.get("alice", 0.0)
        prob_profile = outcome_profile.decisions[0].posterior.distribution.get("alice", 0.0)

        # The frontal face should produce a strictly higher posterior for alice.
        assert prob_frontal > prob_profile, (
            f"Frontal posterior ({prob_frontal:.4f}) should beat profile ({prob_profile:.4f})"
        )


# ---------------------------------------------------------------------------
# Candidate not contradiction (B4 guardrail)
# ---------------------------------------------------------------------------


class TestCandidateNotContradiction:
    """Candidate and unrecognized faces must never trigger the contradiction path."""

    def test_candidate_face_not_contradiction(self):
        """A candidate face for bob does NOT contradict held identity alice."""
        face_likelihood = PosteriorDist({"bob": 0.33, "UNKNOWN": 0.67})
        contradicted = compute_contradiction(
            prev_id="alice",
            face_likelihood=face_likelihood,
            best_face_confidence=None,  # None when only candidate anchors exist
            top_id="UNKNOWN",
            top_prob=0.5,
            margin=0.1,
            config=CommitPolicy(),
        )
        assert contradicted is False

    def test_unrecognized_face_not_contradiction(self):
        """An unrecognized face does NOT contradict a held identity."""
        face_likelihood = PosteriorDist({"UNKNOWN": 1.0})
        contradicted = compute_contradiction(
            prev_id="alice",
            face_likelihood=face_likelihood,
            best_face_confidence=None,  # None for unrecognized
            top_id="UNKNOWN",
            top_prob=0.9,
            margin=0.5,
            config=CommitPolicy(),
        )
        assert contradicted is False

    def test_recognized_different_face_still_contradicts(self):
        """A recognized face for bob at high confidence STILL contradicts held alice."""
        face_likelihood = PosteriorDist({"bob": 0.85, "UNKNOWN": 0.15})
        contradicted = compute_contradiction(
            prev_id="alice",
            face_likelihood=face_likelihood,
            best_face_confidence=0.85,  # recognized anchor confidence
            top_id="bob",
            top_prob=0.85,
            margin=0.3,
            config=CommitPolicy(contradiction_face_confidence=0.70),
        )
        assert contradicted is True

    def test_recognized_same_identity_not_contradiction(self):
        """A recognized face for alice does NOT contradict held alice."""
        face_likelihood = PosteriorDist({"alice": 0.88, "UNKNOWN": 0.12})
        contradicted = compute_contradiction(
            prev_id="alice",
            face_likelihood=face_likelihood,
            best_face_confidence=0.88,
            top_id="alice",
            top_prob=0.88,
            margin=0.3,
            config=CommitPolicy(contradiction_face_confidence=0.70),
        )
        assert contradicted is False

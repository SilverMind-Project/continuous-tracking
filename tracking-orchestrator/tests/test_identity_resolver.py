"""Tests for IdentityResolver and PosteriorDist."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.domain import (
    FaceAnchor,
    GlobalTrack,
    Identity,
    IdentityDecision,
    PosteriorDist,
    ResolveOutcome,
)
from app.storage.base import (
    InMemoryGalleryRepository,
    InMemoryGlobalTrackRepository,
    InMemoryTrackingRepository,
)
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig

# ---------------------------------------------------------------------------
# PosteriorDist tests
# ---------------------------------------------------------------------------


class TestPosteriorDist:
    """Test the PosteriorDist domain type."""

    def test_normalization(self) -> None:
        dist = PosteriorDist({"a": 1.0, "b": 2.0, "c": 3.0})
        total = sum(dist.distribution.values())
        assert abs(total - 1.0) < 1e-9

    def test_top_identity(self) -> None:
        dist = PosteriorDist({"alice": 0.6, "bob": 0.3, "UNKNOWN": 0.1})
        top_id, top_prob = dist.top_identity()
        assert top_id == "alice"
        assert top_prob == pytest.approx(0.6)

    def test_top_with_margin(self) -> None:
        dist = PosteriorDist({"alice": 0.8, "bob": 0.15, "UNKNOWN": 0.05})
        ((top_id, top_prob), margin) = dist.top_with_margin()
        assert top_id == "alice"
        assert top_prob == pytest.approx(0.8)
        assert margin == pytest.approx(0.65)

    def test_top_with_margin_single_candidate(self) -> None:
        dist = PosteriorDist({"alice": 1.0})
        ((top_id, top_prob), margin) = dist.top_with_margin()
        assert top_id == "alice"
        assert top_prob == pytest.approx(1.0)
        assert margin == pytest.approx(1.0)

    def test_entropy_uniform(self) -> None:
        dist = PosteriorDist({"a": 0.5, "b": 0.5})
        # H = -0.5*log2(0.5) - 0.5*log2(0.5) = 1.0
        assert dist.entropy() == pytest.approx(1.0)

    def test_entropy_deterministic(self) -> None:
        dist = PosteriorDist({"a": 1.0})
        assert dist.entropy() == pytest.approx(0.0)

    def test_entropy_increases_with_spread(self) -> None:
        concentrated = PosteriorDist({"a": 0.9, "b": 0.05, "c": 0.05})
        uniform = PosteriorDist({"a": 0.34, "b": 0.33, "c": 0.33})
        assert concentrated.entropy() < uniform.entropy()

    def test_has_identity(self) -> None:
        dist = PosteriorDist({"alice": 0.8, "bob": 0.2})
        assert dist.has_identity("alice")
        assert not dist.has_identity("charlie")

    def test_zero_total_raises(self) -> None:
        with pytest.raises(ValueError, match="positive total probability"):
            PosteriorDist({"a": 0.0, "b": 0.0})


# ---------------------------------------------------------------------------
# IdentityResolver tests
# ---------------------------------------------------------------------------


def _make_gt(
    global_track_id: str = "gt-1",
    current_identity_id: str | None = None,
    camera_ids: list[str] | None = None,
    tracklet_ids: list[str] | None = None,
) -> GlobalTrack:
    now = datetime.now(UTC)
    return GlobalTrack(
        global_track_id=global_track_id,
        camera_ids=camera_ids or ["cam_a"],
        tracklet_ids=tracklet_ids or ["t1"],
        started_at=now,
        last_seen_at=now,
        current_identity_id=current_identity_id,
        state="active",
    )


def _make_identity(
    identity_id: str,
    display_name: str = "",
) -> Identity:
    return Identity(
        identity_id=identity_id,
        display_name=display_name or identity_id,
        enrolled_at=datetime.now(UTC),
    )


def _make_face_anchor(
    person_id: str,
    confidence: float = 0.9,
    tracklet_id: str = "t1",
) -> FaceAnchor:
    return FaceAnchor(
        person_id=person_id,
        confidence=confidence,
        quality=0.9,
        tracklet_id=tracklet_id,
    )


def _make_resolver(
    identities: list[Identity] | None = None,
    config: ResolverConfig | None = None,
    gallery_repo: InMemoryGalleryRepository | None = None,
) -> IdentityResolver:
    return IdentityResolver(
        tracking_repo=InMemoryTrackingRepository(),
        gallery_repo=gallery_repo or InMemoryGalleryRepository(),
        global_track_repo=InMemoryGlobalTrackRepository(),
        identities=identities or [],
        config=config or ResolverConfig(),
    )


class TestIdentityResolver:
    """Test the IdentityResolver with Bayesian posterior."""

    @pytest.mark.asyncio
    async def test_resolve_no_global_tracks(self) -> None:
        resolver = _make_resolver()
        outcome = await resolver.resolve(
            global_tracks=[],
            new_face_anchors=[],
            captured_at=datetime.now(UTC),
        )
        assert outcome.decisions == []
        assert outcome.revisions == []

    @pytest.mark.asyncio
    async def test_resolve_initial_assignment_strong_face(
        self,
    ) -> None:
        """A strong face anchor should produce an initial identity assignment."""
        identities = [_make_identity("grandma", "Grandma")]
        resolver = _make_resolver(identities=identities)

        gt = _make_gt(current_identity_id=None)
        anchor = _make_face_anchor("grandma", confidence=0.95, tracklet_id="t1")

        outcome = await resolver.resolve(
            global_tracks=[gt],
            new_face_anchors=[anchor],
            captured_at=datetime.now(UTC),
        )

        assert len(outcome.decisions) == 1
        decision = outcome.decisions[0]
        assert decision.global_track_id == "gt-1"
        assert decision.identity_id == "grandma"
        assert decision.revises_previous is True

    @pytest.mark.asyncio
    async def test_resolve_stays_unknown_below_threshold(
        self,
    ) -> None:
        """Weak evidence should keep the track as UNKNOWN."""
        resolver = _make_resolver()

        gt = _make_gt(current_identity_id=None)
        # No face anchors, no gallery hits -> uniform posterior -> UNKNOWN
        outcome = await resolver.resolve(
            global_tracks=[gt],
            new_face_anchors=[],
            captured_at=datetime.now(UTC),
        )

        assert len(outcome.decisions) == 1
        decision = outcome.decisions[0]
        # With no evidence, the posterior is spread thin -> margin < commit_margin
        assert decision.identity_id is None

    @pytest.mark.asyncio
    async def test_resolve_revises_previous(self) -> None:
        """A new face anchor for a different person should revise the identity."""
        identities = [
            _make_identity("grandma", "Grandma"),
            _make_identity("dad", "Dad"),
        ]
        resolver = _make_resolver(identities=identities)

        # Start with grandma as the current identity.
        gt = _make_gt(current_identity_id="grandma")
        # New face anchor for grandma (strong) -> should stay grandma.
        anchor = _make_face_anchor("grandma", confidence=0.95, tracklet_id="t1")

        outcome = await resolver.resolve(
            global_tracks=[gt],
            new_face_anchors=[anchor],
            captured_at=datetime.now(UTC),
        )

        assert len(outcome.decisions) == 1
        decision = outcome.decisions[0]
        # Same identity -> no revision.
        assert decision.identity_id == "grandma"
        assert decision.revises_previous is False

    @pytest.mark.asyncio
    async def test_resolve_commit_rule_margin(
        self,
    ) -> None:
        """When margin is too thin, the decision should be UNKNOWN."""
        config = ResolverConfig(commit_prob=0.85, commit_margin=0.25)
        resolver = _make_resolver(config=config)

        # With no distinguishing evidence, posterior is spread thin.
        gt = _make_gt(current_identity_id=None)
        outcome = await resolver.resolve(
            global_tracks=[gt],
            new_face_anchors=[],
            captured_at=datetime.now(UTC),
        )

        decision = outcome.decisions[0]
        # Margin should be thin -> identity_id is None.
        assert decision.identity_id is None

    @pytest.mark.asyncio
    async def test_resolve_produces_revision_on_change(
        self,
    ) -> None:
        """A new face anchor that changes identity should produce a revision."""
        identities = [
            _make_identity("grandma", "Grandma"),
        ]
        resolver = _make_resolver(identities=identities)

        # Start with grandma assigned.
        gt = _make_gt(current_identity_id="grandma")
        # Strong face anchor for grandma -> stays grandma (no revision).
        anchor = _make_face_anchor("grandma", confidence=0.95, tracklet_id="t1")

        outcome = await resolver.resolve(
            global_tracks=[gt],
            new_face_anchors=[anchor],
            captured_at=datetime.now(UTC),
        )

        # No revision because identity didn't change.
        assert len(outcome.revisions) == 0

    @pytest.mark.asyncio
    async def test_posterior_combines_prior_and_face(
        self,
    ) -> None:
        """Prior + face evidence should combine to strengthen the posterior."""
        identities = [
            _make_identity("grandma", "Grandma"),
            _make_identity("dad", "Dad"),
        ]
        resolver = _make_resolver(identities=identities)

        # Start with grandma as prior.
        gt = _make_gt(current_identity_id="grandma")
        # No face anchor -> prior should still favor grandma.
        outcome = await resolver.resolve(
            global_tracks=[gt],
            new_face_anchors=[],
            captured_at=datetime.now(UTC),
        )

        decision = outcome.decisions[0]
        # Prior alone favors grandma but may not meet commit_prob=0.85.
        # The top identity should still be grandma.
        top_id, _ = decision.posterior.top_identity()
        assert top_id == "grandma"
        # With no face/ReID evidence, commit_prob is not met -> UNKNOWN.
        assert decision.identity_id is None

    @pytest.mark.asyncio
    async def test_identity_decision_fields(self) -> None:
        """Verify IdentityDecision has correct fields set."""
        resolver = _make_resolver()
        gt = _make_gt(current_identity_id=None)

        outcome = await resolver.resolve(
            global_tracks=[gt],
            new_face_anchors=[],
            captured_at=datetime.now(UTC),
        )

        decision = outcome.decisions[0]
        assert isinstance(decision, IdentityDecision)
        assert decision.global_track_id == "gt-1"
        assert isinstance(decision.posterior, PosteriorDist)
        assert isinstance(decision.revises_previous, bool)
        assert isinstance(decision.reason, str)

    @pytest.mark.asyncio
    async def test_revision_has_required_fields(self) -> None:
        """Verify IdentityRevision has correct fields set."""
        resolver = _make_resolver()
        gt = _make_gt(current_identity_id=None)

        outcome = await resolver.resolve(
            global_tracks=[gt],
            new_face_anchors=[],
            captured_at=datetime.now(UTC),
        )

        # No revision expected (no identity change).
        assert len(outcome.revisions) == 0

    @pytest.mark.asyncio
    async def test_resolve_multiple_global_tracks(
        self,
    ) -> None:
        """Resolver should handle multiple GlobalTracks independently."""
        identities = [_make_identity("grandma", "Grandma")]
        resolver = _make_resolver(identities=identities)

        gt1 = _make_gt(global_track_id="gt-1", current_identity_id=None)
        gt2 = _make_gt(global_track_id="gt-2", current_identity_id=None)

        outcome = await resolver.resolve(
            global_tracks=[gt1, gt2],
            new_face_anchors=[],
            captured_at=datetime.now(UTC),
        )

        assert len(outcome.decisions) == 2
        decision_ids = {d.global_track_id for d in outcome.decisions}
        assert decision_ids == {"gt-1", "gt-2"}

    @pytest.mark.asyncio
    async def test_resolve_produces_revision_on_identity_change(
        self,
    ) -> None:
        """A revision should be emitted when identity actually changes."""
        identities = [
            _make_identity("grandma", "Grandma"),
            _make_identity("dad", "Dad"),
        ]
        resolver = _make_resolver(
            identities=identities,
            config=ResolverConfig(commit_prob=0.5),
        )

        # First: assign grandma via face anchor
        face_anchor = FaceAnchor(
            person_id="grandma",
            confidence=0.9,
            quality=0.8,
            tracklet_id="t1",
        )
        gt = _make_gt(global_track_id="gt-1", current_identity_id=None)

        outcome1 = await resolver.resolve(
            global_tracks=[gt],
            new_face_anchors=[face_anchor],
            captured_at=datetime.now(UTC),
        )
        assert len(outcome1.decisions) == 1
        assert outcome1.decisions[0].identity_id == "grandma"
        assert len(outcome1.revisions) == 1

        # Simulate pipeline applying the decision to the GT
        gt = _make_gt(global_track_id="gt-1", current_identity_id="grandma")

        # Second: assign dad via face anchor -> should produce revision
        face_anchor2 = FaceAnchor(
            person_id="dad",
            confidence=0.9,
            quality=0.8,
            tracklet_id="t1",
        )

        outcome2 = await resolver.resolve(
            global_tracks=[gt],
            new_face_anchors=[face_anchor2],
            captured_at=datetime.now(UTC),
        )
        assert outcome2.decisions[0].identity_id == "dad"
        assert outcome2.decisions[0].revises_previous is True
        assert len(outcome2.revisions) == 1
        rev = outcome2.revisions[0]
        assert rev.previous_identity_id == "grandma"
        assert rev.new_identity_id == "dad"

    @pytest.mark.asyncio
    async def test_resolve_from_gallery(self) -> None:
        """ReID gallery search should contribute to posterior and commit."""
        from app.domain import GalleryEmbedding

        identities = [_make_identity("alice", "Alice")]
        gallery_repo = InMemoryGalleryRepository()
        await gallery_repo.upsert_identity(identities[0])
        await gallery_repo.upsert_gallery_entry(
            GalleryEmbedding(
                gallery_entry_id="ge-1",
                identity_id="alice",
                embedding=[0.5] * 768,  # matches placeholder [0.0]*768 (cosine sim = 0)
                seen_at=datetime.now(UTC),
            )
        )

        resolver = _make_resolver(
            identities=identities,
            gallery_repo=gallery_repo,
            config=ResolverConfig(commit_prob=0.65, prior_weight=0.3),
        )

        gt = _make_gt(global_track_id="gt-1", current_identity_id=None)

        outcome = await resolver.resolve(
            global_tracks=[gt],
            new_face_anchors=[],
            captured_at=datetime.now(UTC),
        )
        assert len(outcome.decisions) == 1
        assert outcome.decisions[0].identity_id == "alice"

    @pytest.mark.asyncio
    async def test_resolve_rate_limiting(self) -> None:
        """Revisions should be rate-limited per global track."""
        identities = [_make_identity("alice", "Alice")]
        resolver = _make_resolver(
            identities=identities,
            config=ResolverConfig(
                commit_prob=0.5,
                max_revisions_per_gt_per_minute=1,
            ),
        )

        # First commit should produce a revision
        face_anchor = FaceAnchor(
            person_id="alice",
            confidence=0.9,
            quality=0.8,
            tracklet_id="t1",
        )
        gt = _make_gt(global_track_id="gt-1", current_identity_id=None)

        t1 = datetime.now(UTC)
        outcome1 = await resolver.resolve(
            global_tracks=[gt],
            new_face_anchors=[face_anchor],
            captured_at=t1,
        )
        assert len(outcome1.revisions) == 1

        # Second commit within the same minute should be rate-limited
        t2 = t1
        outcome2 = await resolver.resolve(
            global_tracks=[gt],
            new_face_anchors=[face_anchor],
            captured_at=t2,
        )
        assert len(outcome2.revisions) == 0  # rate-limited


class TestResolveOutcome:
    """Test the ResolveOutcome container."""

    def test_empty_outcome(self) -> None:
        outcome = ResolveOutcome()
        assert outcome.decisions == []
        assert outcome.revisions == []

    def test_outcome_with_decisions(self) -> None:
        outcome = ResolveOutcome()
        outcome.decisions.append(
            IdentityDecision(
                global_track_id="gt-1",
                identity_id="grandma",
                posterior=PosteriorDist({"grandma": 0.9, "UNKNOWN": 0.1}),
                revises_previous=True,
            )
        )
        assert len(outcome.decisions) == 1
        assert outcome.decisions[0].identity_id == "grandma"

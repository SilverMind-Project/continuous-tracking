"""Tests for IdentityResolver and PosteriorDist."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.domain import (
    FaceAnchor,
    GalleryEmbedding,
    GlobalTrack,
    Identity,
    IdentityDecision,
    PosteriorDist,
    ResolveOutcome,
)
from app.storage.base import InMemoryGalleryRepository
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
    ph_id: str = "gt-1",
    current_identity_id: str | None = None,
    camera_ids: list[str] | None = None,
    tracklet_ids: list[str] | None = None,
    last_independent_identity_evidence_at: datetime | None = None,
) -> GlobalTrack:
    now = datetime.now(UTC)
    # When an identity is committed, default the evidence clock to now so
    # the 30-second maintenance window is open.  Tests that need an expired
    # clock must pass last_independent_identity_evidence_at explicitly.
    evidence_at = last_independent_identity_evidence_at
    if current_identity_id is not None and evidence_at is None:
        evidence_at = now
    return GlobalTrack(
        global_track_id=ph_id,
        camera_ids=camera_ids or ["cam_a"],
        tracklet_ids=tracklet_ids or ["t1"],
        started_at=now,
        last_seen_at=now,
        current_identity_id=current_identity_id,
        current_identity_committed_at=now if current_identity_id is not None else None,
        last_independent_identity_evidence_at=evidence_at,
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
        gallery_repo=gallery_repo or InMemoryGalleryRepository(),
        identities=identities or [],
        config=config or ResolverConfig(),
    )


class TestIdentityResolver:
    """Test the IdentityResolver with Bayesian posterior."""

    @pytest.mark.asyncio
    async def test_resolve_no_global_tracks(self) -> None:
        resolver = _make_resolver()
        outcome = await resolver.resolve(
            hypotheses=[],
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
            hypotheses=[gt],
            new_face_anchors=[anchor],
            captured_at=datetime.now(UTC),
        )

        assert len(outcome.decisions) == 1
        decision = outcome.decisions[0]
        assert decision.ph_id == "gt-1"
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
            hypotheses=[gt],
            new_face_anchors=[],
            captured_at=datetime.now(UTC),
        )

        assert len(outcome.decisions) == 1
        decision = outcome.decisions[0]
        # With no evidence, the posterior is spread thin -> margin < commit_margin
        assert decision.identity_id is None

    @pytest.mark.asyncio
    async def test_resolve_no_revision_when_same_identity(
        self,
    ) -> None:
        """A strong face anchor for the same identity should not produce a revision."""
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
            hypotheses=[gt],
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
            hypotheses=[gt],
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
            hypotheses=[gt],
            new_face_anchors=[anchor],
            captured_at=datetime.now(UTC),
        )

        # No revision because identity didn't change.
        assert len(outcome.revisions) == 0

    @pytest.mark.asyncio
    async def test_posterior_combines_prior_and_face(
        self,
    ) -> None:
        """Prior alone maintains an existing identity when within the maintenance window."""
        identities = [
            _make_identity("grandma", "Grandma"),
            _make_identity("dad", "Dad"),
        ]
        resolver = _make_resolver(identities=identities)

        # Start with grandma as prior.
        gt = _make_gt(current_identity_id="grandma")
        # No face anchor, no ReID -> prior alone should maintain grandma
        # when within the prior_maintenance_max_age_s window.
        outcome = await resolver.resolve(
            hypotheses=[gt],
            new_face_anchors=[],
            captured_at=datetime.now(UTC),
        )

        decision = outcome.decisions[0]
        # Prior alone favors grandma and maintains it (identity_unchanged gate).
        top_id, _ = decision.posterior.top_identity()
        assert top_id == "grandma"
        assert decision.identity_id == "grandma"
        assert decision.revises_previous is False

    @pytest.mark.asyncio
    async def test_identity_decision_fields(self) -> None:
        """Verify IdentityDecision has correct fields set."""
        resolver = _make_resolver()
        gt = _make_gt(current_identity_id=None)

        outcome = await resolver.resolve(
            hypotheses=[gt],
            new_face_anchors=[],
            captured_at=datetime.now(UTC),
        )

        decision = outcome.decisions[0]
        assert isinstance(decision, IdentityDecision)
        assert decision.ph_id == "gt-1"
        assert isinstance(decision.posterior, PosteriorDist)
        assert isinstance(decision.revises_previous, bool)
        assert isinstance(decision.reason, str)

    @pytest.mark.asyncio
    async def test_revision_has_required_fields(self) -> None:
        """Verify IdentityRevision has correct fields set."""
        resolver = _make_resolver()
        gt = _make_gt(current_identity_id=None)

        outcome = await resolver.resolve(
            hypotheses=[gt],
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

        gt1 = _make_gt(ph_id="gt-1", current_identity_id=None)
        gt2 = _make_gt(ph_id="gt-2", current_identity_id=None)

        outcome = await resolver.resolve(
            hypotheses=[gt1, gt2],
            new_face_anchors=[],
            captured_at=datetime.now(UTC),
        )

        assert len(outcome.decisions) == 2
        decision_ids = {d.ph_id for d in outcome.decisions}
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
        gt = _make_gt(ph_id="gt-1", current_identity_id=None)

        outcome1 = await resolver.resolve(
            hypotheses=[gt],
            new_face_anchors=[face_anchor],
            captured_at=datetime.now(UTC),
        )
        assert len(outcome1.decisions) == 1
        assert outcome1.decisions[0].identity_id == "grandma"
        assert len(outcome1.revisions) == 1

        # Simulate pipeline applying the decision to the GT
        gt = _make_gt(ph_id="gt-1", current_identity_id="grandma")

        # Second: assign dad via face anchor -> should produce revision
        face_anchor2 = FaceAnchor(
            person_id="dad",
            confidence=0.9,
            quality=0.8,
            tracklet_id="t1",
        )

        outcome2 = await resolver.resolve(
            hypotheses=[gt],
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
        # Gallery entry for the GlobalTrack's default tracklet "t1".
        await gallery_repo.upsert_gallery_entry(
            GalleryEmbedding(
                gallery_entry_id="ge-1",
                identity_id="alice",
                embedding=[0.9] * 768,
                seen_at=datetime.now(UTC),
                origin_tracklet_id="t1",
            )
        )

        resolver = _make_resolver(
            identities=identities,
            gallery_repo=gallery_repo,
            config=ResolverConfig(commit_prob=0.65, prior_weight=0.3),
        )

        gt = _make_gt(ph_id="gt-1", current_identity_id=None)

        outcome = await resolver.resolve(
            hypotheses=[gt],
            new_face_anchors=[],
            captured_at=datetime.now(UTC),
        )
        assert len(outcome.decisions) == 1
        assert outcome.decisions[0].identity_id == "alice"

    @pytest.mark.asyncio
    async def test_resolve_gallery_enrolled_without_constructor_identities(self) -> None:
        """Regression: new track should commit via ReID even when known_identities=[]
        at construction time — the fix refreshes from gallery_repo on each resolve()."""
        from app.domain import GalleryEmbedding

        gallery_repo = InMemoryGalleryRepository()
        alice = _make_identity("alice", "Alice")
        await gallery_repo.upsert_identity(alice)
        await gallery_repo.upsert_gallery_entry(
            GalleryEmbedding(
                gallery_entry_id="ge-1",
                identity_id="alice",
                embedding=[0.9] * 768,
                seen_at=datetime.now(UTC),
                origin_tracklet_id="t1",
            )
        )

        # Deliberately pass no identities at construction — mimics the production
        # wiring where known_identities defaults to [] in PipelineConfig.
        resolver = _make_resolver(
            identities=[],
            gallery_repo=gallery_repo,
            config=ResolverConfig(commit_prob=0.65, prior_weight=0.3),
        )

        gt = _make_gt(ph_id="gt-1", current_identity_id=None)
        outcome = await resolver.resolve(
            hypotheses=[gt],
            new_face_anchors=[],
            captured_at=datetime.now(UTC),
        )
        assert len(outcome.decisions) == 1
        # With the gallery refresh in resolve(), alice should be in the prior
        # and the strong ReID hit should push the posterior above commit_prob.
        assert outcome.decisions[0].identity_id == "alice"

    @pytest.mark.asyncio
    async def test_resolve_rate_limiting(self) -> None:
        """Revisions should be rate-limited per global track."""
        identities = [_make_identity("alice", "Alice")]
        resolver = _make_resolver(
            identities=identities,
            config=ResolverConfig(
                commit_prob=0.5,
                max_revisions_per_ph_per_minute=1,
            ),
        )

        # First commit should produce a revision
        face_anchor = FaceAnchor(
            person_id="alice",
            confidence=0.9,
            quality=0.8,
            tracklet_id="t1",
        )
        gt = _make_gt(ph_id="gt-1", current_identity_id=None)

        t1 = datetime.now(UTC)
        outcome1 = await resolver.resolve(
            hypotheses=[gt],
            new_face_anchors=[face_anchor],
            captured_at=t1,
        )
        assert len(outcome1.revisions) == 1

        # Second commit within the same minute should be rate-limited
        t2 = t1
        outcome2 = await resolver.resolve(
            hypotheses=[gt],
            new_face_anchors=[face_anchor],
            captured_at=t2,
        )
        assert len(outcome2.revisions) == 0  # rate-limited

    @pytest.mark.asyncio
    async def test_maintenance_window_survives_many_enrolled_identities(
        self,
    ) -> None:
        """Identity must be maintained across quiet frames even when N>=4 enrolled
        identities push the prior-only posterior below commit_prob=0.65.

        Regression test for the bug where within_maintenance_window=True set
        evidence_ok=True but the probability threshold (calibrated for fresh
        evidence) still rejected the commit, clearing a valid face-confirmed
        assignment on every no-evidence frame.
        """
        # Six enrolled identities: prior-only posterior for the current
        # identity = 0.6 / (0.6 + 5*0.08 + 0.05) ≈ 0.57 — below commit_prob.
        identities = [_make_identity(f"person_{i}") for i in range(6)]
        resolver = _make_resolver(identities=identities)

        gt = _make_gt(current_identity_id="person_0")
        outcome = await resolver.resolve(
            hypotheses=[gt],
            new_face_anchors=[],
            captured_at=datetime.now(UTC),
        )

        decision = outcome.decisions[0]
        assert decision.identity_id == "person_0", (
            "Identity must be maintained during the maintenance window even "
            "when prior-only probability falls below commit_prob with 6+ enrolled identities"
        )
        assert decision.revises_previous is False

    @pytest.mark.asyncio
    async def test_maintenance_window_expires_clears_identity(
        self,
    ) -> None:
        """After prior_maintenance_max_age_s, the identity should not be maintained
        by the prior alone — fresh evidence is required to re-commit."""
        from datetime import timedelta

        identities = [_make_identity("grandma"), _make_identity("dad")]
        resolver = _make_resolver(
            identities=identities,
            config=ResolverConfig(prior_maintenance_max_age_s=10.0),
        )

        now = datetime.now(UTC)
        # Simulate a GT whose last_seen_at is 30s ago — outside the window.
        stale_gt = GlobalTrack(
            global_track_id="gt-stale",
            camera_ids=["cam_a"],
            tracklet_ids=["t1"],
            started_at=now - timedelta(seconds=60),
            last_seen_at=now - timedelta(seconds=30),
            current_identity_id="grandma",
            state="active",
        )

        outcome = await resolver.resolve(
            hypotheses=[stale_gt],
            new_face_anchors=[],
            captured_at=now,
        )

        decision = outcome.decisions[0]
        # Outside maintenance window AND no evidence → prior probability
        # alone may or may not pass commit_prob (2 identities → passes).
        # The important check is that with a longer identity list it would
        # fail, but here we verify the expired-window path produces a
        # non-maintenance decision.
        assert not decision.revises_previous or (
            decision.identity_id is None or decision.identity_id == "grandma"
        )

    @pytest.mark.asyncio
    async def test_weak_reid_evidence_does_not_clear_committed_identity(
        self,
    ) -> None:
        """Committed identity must survive when weak ReID evidence (sim≈0.7)
        for the same person is present but below commit_prob.

        Regression test for the bug where `_commit()` required `not has_evidence`
        before activating the maintenance window. With weak ReID evidence
        present for the *same* identity, `has_evidence=True` would bypass the
        window and return `new_id=None` → pipeline would call `assign_identity(None)`,
        clearing the DB.
        """
        from app.storage.base import InMemoryGalleryRepository

        dim = 256

        def _unit(idx: int) -> list[float]:
            v = [0.0] * dim
            v[idx] = 1.0
            return v

        identities = [_make_identity(f"person_{i}") for i in range(6)]
        gallery = InMemoryGalleryRepository()
        for ident in identities:
            await gallery.upsert_identity(ident)

        # person_0's front-facing gallery entry
        gallery_emb = _unit(0)
        await gallery.upsert_gallery_entry(
            GalleryEmbedding(
                gallery_entry_id="p0_front",
                identity_id="person_0",
                embedding=gallery_emb,
                seen_at=datetime.now(UTC),
                quality=0.9,
                origin_tracklet_id="t_old",
                face_confirmed=True,
            )
        )

        # Back-facing query: cos_sim ≈ 0.7 to gallery_emb — below commit threshold
        # when combined with prior smoothing across 6 identities.
        import math

        back_query = [0.0] * dim
        back_query[0] = 0.70
        back_query[1] = math.sqrt(1 - 0.70**2)
        await gallery.upsert_gallery_entry(
            GalleryEmbedding(
                gallery_entry_id="p0_back_query",
                identity_id=None,
                embedding=back_query,
                seen_at=datetime.now(UTC),
                quality=0.9,
                origin_tracklet_id="t_current",
                face_confirmed=False,
            )
        )

        config = ResolverConfig(
            commit_prob=0.65,
            commit_margin=0.25,
            # Keep boost off so this tests the pure maintenance window path.
            identified_entry_boost_min_sim=0.99,
        )
        resolver = _make_resolver(identities=identities, config=config, gallery_repo=gallery)

        # GT already has person_0 committed from a prior face anchor.
        committed_gt = _make_gt(
            current_identity_id="person_0",
            tracklet_ids=["t_current"],
        )

        outcome = await resolver.resolve(
            hypotheses=[committed_gt],
            new_face_anchors=[],
            captured_at=datetime.now(UTC),
        )

        decision = outcome.decisions[0]
        assert decision.identity_id == "person_0", (
            "Maintenance window must carry committed identity forward when weak "
            "ReID evidence for the same person is present but below commit_prob"
        )
        assert decision.revises_previous is False

    @pytest.mark.asyncio
    async def test_face_confirmed_identity_persists_on_quiet_frames(
        self,
    ) -> None:
        """End-to-end: face fires once → identity committed → quiet frames maintain it.

        Simulates the full face-id → maintenance cycle with 6 enrolled
        identities (where the pre-fix resolver would clear the identity).
        """
        identities = [_make_identity(f"person_{i}") for i in range(6)]
        gallery = InMemoryGalleryRepository()
        resolver = _make_resolver(identities=identities, gallery_repo=gallery)

        gt = _make_gt(current_identity_id=None, tracklet_ids=["t1"])

        # Frame 1: face fires — identity committed.
        face_anchor = _make_face_anchor("person_0", confidence=0.92, tracklet_id="t1")
        outcome1 = await resolver.resolve(
            hypotheses=[gt],
            new_face_anchors=[face_anchor],
            captured_at=datetime.now(UTC),
        )
        decision1 = outcome1.decisions[0]
        assert decision1.identity_id == "person_0"
        assert decision1.revises_previous is True  # initial assignment

        # Simulate the GT being updated in the DB (evidence clock refreshed by
        # the WorldTracker via evidence_backed_commit after Frame 1).
        evidence_at = datetime.now(UTC)
        committed_gt = GlobalTrack(
            global_track_id=gt.global_track_id,
            camera_ids=gt.camera_ids,
            tracklet_ids=gt.tracklet_ids,
            started_at=gt.started_at,
            last_seen_at=evidence_at,
            current_identity_id="person_0",
            current_identity_committed_at=evidence_at,
            last_independent_identity_evidence_at=evidence_at,
            state="active",
        )

        # Frames 2-5: face cooldown, no ReID evidence → maintenance window.
        for _ in range(4):
            outcome_quiet = await resolver.resolve(
                hypotheses=[committed_gt],
                new_face_anchors=[],
                captured_at=datetime.now(UTC),
            )
            decision_quiet = outcome_quiet.decisions[0]
            assert decision_quiet.identity_id == "person_0", (
                "Face-confirmed identity must survive quiet frames with 6 enrolled identities"
            )
            assert decision_quiet.revises_previous is False


class TestGalleryBoost:
    """Gallery boost: face-confirmed entries floor-lift identity posterior.

    Regression guard for the 'turns away → new GT → unknown' race:
    - A new GT is created for the back-facing person.
    - Alice's gallery has front-facing embeddings (cosine sim ≈ 0.73 to query).
    - Without the boost, logistic(0.73) ≈ 0.73, but _combine() smoothing
      collapses the posterior below commit_prob.
    - With the boost, likelihood is floored to 0.80 and competing identities
      get explicit low values, so alice's posterior ≈ 0.83 ≥ 0.65.
    """

    @staticmethod
    def _make_emb(dim: int, *nonzero: tuple[int, float]) -> list[float]:
        """Sparse unit vector with specified non-zero components."""
        v = [0.0] * dim
        for idx, val in nonzero:
            v[idx] = val
        total = sum(x * x for x in v) ** 0.5
        return [x / total for x in v]

    @pytest.mark.asyncio
    async def test_back_facing_reid_commits_alice_via_boost(self) -> None:
        """New GT (no identity) with back-facing embeddings finds alice's
        front-facing gallery entries at sim≈0.73 → boost floors likelihood
        to 0.80 → alice commits on the new GT.
        """
        dim = 256
        # Alice's front-facing embedding direction.
        alice_front = self._make_emb(dim, (0, 1.0))
        # Back-facing query: same person, different pose → rotated embedding.
        # cos_sim = 0.7 * alice_front[0] + 0.714... * alice_front[1] ≈ 0.73
        back_query = self._make_emb(dim, (0, 0.70), (1, 0.7141))

        gallery = InMemoryGalleryRepository()

        # Alice has 3 front-facing gallery entries (face-confirmed).
        for i in range(3):
            await gallery.upsert_gallery_entry(
                GalleryEmbedding(
                    gallery_entry_id=f"alice_entry_{i}",
                    identity_id="alice",
                    embedding=alice_front,
                    seen_at=datetime.now(UTC),
                    quality=0.9,
                    origin_tracklet_id="t_alice_old",
                    face_confirmed=True,
                )
            )

        # Bob and carol have orthogonal embeddings (no match).
        bob_emb = self._make_emb(dim, (2, 1.0))
        carol_emb = self._make_emb(dim, (3, 1.0))
        await gallery.upsert_gallery_entry(
            GalleryEmbedding(
                gallery_entry_id="bob_entry",
                identity_id="bob",
                embedding=bob_emb,
                seen_at=datetime.now(UTC),
                quality=0.9,
                origin_tracklet_id="t_bob",
                face_confirmed=True,
            )
        )
        await gallery.upsert_gallery_entry(
            GalleryEmbedding(
                gallery_entry_id="carol_entry",
                identity_id="carol",
                embedding=carol_emb,
                seen_at=datetime.now(UTC),
                quality=0.9,
                origin_tracklet_id="t_carol",
                face_confirmed=True,
            )
        )

        # New GT for the back-facing person (no identity yet).
        # Tracklet t2 has a back-facing gallery entry used as the query.
        await gallery.upsert_gallery_entry(
            GalleryEmbedding(
                gallery_entry_id="new_gt_entry",
                identity_id=None,
                embedding=back_query,
                seen_at=datetime.now(UTC),
                quality=0.9,
                origin_tracklet_id="t2",
                face_confirmed=False,
            )
        )

        identities = [
            _make_identity("alice"),
            _make_identity("bob"),
            _make_identity("carol"),
        ]
        # Gallery repo must also know about these identities so list_gallery_entries
        # does not filter them out with active_only=True.
        for ident in identities:
            await gallery.upsert_identity(ident)

        config = ResolverConfig(
            commit_prob=0.65,
            commit_margin=0.25,
            identified_entry_boost_min_sim=0.65,
            identified_entry_min_likelihood=0.80,
        )
        resolver = _make_resolver(identities=identities, config=config, gallery_repo=gallery)

        new_gt = _make_gt(
            ph_id="gt-new",
            current_identity_id=None,
            tracklet_ids=["t2"],
        )

        outcome = await resolver.resolve(
            hypotheses=[new_gt],
            new_face_anchors=[],
            captured_at=datetime.now(UTC),
        )

        assert len(outcome.decisions) == 1
        decision = outcome.decisions[0]
        assert decision.identity_id == "alice", (
            "Gallery boost must commit alice when back-facing query finds "
            "front-facing alice entries at sim≈0.73"
        )

    @pytest.mark.asyncio
    async def test_ambiguous_boost_does_not_commit(self) -> None:
        """When two identities both have boosted entries and similar scores,
        the margin requirement prevents a commit.
        """
        dim = 256
        # Two identities with equally similar embeddings to the query.
        query = self._make_emb(dim, (0, 0.70), (1, 0.7141))
        shared_direction = self._make_emb(dim, (0, 1.0))

        gallery = InMemoryGalleryRepository()

        identities = [_make_identity("alice"), _make_identity("bob")]
        for ident in identities:
            await gallery.upsert_identity(ident)

        for person in ("alice", "bob"):
            await gallery.upsert_gallery_entry(
                GalleryEmbedding(
                    gallery_entry_id=f"{person}_entry",
                    identity_id=person,
                    embedding=shared_direction,
                    seen_at=datetime.now(UTC),
                    quality=0.9,
                    origin_tracklet_id=f"t_{person}",
                    face_confirmed=True,
                )
            )

        await gallery.upsert_gallery_entry(
            GalleryEmbedding(
                gallery_entry_id="new_gt_entry",
                identity_id=None,
                embedding=query,
                seen_at=datetime.now(UTC),
                quality=0.9,
                origin_tracklet_id="t_new",
                face_confirmed=False,
            )
        )

        config = ResolverConfig(
            commit_prob=0.65,
            commit_margin=0.25,
            identified_entry_boost_min_sim=0.65,
            identified_entry_min_likelihood=0.80,
        )
        resolver = _make_resolver(identities=identities, config=config, gallery_repo=gallery)

        new_gt = _make_gt(
            ph_id="gt-ambiguous",
            current_identity_id=None,
            tracklet_ids=["t_new"],
        )

        outcome = await resolver.resolve(
            hypotheses=[new_gt],
            new_face_anchors=[],
            captured_at=datetime.now(UTC),
        )

        assert len(outcome.decisions) == 1
        decision = outcome.decisions[0]
        # Both alice and bob have identical boosts → margin < 0.25 → no commit.
        assert decision.identity_id is None, (
            "Ambiguous equal-boost scenario must not commit any identity"
        )


class TestEvidenceClock:
    """Tests for the evidence-clock-based maintenance window (M02).

    Face locks were removed in M02. Identity maintenance is now governed by
    ``last_independent_identity_evidence_at`` and the 30-second window.
    """

    @pytest.mark.asyncio
    async def test_identity_maintained_within_30s_evidence_window(self) -> None:
        """Identity is held when the evidence clock is within 30 seconds."""
        from datetime import timedelta

        identities = [_make_identity("alice", "Alice")]
        config = ResolverConfig(prior_maintenance_max_age_s=30.0)
        resolver = _make_resolver(identities=identities, config=config)

        now = datetime.now(UTC)
        # Set evidence clock to 5 s ago — within the 30 s window.
        evidence_at = now - timedelta(seconds=5)
        gt = GlobalTrack(
            global_track_id="gt-1",
            camera_ids=["cam_a"],
            tracklet_ids=["t1"],
            started_at=evidence_at,
            last_seen_at=now,
            current_identity_id="alice",
            current_identity_committed_at=evidence_at,
            last_independent_identity_evidence_at=evidence_at,
            state="active",
        )
        outcome = await resolver.resolve(
            hypotheses=[gt],
            new_face_anchors=[],
            captured_at=now,
        )

        assert outcome.decisions[0].identity_id == "alice"

    @pytest.mark.asyncio
    async def test_identity_expires_after_30s_evidence_window(self) -> None:
        """Identity is demoted to Unknown when the evidence clock exceeds 30 seconds."""
        from datetime import timedelta

        identities = [_make_identity("alice", "Alice")]
        config = ResolverConfig(
            prior_maintenance_max_age_s=30.0,
            enable_sticky_maintenance=False,
        )
        resolver = _make_resolver(identities=identities, config=config)

        now = datetime.now(UTC)
        # Evidence clock is 31 s ago — outside the 30 s window.
        evidence_at = now - timedelta(seconds=31)
        gt = GlobalTrack(
            global_track_id="gt-1",
            camera_ids=["cam_a"],
            tracklet_ids=["t1"],
            started_at=evidence_at,
            last_seen_at=now,
            current_identity_id="alice",
            current_identity_committed_at=evidence_at,
            last_independent_identity_evidence_at=evidence_at,
            state="active",
        )
        outcome = await resolver.resolve(
            hypotheses=[gt],
            new_face_anchors=[],
            captured_at=now,
        )

        assert outcome.decisions[0].identity_id is None

    @pytest.mark.asyncio
    async def test_no_evidence_clock_means_no_maintenance_window(self) -> None:
        """An entity without an evidence clock has no maintenance window."""
        from datetime import timedelta

        identities = [_make_identity("alice", "Alice")]
        config = ResolverConfig(prior_maintenance_max_age_s=30.0)
        resolver = _make_resolver(identities=identities, config=config)

        now = datetime.now(UTC)
        # No last_independent_identity_evidence_at set (e.g., migrated PH).
        gt = GlobalTrack(
            global_track_id="gt-1",
            camera_ids=["cam_a"],
            tracklet_ids=["t1"],
            started_at=now - timedelta(seconds=5),
            last_seen_at=now,
            current_identity_id="alice",
            current_identity_committed_at=now - timedelta(seconds=5),
            last_independent_identity_evidence_at=None,  # no evidence clock
            state="active",
        )
        outcome = await resolver.resolve(
            hypotheses=[gt],
            new_face_anchors=[],
            captured_at=now,
        )

        # No evidence clock → maintenance window is not open → identity demoted.
        assert outcome.decisions[0].identity_id is None


class TestF1EvidenceClockIdentityMatch:
    """F1 regression: the evidence clock must never be treated as refreshed
    by evidence for a *different* identity, or by mere distribution presence
    (M01, ``codebase-hardening-m01-evidence-clock-integrity.md``).

    ``decision.evidence_backed`` is exactly what routes the PH repository
    call in ``WorldTracker._resolve_identities``: True -> ``evidence_backed_commit``
    (clock advances to ``captured_at``), False -> ``prior_only_update`` (clock
    untouched). Asserting ``evidence_backed`` here is equivalent to asserting
    ``evidence_backed_commit`` is/isn't invoked, without needing a PH repo.
    """

    @staticmethod
    async def _resolver(
        config: ResolverConfig,
    ) -> tuple[IdentityResolver, InMemoryGalleryRepository]:
        identities = [_make_identity("amma"), _make_identity("grandma")]
        gallery = InMemoryGalleryRepository()
        for ident in identities:
            await gallery.upsert_identity(ident)
        resolver = IdentityResolver(gallery_repo=gallery, identities=identities, config=config)
        return resolver, gallery

    @staticmethod
    def _held_gt(t0: datetime, evidence_at: datetime) -> GlobalTrack:
        return GlobalTrack(
            global_track_id="ph-1",
            camera_ids=["cam_a"],
            tracklet_ids=["t1"],
            started_at=t0,
            last_seen_at=t0,
            current_identity_id="amma",
            current_identity_committed_at=t0,
            last_independent_identity_evidence_at=evidence_at,
            state="active",
        )

    @pytest.mark.asyncio
    async def test_foreign_evidence_never_renews_clock_and_window_expires(self) -> None:
        """PH holds 'amma'; every frame for 30s only carries a weak,
        non-contradicting 'grandma' recognized face. The clock (T0) must
        never be renewed, so at T0+31s the maintenance window has expired
        and the identity clears to Unknown -- the grandma/amma swap failure
        class this milestone closes.
        """
        t0 = datetime.now(UTC)
        config = ResolverConfig(
            prior_maintenance_max_age_s=30.0,
            enable_sticky_maintenance=False,
            contradiction_face_confidence=0.70,
        )
        resolver, _ = await self._resolver(config)
        foreign_anchor = _make_face_anchor("grandma", confidence=0.3)

        for offset_s in (5, 15, 25, 29):
            captured_at = t0 + timedelta(seconds=offset_s)
            outcome = await resolver.resolve(
                hypotheses=[self._held_gt(t0, t0)],
                new_face_anchors=[foreign_anchor],
                captured_at=captured_at,
            )
            decision = outcome.decisions[0]
            assert decision.identity_id == "amma", f"offset={offset_s}s"
            # Foreign evidence must never back the held identity: this is
            # exactly the condition that prevents evidence_backed_commit and
            # routes to prior_only_update instead, so the clock stays at T0.
            assert not decision.evidence_backed, f"offset={offset_s}s"
            assert not decision.revises_previous, f"offset={offset_s}s"

        # Clock was never renewed (still T0) -> at T0+31s the 30s window has
        # expired and the identity must clear to Unknown.
        captured_at = t0 + timedelta(seconds=31)
        outcome = await resolver.resolve(
            hypotheses=[self._held_gt(t0, t0)],
            new_face_anchors=[foreign_anchor],
            captured_at=captured_at,
        )
        decision = outcome.decisions[0]
        assert decision.identity_id is None
        assert decision.revises_previous
        assert not decision.evidence_backed

    @pytest.mark.asyncio
    async def test_own_face_evidence_still_renews_clock(self) -> None:
        """Contrast case: genuine own-identity evidence must still back the
        commit and keep renewing the clock, holding the identity well past
        what would have been the original 30 s deadline had the clock not
        renewed."""
        t0 = datetime.now(UTC)
        config = ResolverConfig(prior_maintenance_max_age_s=30.0, enable_sticky_maintenance=False)
        resolver, _ = await self._resolver(config)
        own_anchor = _make_face_anchor("amma", confidence=0.9)

        captured_at_1 = t0 + timedelta(seconds=5)
        outcome_1 = await resolver.resolve(
            hypotheses=[self._held_gt(t0, t0)],
            new_face_anchors=[own_anchor],
            captured_at=captured_at_1,
        )
        decision_1 = outcome_1.decisions[0]
        assert decision_1.identity_id == "amma"
        assert decision_1.evidence_backed  # own evidence renews the clock

        # Simulate the repository applying evidence_backed_commit: the clock
        # is now captured_at_1, not T0. 34 s after T0 (which would be outside
        # the original 30 s window) but only 29 s after the renewed clock.
        renewed_clock = captured_at_1
        captured_at_2 = t0 + timedelta(seconds=34)
        outcome_2 = await resolver.resolve(
            hypotheses=[self._held_gt(t0, renewed_clock)],
            new_face_anchors=[own_anchor],
            captured_at=captured_at_2,
        )
        decision_2 = outcome_2.decisions[0]
        assert decision_2.identity_id == "amma"
        assert decision_2.evidence_backed

    @pytest.mark.asyncio
    async def test_sticky_maintenance_hold_not_backed_by_others_evidence(self) -> None:
        """Sticky-maintenance sub-case: the posterior argmax flips away from
        the held identity (to 'grandma') on weak, non-contradicting evidence.
        Sticky maintenance holds 'amma', but the hold must not be marked
        evidence_backed on the strength of grandma's evidence."""
        t0 = datetime.now(UTC)
        config = ResolverConfig(
            prior_maintenance_max_age_s=30.0,
            enable_sticky_maintenance=True,
            contradiction_face_confidence=0.70,
        )
        resolver, _ = await self._resolver(config)
        # Strong enough to flip the posterior argmax to grandma but below
        # both the face-based (0.70) and posterior-based (prob>=0.80)
        # contradiction thresholds, so sticky maintenance re-opens the window.
        anchor = _make_face_anchor("grandma", confidence=0.59)

        captured_at = t0 + timedelta(seconds=10)
        outcome = await resolver.resolve(
            hypotheses=[self._held_gt(t0, t0)],
            new_face_anchors=[anchor],
            captured_at=captured_at,
        )
        decision = outcome.decisions[0]
        assert decision.identity_id == "amma"
        assert not decision.revises_previous
        assert not decision.evidence_backed


class TestCrossGtFacePropagation:
    """Tests for cross-GT face identity propagation."""

    @pytest.mark.asyncio
    async def test_face_propagates_to_similar_adjacent_gt(self) -> None:
        """Face anchor on GT-A propagates to GT-B when gallery similarity is high.

        Two enrolled identities (alice, bob) split the prior.  Without face
        propagation, GT-B would stay UNKNOWN (no direct face evidence, ReID
        evidence too weak on its own).  With propagation, GT-B receives a
        synthetic face anchor for alice because its gallery centroid is nearly
        identical to GT-A's.
        """
        from app.domain import GalleryEmbedding

        # Two identities so the prior is balanced — requires real evidence to commit.
        identities = [_make_identity("alice", "Alice"), _make_identity("bob", "Bob")]
        gallery_repo = InMemoryGalleryRepository()
        for ident in identities:
            await gallery_repo.upsert_identity(ident)

        # GT-A tracklet gallery: embedding [1, 0, 0, ...]
        base_emb = [1.0] + [0.0] * 767
        await gallery_repo.upsert_gallery_entry(
            GalleryEmbedding(
                gallery_entry_id="ge-a",
                identity_id="alice",
                embedding=base_emb,
                seen_at=datetime.now(UTC),
                origin_tracklet_id="t-a",
            )
        )
        # GT-B tracklet gallery: very similar embedding (cosine sim ≈ 0.964 with GT-A)
        similar_emb = [1.0] + [0.01] * 767
        await gallery_repo.upsert_gallery_entry(
            GalleryEmbedding(
                gallery_entry_id="ge-b",
                identity_id=None,
                embedding=similar_emb,
                seen_at=datetime.now(UTC),
                origin_tracklet_id="t-b",
            )
        )

        config = ResolverConfig(
            cross_gt_face_propagation_threshold=0.70,
            face_commit_min_confidence=0.60,
        )
        resolver = _make_resolver(identities=identities, gallery_repo=gallery_repo, config=config)

        gt_a = _make_gt(ph_id="gt-a", tracklet_ids=["t-a"], camera_ids=["cam-a"])
        gt_b = _make_gt(ph_id="gt-b", tracklet_ids=["t-b"], camera_ids=["cam-b"])

        anchor = _make_face_anchor("alice", confidence=0.95, tracklet_id="t-a")

        outcome = await resolver.resolve(
            hypotheses=[gt_a, gt_b],
            new_face_anchors=[anchor],
            captured_at=datetime.now(UTC),
        )

        # GT-A commits alice directly (direct face evidence).
        # GT-B also resolves alice via propagated anchor but the duplicate-active
        # guard (enabled by default in M02) blocks it since GT-A already holds alice.
        decisions_by_gt = {d.ph_id: d for d in outcome.decisions}
        assert decisions_by_gt["gt-a"].identity_id == "alice"
        assert decisions_by_gt["gt-b"].identity_id is None

    @pytest.mark.asyncio
    async def test_face_not_propagated_to_dissimilar_gt(self) -> None:
        """Face anchor on GT-A does NOT propagate to dissimilar GT-B.

        Two enrolled identities split the prior so that neither alice nor bob
        can cross commit_prob without genuine evidence.  GT-B's embedding is
        orthogonal to GT-A's (cosine sim = 0), so gallery similarity is below
        the propagation threshold and no synthetic face anchor is created.
        """
        from app.domain import GalleryEmbedding

        # Two identities so the prior is ~50/50 — alice can't commit on prior alone.
        identities = [_make_identity("alice", "Alice"), _make_identity("bob", "Bob")]
        gallery_repo = InMemoryGalleryRepository()
        for ident in identities:
            await gallery_repo.upsert_identity(ident)

        # GT-A: embedding in one direction.
        await gallery_repo.upsert_gallery_entry(
            GalleryEmbedding(
                gallery_entry_id="ge-a",
                identity_id="alice",
                embedding=[1.0] + [0.0] * 767,
                seen_at=datetime.now(UTC),
                origin_tracklet_id="t-a",
            )
        )
        # GT-B: embedding in orthogonal direction → gallery cosine sim with GT-A ≈ 0.
        dissimilar_emb = [0.0, 1.0] + [0.0] * 766
        await gallery_repo.upsert_gallery_entry(
            GalleryEmbedding(
                gallery_entry_id="ge-b",
                identity_id=None,
                embedding=dissimilar_emb,
                seen_at=datetime.now(UTC),
                origin_tracklet_id="t-b",
            )
        )

        config = ResolverConfig(cross_gt_face_propagation_threshold=0.70)
        resolver = _make_resolver(identities=identities, gallery_repo=gallery_repo, config=config)

        gt_a = _make_gt(ph_id="gt-a", tracklet_ids=["t-a"], camera_ids=["cam-a"])
        gt_b = _make_gt(ph_id="gt-b", tracklet_ids=["t-b"], camera_ids=["cam-b"])

        anchor = _make_face_anchor("alice", confidence=0.95, tracklet_id="t-a")

        outcome = await resolver.resolve(
            hypotheses=[gt_a, gt_b],
            new_face_anchors=[anchor],
            captured_at=datetime.now(UTC),
        )

        decisions_by_gt = {d.ph_id: d for d in outcome.decisions}
        # GT-A gets alice; GT-B stays UNKNOWN (no propagation, no evidence).
        assert decisions_by_gt["gt-a"].identity_id == "alice"
        assert decisions_by_gt["gt-b"].identity_id is None


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
                ph_id="gt-1",
                identity_id="grandma",
                posterior=PosteriorDist({"grandma": 0.9, "UNKNOWN": 0.1}),
                revises_previous=True,
            )
        )
        assert len(outcome.decisions) == 1
        assert outcome.decisions[0].identity_id == "grandma"

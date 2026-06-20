"""M02 evidence clock and authority unit tests.

Covers the core identity integrity claims from M02:
- Evidence clock boundaries (30-second window)
- Prior-only decisions do not advance the clock
- ReID commits refresh the clock; height/propagated face do not
- ArcFace authority: fails closed without calibration; wins when calibrated
- ArcFace conflict → unknown
- Duplicate-active guard: weaker cleared; all cleared on tie; off-frame incumbent blocks
- Conflict overrides sticky maintenance and flip debounce
- In-memory repository parity for the three new identity operations
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.domain import (
    FaceAnchor,
    GalleryEmbedding,
    GlobalTrack,
    Identity,
    PersonHypothesis,
)
from app.inference.evidence import FaceEvidence
from app.storage.base import InMemoryGalleryRepository, InMemoryPHRepository
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_identity(identity_id: str, display_name: str = "") -> Identity:
    return Identity(
        identity_id=identity_id,
        display_name=display_name or identity_id,
        enrolled_at=datetime.now(UTC),
    )


def _make_resolver(
    *,
    identities: list[Identity] | None = None,
    config: ResolverConfig | None = None,
    gallery_repo: InMemoryGalleryRepository | None = None,
) -> IdentityResolver:
    return IdentityResolver(
        gallery_repo=gallery_repo or InMemoryGalleryRepository(),
        identities=identities or [],
        config=config or ResolverConfig(),
    )


def _gt_with_evidence(
    *,
    ph_id: str = "gt-1",
    identity_id: str = "alice",
    evidence_at: datetime,
    camera_ids: list[str] | None = None,
    tracklet_ids: list[str] | None = None,
) -> GlobalTrack:
    """Build a GlobalTrack with a committed identity and set evidence clock."""
    return GlobalTrack(
        global_track_id=ph_id,
        camera_ids=camera_ids or ["cam_a"],
        tracklet_ids=tracklet_ids or ["t1"],
        started_at=evidence_at,
        last_seen_at=evidence_at,
        current_identity_id=identity_id,
        current_identity_committed_at=evidence_at,
        last_independent_identity_evidence_at=evidence_at,
        state="active",
    )


def _face_anchor(
    person_id: str,
    *,
    confidence: float = 0.92,
    calibrated_confidence: float | None = None,
    tracklet_id: str = "t1",
) -> FaceAnchor:
    return FaceAnchor(
        person_id=person_id,
        confidence=confidence,
        quality=0.9,
        tracklet_id=tracklet_id,
        calibrated_confidence=calibrated_confidence,
    )


# ---------------------------------------------------------------------------
# 1. Evidence clock timing
# ---------------------------------------------------------------------------


class TestEvidenceClockBoundary:
    """Prior is held within the window and released at the boundary."""

    @pytest.mark.asyncio
    async def test_prior_held_at_29_9_seconds(self) -> None:
        now = datetime.now(UTC)
        evidence_at = now - timedelta(seconds=29.9)
        gt = _gt_with_evidence(evidence_at=evidence_at)

        resolver = _make_resolver(
            identities=[_make_identity("alice")],
            config=ResolverConfig(
                prior_maintenance_max_age_s=30.0,
                enable_sticky_maintenance=False,
            ),
        )

        outcome = await resolver.resolve(hypotheses=[gt], new_face_anchors=[], captured_at=now)
        assert outcome.decisions[0].identity_id == "alice"

    @pytest.mark.asyncio
    async def test_prior_released_at_30_1_seconds(self) -> None:
        now = datetime.now(UTC)
        evidence_at = now - timedelta(seconds=30.1)
        gt = _gt_with_evidence(evidence_at=evidence_at)

        resolver = _make_resolver(
            identities=[_make_identity("alice")],
            config=ResolverConfig(
                prior_maintenance_max_age_s=30.0,
                enable_sticky_maintenance=False,
            ),
        )

        outcome = await resolver.resolve(hypotheses=[gt], new_face_anchors=[], captured_at=now)
        assert outcome.decisions[0].identity_id is None

    @pytest.mark.asyncio
    async def test_no_evidence_clock_means_no_window(self) -> None:
        """PH with no evidence clock has no maintenance window, even within 30 s."""
        now = datetime.now(UTC)
        gt = GlobalTrack(
            global_track_id="gt-1",
            camera_ids=["cam_a"],
            tracklet_ids=["t1"],
            started_at=now - timedelta(seconds=5),
            last_seen_at=now,
            current_identity_id="alice",
            current_identity_committed_at=now - timedelta(seconds=5),
            last_independent_identity_evidence_at=None,
            state="active",
        )

        resolver = _make_resolver(
            identities=[_make_identity("alice")],
            config=ResolverConfig(
                prior_maintenance_max_age_s=30.0,
                enable_sticky_maintenance=False,
            ),
        )

        outcome = await resolver.resolve(hypotheses=[gt], new_face_anchors=[], captured_at=now)
        assert outcome.decisions[0].identity_id is None


# ---------------------------------------------------------------------------
# 2. Prior-only decisions do not advance the clock
# ---------------------------------------------------------------------------


class TestPriorOnlyNoClock:
    @pytest.mark.asyncio
    async def test_repeated_prior_decisions_do_not_advance_clock(self) -> None:
        """Making multiple prior-only decisions must not push the evidence clock forward."""
        ph_repo = InMemoryPHRepository()
        evidence_at = datetime.now(UTC) - timedelta(seconds=5)
        now = datetime.now(UTC)

        ph = PersonHypothesis(
            ph_id="ph-1",
            state_mean=(0.0, 0.0, 0.0, 0.0),
            state_cov=(1.0,) * 16,
            born_at=evidence_at,
            last_seen_at=now,
            last_seen_camera="cam-a",
            observation_count=3,
            current_identity_id="alice",
            current_identity_committed_at=evidence_at,
            last_independent_identity_evidence_at=evidence_at,
        )
        await ph_repo.save(ph)

        # prior_only_update: simulates what WorldTracker calls for prior-only decisions.
        await ph_repo.prior_only_update(
            ph_id="ph-1",
            identity_id="alice",
            committed_at=now,
        )

        stored = await ph_repo.get("ph-1")
        assert stored is not None
        # Identity label may be updated...
        assert stored.current_identity_id == "alice"
        # ...but evidence clock must NOT advance.
        assert stored.last_independent_identity_evidence_at == evidence_at

    @pytest.mark.asyncio
    async def test_clear_to_unknown_does_not_advance_clock(self) -> None:
        """clear_to_unknown removes the label but does not change the evidence clock."""
        ph_repo = InMemoryPHRepository()
        evidence_at = datetime.now(UTC) - timedelta(seconds=5)
        now = datetime.now(UTC)

        ph = PersonHypothesis(
            ph_id="ph-1",
            state_mean=(0.0, 0.0, 0.0, 0.0),
            state_cov=(1.0,) * 16,
            born_at=evidence_at,
            last_seen_at=now,
            last_seen_camera="cam-a",
            observation_count=3,
            current_identity_id="alice",
            current_identity_committed_at=evidence_at,
            last_independent_identity_evidence_at=evidence_at,
        )
        await ph_repo.save(ph)

        await ph_repo.clear_to_unknown(ph_id="ph-1", committed_at=now)

        stored = await ph_repo.get("ph-1")
        assert stored is not None
        assert stored.current_identity_id is None
        # Clock unchanged after clear.
        assert stored.last_independent_identity_evidence_at == evidence_at


# ---------------------------------------------------------------------------
# 3. Evidence-backed commit refreshes the clock
# ---------------------------------------------------------------------------


class TestEvidenceBackedRefreshes:
    @pytest.mark.asyncio
    async def test_evidence_backed_commit_refreshes_clock(self) -> None:
        """evidence_backed_commit writes new evidence timestamp."""
        ph_repo = InMemoryPHRepository()
        old_evidence_at = datetime.now(UTC) - timedelta(seconds=60)
        new_evidence_at = datetime.now(UTC)

        ph = PersonHypothesis(
            ph_id="ph-1",
            state_mean=(0.0, 0.0, 0.0, 0.0),
            state_cov=(1.0,) * 16,
            born_at=old_evidence_at,
            last_seen_at=new_evidence_at,
            last_seen_camera="cam-a",
            observation_count=3,
            last_independent_identity_evidence_at=old_evidence_at,
        )
        await ph_repo.save(ph)

        await ph_repo.evidence_backed_commit(
            ph_id="ph-1",
            identity_id="alice",
            evidence_at=new_evidence_at,
            committed_at=new_evidence_at,
        )

        stored = await ph_repo.get("ph-1")
        assert stored is not None
        assert stored.current_identity_id == "alice"
        assert stored.last_independent_identity_evidence_at == new_evidence_at


# ---------------------------------------------------------------------------
# 4. ArcFace authority
# ---------------------------------------------------------------------------


class TestArcFaceAuthority:
    @pytest.mark.asyncio
    async def test_arcface_authority_fails_closed_without_calibration(self) -> None:
        """With no calibrated_confidence, ArcFace anchor cannot create authority."""
        resolver = _make_resolver(
            identities=[_make_identity("alice")],
            config=ResolverConfig(arcface_authority_calibrated_confidence=0.80),
        )

        gt = GlobalTrack(
            global_track_id="gt-1",
            camera_ids=["cam_a"],
            tracklet_ids=["t1"],
            started_at=datetime.now(UTC),
            last_seen_at=datetime.now(UTC),
            current_identity_id=None,
            state="active",
        )
        # Anchor without calibrated_confidence (default None).
        anchor = _face_anchor("alice", confidence=0.95, calibrated_confidence=None)

        outcome = await resolver.resolve(
            hypotheses=[gt],
            new_face_anchors=[anchor],
            captured_at=datetime.now(UTC),
        )

        decision = outcome.decisions[0]
        # Authority was not triggered (no calibration). Alice may still be committed
        # via the standard face path, but reason must not say "arcface_authority".
        reason = decision.reason or ""
        assert "arcface_authority" not in reason

    @pytest.mark.asyncio
    async def test_arcface_authority_below_threshold_does_not_preempt(self) -> None:
        """calibrated_confidence below the threshold must not trigger authority."""
        resolver = _make_resolver(
            identities=[_make_identity("alice"), _make_identity("bob")],
            config=ResolverConfig(arcface_authority_calibrated_confidence=0.80),
        )

        gt = GlobalTrack(
            global_track_id="gt-1",
            camera_ids=["cam_a"],
            tracklet_ids=["t1"],
            started_at=datetime.now(UTC),
            last_seen_at=datetime.now(UTC),
            current_identity_id=None,
            state="active",
        )
        anchor = _face_anchor("alice", confidence=0.95, calibrated_confidence=0.75)

        outcome = await resolver.resolve(
            hypotheses=[gt],
            new_face_anchors=[anchor],
            captured_at=datetime.now(UTC),
        )

        reason = outcome.decisions[0].reason or ""
        assert "arcface_authority" not in reason

    @pytest.mark.asyncio
    async def test_arcface_authority_above_threshold_wins(self) -> None:
        """calibrated_confidence >= threshold triggers authority pre-emption."""
        identities = [_make_identity("alice"), _make_identity("bob")]
        gallery = InMemoryGalleryRepository()
        for ident in identities:
            await gallery.upsert_identity(ident)

        resolver = _make_resolver(
            identities=identities,
            gallery_repo=gallery,
            config=ResolverConfig(arcface_authority_calibrated_confidence=0.80),
        )

        gt = GlobalTrack(
            global_track_id="gt-1",
            camera_ids=["cam_a"],
            tracklet_ids=["t1"],
            started_at=datetime.now(UTC),
            last_seen_at=datetime.now(UTC),
            current_identity_id=None,
            state="active",
        )
        anchor = _face_anchor("alice", confidence=0.95, calibrated_confidence=0.85)

        outcome = await resolver.resolve(
            hypotheses=[gt],
            new_face_anchors=[anchor],
            captured_at=datetime.now(UTC),
        )

        decision = outcome.decisions[0]
        assert decision.identity_id == "alice"
        assert "arcface_authority" in (decision.reason or "")

    @pytest.mark.asyncio
    async def test_arcface_conflict_yields_unknown(self) -> None:
        """Two qualifying ArcFace anchors for the same PH with different identities → Unknown."""
        identities = [_make_identity("alice"), _make_identity("bob")]
        gallery = InMemoryGalleryRepository()
        for ident in identities:
            await gallery.upsert_identity(ident)

        resolver = _make_resolver(
            identities=identities,
            gallery_repo=gallery,
            config=ResolverConfig(arcface_authority_calibrated_confidence=0.80),
        )

        gt = GlobalTrack(
            global_track_id="gt-1",
            camera_ids=["cam_a"],
            tracklet_ids=["t1"],
            started_at=datetime.now(UTC),
            last_seen_at=datetime.now(UTC),
            current_identity_id=None,
            state="active",
        )
        anchor_alice = _face_anchor("alice", confidence=0.95, calibrated_confidence=0.88)
        anchor_bob = _face_anchor(
            "bob", confidence=0.94, calibrated_confidence=0.85, tracklet_id="t1"
        )

        outcome = await resolver.resolve(
            hypotheses=[gt],
            new_face_anchors=[anchor_alice, anchor_bob],
            captured_at=datetime.now(UTC),
        )

        decision = outcome.decisions[0]
        assert decision.identity_id is None
        assert "arcface_authority_conflict" in (decision.reason or "")

    @pytest.mark.asyncio
    async def test_arcface_conflict_overrides_sticky_maintenance(self) -> None:
        """ArcFace authority conflict overrides sticky maintenance and flip debounce."""
        now = datetime.now(UTC)
        identities = [_make_identity("alice"), _make_identity("bob")]
        gallery = InMemoryGalleryRepository()
        for ident in identities:
            await gallery.upsert_identity(ident)

        resolver = _make_resolver(
            identities=identities,
            gallery_repo=gallery,
            config=ResolverConfig(
                arcface_authority_calibrated_confidence=0.80,
                enable_sticky_maintenance=True,
                prior_maintenance_max_age_s=30.0,
            ),
        )

        gt = _gt_with_evidence(
            identity_id="alice",
            evidence_at=now - timedelta(seconds=5),
        )
        # Two conflicting calibrated anchors → conflict must clear alice.
        outcome = await resolver.resolve(
            hypotheses=[gt],
            new_face_anchors=[
                _face_anchor("alice", confidence=0.95, calibrated_confidence=0.88),
                _face_anchor("bob", confidence=0.93, calibrated_confidence=0.85),
            ],
            captured_at=now,
        )

        decision = outcome.decisions[0]
        assert decision.identity_id is None  # conflict → Unknown, not sticky-held alice


# ---------------------------------------------------------------------------
# 5. Duplicate-active guard
# ---------------------------------------------------------------------------


class TestDuplicateActiveGuard:
    @pytest.mark.asyncio
    async def test_stronger_contender_wins(self) -> None:
        """Weaker evidence contender is cleared; stronger holds the identity."""
        identities = [_make_identity("alice")]
        gallery = InMemoryGalleryRepository()
        await gallery.upsert_identity(identities[0])
        await gallery.upsert_gallery_entry(
            GalleryEmbedding(
                gallery_entry_id="ge-1",
                identity_id="alice",
                embedding=[1.0] + [0.0] * 767,
                seen_at=datetime.now(UTC),
            )
        )

        resolver = _make_resolver(
            identities=identities,
            gallery_repo=gallery,
            config=ResolverConfig(
                enable_duplicate_active_identity_guard=True,
                face_commit_min_confidence=0.60,
            ),
        )

        now = datetime.now(UTC)
        # gt-a: strong direct face evidence (evidence_backed=True).
        gt_a = GlobalTrack(
            global_track_id="gt-a",
            camera_ids=["cam_a"],
            tracklet_ids=["t1"],
            started_at=now,
            last_seen_at=now,
            current_identity_id=None,
            state="active",
        )
        # gt-b: no direct evidence.
        gt_b = GlobalTrack(
            global_track_id="gt-b",
            camera_ids=["cam_b"],
            tracklet_ids=["t2"],
            started_at=now,
            last_seen_at=now,
            current_identity_id=None,
            state="active",
        )

        outcome = await resolver.resolve(
            hypotheses=[gt_a, gt_b],
            new_face_anchors=[
                _face_anchor("alice", confidence=0.95, tracklet_id="t1"),
            ],
            captured_at=now,
        )

        by_id = {d.ph_id: d for d in outcome.decisions}
        # gt-a wins with direct face evidence.
        assert by_id["gt-a"].identity_id == "alice"
        # gt-b is blocked (no direct face evidence vs. gt-a's direct face).
        assert by_id["gt-b"].identity_id is None

    @pytest.mark.asyncio
    async def test_off_frame_incumbent_blocks_weak_new_assignment(self) -> None:
        """An off-frame incumbent blocks weak (below-holder-threshold) new assignments.

        The guard uses duplicate_identity_direct_face_min_confidence=0.90 to
        distinguish "holder" from "candidate".  A face anchor below 0.90 is a
        candidate and is blocked by the external incumbent.  An anchor at or
        above 0.90 is treated as a privileged holder (overrides the incumbent).
        """
        now = datetime.now(UTC)
        identities = [_make_identity("alice")]

        resolver = _make_resolver(
            identities=identities,
            config=ResolverConfig(
                enable_duplicate_active_identity_guard=True,
                face_commit_min_confidence=0.60,
                duplicate_identity_direct_face_min_confidence=0.90,
            ),
        )

        # gt-new is in this batch; open_ph_identities says alice is held off-frame.
        gt_new = GlobalTrack(
            global_track_id="gt-new",
            camera_ids=["cam_a"],
            tracklet_ids=["t1"],
            started_at=now,
            last_seen_at=now,
            current_identity_id=None,
            state="active",
        )

        outcome = await resolver.resolve(
            hypotheses=[gt_new],
            new_face_anchors=[
                # Below holder threshold (0.90) → candidate, blocked by external incumbent.
                _face_anchor("alice", confidence=0.75, tracklet_id="t1"),
            ],
            captured_at=now,
            open_ph_identities={"ph-incumbent": "alice"},
        )

        # New assignment blocked: alice is occupied by the off-frame incumbent.
        assert outcome.decisions[0].identity_id is None

    @pytest.mark.asyncio
    async def test_tied_new_contenders_all_blocked(self) -> None:
        """New evidence-tied contenders for the same identity are all blocked.

        When two GTs both want alice as a *new* assignment (revises_previous=True)
        and their evidence ranking is perfectly tied, the duplicate guard cannot
        pick a winner and blocks all of them.
        """
        from app.domain import GalleryEmbedding

        now = datetime.now(UTC)
        identities = [_make_identity("alice")]
        gallery = InMemoryGalleryRepository()
        await gallery.upsert_identity(identities[0])
        # Identical gallery entry accessible from both GTs.
        for i, tid in enumerate(["t1", "t2"]):
            await gallery.upsert_gallery_entry(
                GalleryEmbedding(
                    gallery_entry_id=f"ge-{i}",
                    identity_id="alice",
                    embedding=[1.0] + [0.0] * 767,
                    seen_at=now,
                    origin_tracklet_id=tid,
                )
            )

        resolver = _make_resolver(
            identities=identities,
            gallery_repo=gallery,
            config=ResolverConfig(
                enable_duplicate_active_identity_guard=True,
                face_commit_min_confidence=0.60,
                # Keep holder threshold high so neither face anchor qualifies.
                duplicate_identity_direct_face_min_confidence=0.99,
            ),
        )

        # Both GTs get the same face confidence — perfectly tied.
        gt_a = GlobalTrack(
            global_track_id="gt-a",
            camera_ids=["cam_a"],
            tracklet_ids=["t1"],
            started_at=now,
            last_seen_at=now,
            current_identity_id=None,
            state="active",
        )
        gt_b = GlobalTrack(
            global_track_id="gt-b",
            camera_ids=["cam_b"],
            tracklet_ids=["t2"],
            started_at=now,
            last_seen_at=now,
            current_identity_id=None,
            state="active",
        )

        outcome = await resolver.resolve(
            hypotheses=[gt_a, gt_b],
            new_face_anchors=[
                _face_anchor("alice", confidence=0.80, tracklet_id="t1"),
                _face_anchor("alice", confidence=0.80, tracklet_id="t2"),
            ],
            captured_at=now,
        )

        active_holders = [d for d in outcome.decisions if d.identity_id == "alice"]
        # Tie → duplicate guard clears all contenders; both blocked.
        assert len(active_holders) == 0


# ---------------------------------------------------------------------------
# 6. Resolver-level evidence routing: propagated face and height are not
#    qualifying evidence and must set evidence_backed=False on the decision.
# ---------------------------------------------------------------------------


class TestResolverEvidenceRouting:
    """Verify that the resolver marks evidence_backed correctly per source.

    ``evidence_backed=True``  → WorldTracker calls evidence_backed_commit (advances clock).
    ``evidence_backed=False`` → WorldTracker calls prior_only_update (clock unchanged).

    The crux of M02: propagated face and height are maintenance signals, not
    independent sightings.  Only direct_face and reid advance the evidence clock.
    """

    @pytest.mark.asyncio
    async def test_direct_face_sets_evidence_backed_true(self) -> None:
        """Direct ArcFace match → evidence_backed=True.

        In production, FaceIdentityStage creates both a FaceAnchor and a
        FaceEvidence(source='direct') for each real ArcFace call.  Both must be
        passed to resolve() for evidence_backed to be True.
        """
        now = datetime.now(UTC)
        resolver = _make_resolver(
            identities=[_make_identity("alice")],
            config=ResolverConfig(
                face_commit_min_confidence=0.60,
                enable_sticky_maintenance=False,
            ),
        )

        gt = GlobalTrack(
            global_track_id="gt-1",
            camera_ids=["cam_a"],
            tracklet_ids=["t1"],
            started_at=now,
            last_seen_at=now,
            current_identity_id=None,
            state="active",
        )
        outcome = await resolver.resolve(
            hypotheses=[gt],
            new_face_anchors=[_face_anchor("alice", confidence=0.90, tracklet_id="t1")],
            # FaceIdentityStage mirrors each anchor as FaceEvidence(source="direct").
            face_evidence=[
                FaceEvidence(
                    person_id="alice",
                    confidence=0.90,
                    tracklet_id="t1",
                    source="direct",
                    captured_at=now,
                )
            ],
            captured_at=now,
        )

        decision = outcome.decisions[0]
        assert decision.identity_id == "alice"
        assert decision.evidence_backed is True

    @pytest.mark.asyncio
    async def test_propagated_face_sets_evidence_backed_false(self) -> None:
        """Propagated face (source='propagated') must not set evidence_backed=True.

        WorldTracker must route propagated-face decisions to prior_only_update,
        not evidence_backed_commit, so the evidence clock is never advanced by
        cross-GT face propagation.
        """
        now = datetime.now(UTC)
        identities = [_make_identity("alice")]
        resolver = _make_resolver(
            identities=identities,
            config=ResolverConfig(
                face_commit_min_confidence=0.60,
                prior_maintenance_max_age_s=30.0,
                enable_sticky_maintenance=False,
            ),
        )

        # GT already holds alice within the evidence clock window.
        gt = GlobalTrack(
            global_track_id="gt-1",
            camera_ids=["cam_a"],
            tracklet_ids=["t1"],
            started_at=now - timedelta(seconds=5),
            last_seen_at=now,
            current_identity_id="alice",
            current_identity_committed_at=now - timedelta(seconds=5),
            last_independent_identity_evidence_at=now - timedelta(seconds=5),
            state="active",
        )

        # Propagated face (cross-GT) for alice — not a real ArcFace call.
        propagated = FaceEvidence(
            person_id="alice",
            confidence=0.95,
            tracklet_id="t1",
            source="propagated",
            captured_at=now,
        )

        outcome = await resolver.resolve(
            hypotheses=[gt],
            new_face_anchors=[],
            face_evidence=[propagated],
            captured_at=now,
        )

        decision = outcome.decisions[0]
        # Alice is maintained (prior window is open) but not evidence-backed.
        assert decision.identity_id == "alice"
        assert decision.evidence_backed is False

    @pytest.mark.asyncio
    async def test_height_alone_does_not_set_evidence_backed(self) -> None:
        """Height-proxy evidence alone must not produce evidence_backed=True.

        Height is an indirect demographic signal.  Without direct_face or reid
        evidence, evidence_backed must remain False even when height narrows the
        posterior toward a specific identity.
        """
        now = datetime.now(UTC)
        tall_alice = Identity(
            identity_id="alice",
            display_name="Alice",
            enrolled_at=now,
            height_mm=1700.0,
            height_sigma_mm=30.0,
        )
        resolver = _make_resolver(
            identities=[tall_alice],
            config=ResolverConfig(enable_sticky_maintenance=False),
        )

        gt = GlobalTrack(
            global_track_id="gt-1",
            camera_ids=["cam_a"],
            tracklet_ids=["t1"],
            started_at=now,
            last_seen_at=now,
            current_identity_id=None,
            state="active",
        )

        outcome = await resolver.resolve(
            hypotheses=[gt],
            new_face_anchors=[],
            captured_at=now,
            ph_heights={"gt-1": 1.70},  # matches alice's height exactly
        )

        decision = outcome.decisions[0]
        # Height alone cannot create an identity and cannot set evidence_backed.
        assert decision.evidence_backed is False


# ---------------------------------------------------------------------------
# 7. In-memory repository parity
# ---------------------------------------------------------------------------


class TestInMemoryRepoParity:
    @pytest.mark.asyncio
    async def test_three_operations_have_distinct_clock_semantics(self) -> None:
        """All three repo operations have distinct clock semantics."""
        repo = InMemoryPHRepository()
        base_time = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
        newer_time = base_time + timedelta(seconds=10)

        ph = PersonHypothesis(
            ph_id="ph-test",
            state_mean=(0.0, 0.0, 0.0, 0.0),
            state_cov=(1.0,) * 16,
            born_at=base_time,
            last_seen_at=newer_time,
            last_seen_camera="cam-a",
            observation_count=1,
            last_independent_identity_evidence_at=base_time,
        )
        await repo.save(ph)

        # evidence_backed_commit: refreshes clock.
        await repo.evidence_backed_commit(
            ph_id="ph-test",
            identity_id="alice",
            evidence_at=newer_time,
            committed_at=newer_time,
        )
        after_commit = await repo.get("ph-test")
        assert after_commit is not None
        assert after_commit.current_identity_id == "alice"
        assert after_commit.last_independent_identity_evidence_at == newer_time

        # prior_only_update: must NOT change clock.
        far_future = newer_time + timedelta(minutes=5)
        await repo.prior_only_update(
            ph_id="ph-test",
            identity_id="alice",
            committed_at=far_future,
        )
        after_prior = await repo.get("ph-test")
        assert after_prior is not None
        assert after_prior.last_independent_identity_evidence_at == newer_time  # unchanged

        # clear_to_unknown: must NOT change clock.
        await repo.clear_to_unknown(ph_id="ph-test", committed_at=far_future)
        after_clear = await repo.get("ph-test")
        assert after_clear is not None
        assert after_clear.current_identity_id is None
        assert after_clear.last_independent_identity_evidence_at == newer_time  # still unchanged

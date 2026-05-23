"""Tests for IdentityEvidence, posterior combiner, and commit policy."""

from __future__ import annotations

from datetime import UTC, datetime

from app.tracking.identity.commit_policy import (
    CommitPolicy,
    CommitPolicyState,
    evaluate_commit,
)
from app.tracking.identity.evidence import (
    CAN_CREATE_IDENTITY,
    CAN_SET_FACE_LOCK,
    IdentityEvidence,
)
from app.tracking.identity.posterior import (
    EvidencePosterior,
    combine_evidence,
)

_NOW = datetime.now(UTC)


# ---------------------------------------------------------------------------
# Evidence type tests
# ---------------------------------------------------------------------------


class TestIdentityEvidence:
    def test_direct_face_can_create_and_set_face_lock(self) -> None:
        ev = IdentityEvidence.direct_face(
            identity_id="alice", confidence=0.95, tracklet_id="tl-1", captured_at=_NOW
        )
        assert ev.can_create_identity
        assert ev.can_set_face_lock
        assert ev.source == "direct_face"

    def test_association_hint_cannot_create_or_set_face_lock(self) -> None:
        ev = IdentityEvidence.association_hint(
            identity_id="alice", confidence=0.8, tracklet_id="tl-1", captured_at=_NOW
        )
        assert not ev.can_create_identity
        assert not ev.can_set_face_lock
        assert ev.source == "association_hint"

    def test_temporal_prior_cannot_create_identity(self) -> None:
        ev = IdentityEvidence.temporal_prior(identity_id="alice", confidence=0.6)
        assert not ev.can_create_identity
        assert not ev.can_set_face_lock
        assert ev.source == "temporal_prior"

    def test_reid_can_create_but_not_set_face_lock(self) -> None:
        ev = IdentityEvidence.reid(identity_id="bob", confidence=0.75)
        assert ev.can_create_identity
        assert not ev.can_set_face_lock

    def test_operator_can_create_and_set_face_lock(self) -> None:
        ev = IdentityEvidence.operator_correction(identity_id="alice")
        assert ev.can_create_identity
        assert ev.can_set_face_lock

    def test_source_sets_are_correct(self) -> None:
        assert {"direct_face", "reid", "operator"} == CAN_CREATE_IDENTITY
        assert {"direct_face", "operator"} == CAN_SET_FACE_LOCK


# ---------------------------------------------------------------------------
# Posterior combiner tests
# ---------------------------------------------------------------------------


class TestPosteriorCombiner:
    def test_direct_face_beats_reid(self) -> None:
        """Direct face evidence should dominate ReID in the posterior."""
        evidence = [
            IdentityEvidence.direct_face("alice", 0.95, "tl-1", _NOW),
            IdentityEvidence.reid("bob", 0.75),
        ]
        ep = combine_evidence(evidence, known_identities={"alice", "bob"})

        assert ep.top_identity == "alice"
        assert ep.distribution["alice"] > ep.distribution["bob"]
        assert ep.face_evidence_present
        assert ep.reid_evidence_present

    def test_reid_only_commits_when_strong(self) -> None:
        """ReID alone can create an identity when strong enough."""
        evidence = [
            IdentityEvidence.reid("alice", 0.85, quality=1.0),
        ]
        ep = combine_evidence(evidence, known_identities={"alice", "bob"})

        assert ep.top_identity == "alice"
        assert ep.reid_evidence_present

    def test_temporal_prior_cannot_create_identity(self) -> None:
        """Temporal prior alone should not create an identity."""
        evidence = [
            IdentityEvidence.temporal_prior("alice", 0.8),
        ]
        ep = combine_evidence(evidence, known_identities={"alice"})

        # UNKNOWN should dominate when only prior is present.
        assert ep.distribution.get("UNKNOWN", 0) > 0

    def test_temporal_prior_maintains_identity(self) -> None:
        """Temporal prior with previous identity maintains it in posterior."""
        evidence: list[IdentityEvidence] = []
        ep = combine_evidence(
            evidence,
            known_identities={"alice"},
            previous_identity_id="alice",
        )

        # Prior gives some mass to the previous identity.
        assert ep.distribution.get("alice", 0) > 0

    def test_association_hint_cannot_set_face_lock(self) -> None:
        """Association hints must not be eligible for face lock."""
        ev = IdentityEvidence.association_hint("alice", 0.95, "tl-1", _NOW)
        assert not ev.can_set_face_lock

    def test_entropy_is_computed(self) -> None:
        """Posterior must have non-zero entropy with mixed evidence."""
        evidence = [
            IdentityEvidence.direct_face("alice", 0.7, "tl-1", _NOW),
            IdentityEvidence.reid("bob", 0.6),
        ]
        ep = combine_evidence(evidence, known_identities={"alice", "bob"})
        assert ep.entropy > 0
        assert ep.margin > 0


# ---------------------------------------------------------------------------
# Commit policy tests
# ---------------------------------------------------------------------------


class TestCommitPolicy:
    def test_strong_face_commits(self) -> None:
        """Strong direct face evidence should trigger a commit."""
        config = CommitPolicy(commit_prob=0.65, commit_margin=0.15)
        state = CommitPolicyState()
        ep = EvidencePosterior(
            distribution={"alice": 0.94, "UNKNOWN": 0.06},
            entropy=0.2,
            top_identity="alice",
            top_probability=0.94,
            margin=0.88,
            face_evidence_present=True,
            reid_evidence_present=False,
            evidence_summary={"direct_face": 1},
        )
        evidence = [
            IdentityEvidence.direct_face("alice", 0.95, "tl-1", _NOW),
        ]

        decision = evaluate_commit(
            posterior=ep,
            evidence_list=evidence,
            previous_identity_id=None,
            committed_at=None,
            captured_at=_NOW,
            global_track_id="gt-1",
            state=state,
            config=config,
        )

        assert decision.identity_id == "alice"
        assert decision.evidence_backed
        assert "face" in decision.reason

    def test_propagated_hint_cannot_set_face_lock(self) -> None:
        """Propagated/association hints must not create a face lock."""
        config = CommitPolicy(face_commit_min_confidence=0.7)
        state = CommitPolicyState()
        ep = EvidencePosterior(
            distribution={"alice": 0.94, "UNKNOWN": 0.06},
            entropy=0.2,
            top_identity="alice",
            top_probability=0.94,
            margin=0.88,
            face_evidence_present=False,
            reid_evidence_present=False,
            evidence_summary={"association_hint": 1},
        )
        evidence = [
            IdentityEvidence.association_hint("alice", 0.95, "tl-1", _NOW),
        ]

        evaluate_commit(
            posterior=ep,
            evidence_list=evidence,
            previous_identity_id=None,
            committed_at=None,
            captured_at=_NOW,
            global_track_id="gt-1",
            state=state,
            config=config,
        )

        # Association hint should NOT create a face lock.
        assert state.face_locks.get("gt-1") is None

    def test_direct_face_sets_face_lock(self) -> None:
        """Direct face evidence above threshold should set a face lock."""
        config = CommitPolicy(face_commit_min_confidence=0.7)
        state = CommitPolicyState()
        ep = EvidencePosterior(
            distribution={"alice": 0.94, "UNKNOWN": 0.06},
            entropy=0.2,
            top_identity="alice",
            top_probability=0.94,
            margin=0.88,
            face_evidence_present=True,
            reid_evidence_present=False,
            evidence_summary={"direct_face": 1},
        )
        evidence = [
            IdentityEvidence.direct_face("alice", 0.95, "tl-1", _NOW),
        ]

        evaluate_commit(
            posterior=ep,
            evidence_list=evidence,
            previous_identity_id=None,
            committed_at=None,
            captured_at=_NOW,
            global_track_id="gt-1",
            state=state,
            config=config,
        )

        lock = state.face_locks.get("gt-1")
        assert lock is not None
        assert lock.identity_id == "alice"

    def test_insufficient_evidence_returns_none(self) -> None:
        """Weak evidence should result in no commit."""
        config = CommitPolicy(commit_prob=0.65, commit_margin=0.15)
        state = CommitPolicyState()
        ep = EvidencePosterior(
            distribution={"alice": 0.50, "UNKNOWN": 0.50},
            entropy=1.0,
            top_identity="alice",
            top_probability=0.50,
            margin=0.0,
            face_evidence_present=False,
            reid_evidence_present=False,
            evidence_summary={},
        )
        evidence: list[IdentityEvidence] = []

        decision = evaluate_commit(
            posterior=ep,
            evidence_list=evidence,
            previous_identity_id=None,
            committed_at=None,
            captured_at=_NOW,
            global_track_id="gt-1",
            state=state,
            config=config,
        )

        assert decision.identity_id is None
        assert not decision.evidence_backed

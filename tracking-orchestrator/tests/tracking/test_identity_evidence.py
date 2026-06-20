"""Tests for IdentityEvidence, posterior combiners, and commit policy."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.domain import GlobalTrack, PosteriorDist
from app.tracking.identity.commit_policy import (
    CommitEvaluation,
    CommitPolicy,
    FaceLock,
    evaluate_commit,
)
from app.tracking.identity.evidence import (
    CAN_CREATE_IDENTITY,
    CAN_SET_FACE_LOCK,
    EvidenceSource,
    IdentityEvidence,
)
from app.tracking.identity.posterior import combine_evidence

_NOW = datetime.now(UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gt(
    ph_id: str = "gt-1",
    current_identity_id: str | None = None,
    current_identity_committed_at: datetime | None = None,
) -> GlobalTrack:
    return GlobalTrack(
        global_track_id=ph_id,
        camera_ids=["cam_a"],
        tracklet_ids=["tl-1"],
        started_at=_NOW,
        last_seen_at=_NOW,
        current_identity_id=current_identity_id,
        state="active",
        current_identity_committed_at=current_identity_committed_at,
    )


def _call_evaluate(
    entity: GlobalTrack,
    posterior: PosteriorDist,
    face_likelihood: PosteriorDist,
    reid_likelihood: PosteriorDist,
    *,
    config: CommitPolicy | None = None,
    face_locks: dict[str, FaceLock] | None = None,
    contradicted: bool = False,
) -> CommitEvaluation:
    return evaluate_commit(
        entity=entity,
        posterior=posterior,
        face_likelihood=face_likelihood,
        reid_likelihood=reid_likelihood,
        captured_at=_NOW,
        entity_quality=1.0,
        face_locks=face_locks if face_locks is not None else {},
        config=config or CommitPolicy(),
        contradicted=contradicted,
        enable_sticky_maintenance=False,
        enforce_quality_gate=False,
        enforce_flip_debounce=False,
    )


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
        assert ev.source == EvidenceSource.DIRECT_FACE

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
        expected_create = {EvidenceSource.DIRECT_FACE, EvidenceSource.REID, EvidenceSource.OPERATOR}
        assert expected_create == CAN_CREATE_IDENTITY
        assert {EvidenceSource.DIRECT_FACE, EvidenceSource.OPERATOR} == CAN_SET_FACE_LOCK

    def test_source_sets_equal_strings(self) -> None:
        # StrEnum: members compare equal to their wire strings.
        assert {"direct_face", "reid", "operator"} == CAN_CREATE_IDENTITY
        assert {"direct_face", "operator"} == CAN_SET_FACE_LOCK

    def test_evidence_source_is_str_subtype(self) -> None:
        assert EvidenceSource.DIRECT_FACE == "direct_face"
        assert isinstance(EvidenceSource.REID, str)


# ---------------------------------------------------------------------------
# Posterior combiner tests (additive combine_evidence path)
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
# Canonical commit policy tests
# ---------------------------------------------------------------------------


class TestCommitPolicy:
    def test_strong_face_commits(self) -> None:
        """Strong direct face evidence should trigger a commit."""
        entity = _make_gt(ph_id="gt-1", current_identity_id=None)
        posterior = PosteriorDist({"alice": 0.94, "UNKNOWN": 0.06})
        face_likelihood = PosteriorDist({"alice": 1.0})
        reid_likelihood = PosteriorDist({"UNKNOWN": 1.0})

        result = _call_evaluate(entity, posterior, face_likelihood, reid_likelihood)

        assert result.new_id == "alice"
        assert result.evidence_backed
        assert result.has_evidence

    def test_reid_evidence_commits(self) -> None:
        """ReID evidence in the likelihood marks has_evidence when it tops the posterior."""
        entity = _make_gt(ph_id="gt-1", current_identity_id=None)
        posterior = PosteriorDist({"alice": 0.80, "UNKNOWN": 0.20})
        face_likelihood = PosteriorDist({"UNKNOWN": 1.0})
        reid_likelihood = PosteriorDist({"alice": 0.85, "UNKNOWN": 0.15})

        result = _call_evaluate(entity, posterior, face_likelihood, reid_likelihood)

        assert result.new_id == "alice"
        assert result.has_evidence  # alice is in reid_likelihood

    def test_association_hint_not_has_evidence(self) -> None:
        """When top posterior identity is not in face or reid likelihood, has_evidence is False."""
        entity = _make_gt(ph_id="gt-1", current_identity_id=None)
        posterior = PosteriorDist({"alice": 0.94, "UNKNOWN": 0.06})
        # Neither face nor reid has alice — only prior-like signals drove the posterior.
        face_likelihood = PosteriorDist({"UNKNOWN": 1.0})
        reid_likelihood = PosteriorDist({"UNKNOWN": 1.0})

        result = _call_evaluate(entity, posterior, face_likelihood, reid_likelihood)

        # Posterior clears threshold but has_evidence=False because no face/reid backs it.
        assert result.new_id is None  # no has_evidence → no commit

    def test_insufficient_evidence_returns_none(self) -> None:
        """Weak evidence should result in no commit."""
        entity = _make_gt(ph_id="gt-1", current_identity_id=None)
        posterior = PosteriorDist({"alice": 0.50, "UNKNOWN": 0.50})
        face_likelihood = PosteriorDist({"UNKNOWN": 1.0})
        reid_likelihood = PosteriorDist({"UNKNOWN": 1.0})

        result = _call_evaluate(entity, posterior, face_likelihood, reid_likelihood)

        assert result.new_id is None
        assert not result.evidence_backed

    def test_dense_scene_raises_threshold(self) -> None:
        """Dense scene (≥2 identities with p>0.3) requires higher probability."""
        entity = _make_gt(ph_id="gt-1", current_identity_id=None)
        # Both alice (0.70) and bob (0.35) are strictly > 0.3 → dense.
        # PosteriorDist normalises on construction: 0.70/1.05 ≈ 0.667, 0.35/1.05 ≈ 0.333.
        posterior = PosteriorDist({"alice": 0.70, "bob": 0.35})
        face_likelihood = PosteriorDist({"alice": 0.90, "bob": 0.10})
        reid_likelihood = PosteriorDist({"UNKNOWN": 1.0})
        config = CommitPolicy(commit_prob=0.65, commit_prob_dense=0.80)

        result = _call_evaluate(
            entity, posterior, face_likelihood, reid_likelihood, config=config
        )

        # alice p ≈ 0.667 < 0.80 (dense threshold) → no commit.
        assert result.new_id is None
        assert result.effective_commit_prob == pytest.approx(0.80)

    def test_maintenance_window_holds_identity(self) -> None:
        """Within maintenance window, held identity persists even without evidence."""
        committed_at = _NOW
        entity = _make_gt(
            ph_id="gt-1",
            current_identity_id="alice",
            current_identity_committed_at=committed_at,
        )
        # Posterior still says alice but no face/reid → within_maintenance_window matters.
        posterior = PosteriorDist({"alice": 0.55, "UNKNOWN": 0.45})
        face_likelihood = PosteriorDist({"UNKNOWN": 1.0})
        reid_likelihood = PosteriorDist({"UNKNOWN": 1.0})
        config = CommitPolicy(prior_maintenance_max_age_s=120.0)

        result = _call_evaluate(
            entity, posterior, face_likelihood, reid_likelihood, config=config
        )

        assert result.new_id == "alice"
        assert result.within_maintenance_window

    def test_quality_gate_blocks_when_enforced(self) -> None:
        """Quality gate blocks low-quality new commits when enabled."""
        entity = _make_gt(ph_id="gt-1", current_identity_id=None)
        posterior = PosteriorDist({"alice": 0.94, "UNKNOWN": 0.06})
        face_likelihood = PosteriorDist({"alice": 1.0})
        reid_likelihood = PosteriorDist({"UNKNOWN": 1.0})
        config = CommitPolicy(min_quality_to_commit=0.50, enable_quality_gate=True)

        result = evaluate_commit(
            entity=entity,
            posterior=posterior,
            face_likelihood=face_likelihood,
            reid_likelihood=reid_likelihood,
            captured_at=_NOW,
            entity_quality=0.30,  # below threshold
            face_locks={},
            config=config,
            enable_sticky_maintenance=False,
            enforce_quality_gate=True,
            enforce_flip_debounce=False,
        )

        assert result.new_id is None
        assert result.quality_gate_blocked

    def test_quality_gate_shadow_does_not_block(self) -> None:
        """Quality gate shadow (enforce=False) records block but does not suppress commit."""
        entity = _make_gt(ph_id="gt-1", current_identity_id=None)
        posterior = PosteriorDist({"alice": 0.94, "UNKNOWN": 0.06})
        face_likelihood = PosteriorDist({"alice": 1.0})
        reid_likelihood = PosteriorDist({"UNKNOWN": 1.0})
        config = CommitPolicy(min_quality_to_commit=0.50, enable_quality_gate=False)

        result = evaluate_commit(
            entity=entity,
            posterior=posterior,
            face_likelihood=face_likelihood,
            reid_likelihood=reid_likelihood,
            captured_at=_NOW,
            entity_quality=0.30,
            face_locks={},
            config=config,
            enable_sticky_maintenance=False,
            enforce_quality_gate=False,
            enforce_flip_debounce=False,
        )

        assert result.new_id == "alice"  # shadow only, not blocked
        assert result.quality_gate_blocked  # still flagged for metrics

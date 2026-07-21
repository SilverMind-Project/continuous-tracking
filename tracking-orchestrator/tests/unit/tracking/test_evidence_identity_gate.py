"""Tests for identity-matched evidence gating (F1 / M01).

``collect_evidence_identity_ids`` and the ``has_evidence`` gate in
``evaluate_commit`` must test evidence *identity*, not evidence *presence*.
Smoothing mass spread across every enrolled identity, and evidence for a
different identity, must never make ``has_evidence`` true for the held
identity.
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.domain import FaceAnchor, GlobalTrack, PosteriorDist
from app.inference.evidence import FaceEvidence
from app.tracking.identity.commit_policy import (
    CommitPolicy,
    collect_evidence_identity_ids,
    evaluate_commit,
)

_NOW = datetime.now(UTC)


def _anchor(
    person_id: str,
    *,
    tracklet_id: str = "t1",
    detection_id: str = "",
    recognition_state: str = "recognized",
    confidence: float = 0.9,
    quality: float = 0.9,
    similarity: float = 0.9,
) -> FaceAnchor:
    return FaceAnchor(
        person_id=person_id,
        confidence=confidence,
        quality=quality,
        tracklet_id=tracklet_id,
        detection_id=detection_id,
        recognition_state=recognition_state,
        similarity=similarity,
    )


def _evidence(
    person_id: str,
    *,
    tracklet_id: str = "",
    detection_id: str = "t1",
    source: str = "direct",
) -> FaceEvidence:
    return FaceEvidence(
        person_id=person_id,
        confidence=0.9,
        tracklet_id=tracklet_id,
        detection_id=detection_id,
        source=source,  # type: ignore[arg-type]
    )


def _make_gt(
    *,
    current_identity_id: str | None = None,
) -> GlobalTrack:
    return GlobalTrack(
        global_track_id="gt-1",
        camera_ids=["cam_a"],
        tracklet_ids=["t1"],
        started_at=_NOW,
        last_seen_at=_NOW,
        current_identity_id=current_identity_id,
        state="active",
        current_identity_committed_at=_NOW if current_identity_id else None,
        last_independent_identity_evidence_at=_NOW if current_identity_id else None,
    )


# ---------------------------------------------------------------------------
# collect_evidence_identity_ids
# ---------------------------------------------------------------------------


class TestCollectEvidenceIdentityIds:
    def test_empty_inputs_yield_empty_set(self) -> None:
        result = collect_evidence_identity_ids(
            entity_obs_ids=frozenset(),
            entity_id="gt-1",
            face_anchors=[],
            face_evidence=[],
            reid_likelihood=PosteriorDist({}),
        )
        assert result == frozenset()

    def test_recognized_anchor_for_matched_entity_qualifies(self) -> None:
        anchor = _anchor("alice", tracklet_id="t1")
        result = collect_evidence_identity_ids(
            entity_obs_ids=frozenset({"t1"}),
            entity_id="gt-1",
            face_anchors=[anchor],
            face_evidence=[],
            reid_likelihood=PosteriorDist({}),
        )
        assert result == frozenset({"alice"})

    def test_unmatched_anchor_does_not_qualify(self) -> None:
        anchor = _anchor("alice", tracklet_id="some-other-tracklet")
        result = collect_evidence_identity_ids(
            entity_obs_ids=frozenset({"t1"}),
            entity_id="gt-1",
            face_anchors=[anchor],
            face_evidence=[],
            reid_likelihood=PosteriorDist({}),
        )
        assert result == frozenset()

    def test_propagated_evidence_record_excludes_the_anchor(self) -> None:
        anchor = _anchor("alice", tracklet_id="t1", detection_id="d1")
        evidence = _evidence("alice", detection_id="d1", source="propagated")
        result = collect_evidence_identity_ids(
            entity_obs_ids=frozenset({"t1"}),
            entity_id="gt-1",
            face_anchors=[anchor],
            face_evidence=[evidence],
            reid_likelihood=PosteriorDist({}),
        )
        assert result == frozenset()

    def test_direct_evidence_record_still_qualifies(self) -> None:
        anchor = _anchor("alice", tracklet_id="t1", detection_id="d1")
        evidence = _evidence("alice", detection_id="d1", source="direct")
        result = collect_evidence_identity_ids(
            entity_obs_ids=frozenset({"t1"}),
            entity_id="gt-1",
            face_anchors=[anchor],
            face_evidence=[evidence],
            reid_likelihood=PosteriorDist({}),
        )
        assert result == frozenset({"alice"})

    def test_candidate_anchor_included_for_commit_eligibility(self) -> None:
        anchor = _anchor("alice", tracklet_id="t1", recognition_state="candidate")
        result = collect_evidence_identity_ids(
            entity_obs_ids=frozenset({"t1"}),
            entity_id="gt-1",
            face_anchors=[anchor],
            face_evidence=[],
            reid_likelihood=PosteriorDist({}),
        )
        assert result == frozenset({"alice"})

    def test_candidate_anchor_excluded_when_recognized_only(self) -> None:
        anchor = _anchor("alice", tracklet_id="t1", recognition_state="candidate")
        result = collect_evidence_identity_ids(
            entity_obs_ids=frozenset({"t1"}),
            entity_id="gt-1",
            face_anchors=[anchor],
            face_evidence=[],
            reid_likelihood=PosteriorDist({}),
            recognized_only=True,
        )
        assert result == frozenset()

    def test_unrecognized_anchor_never_qualifies(self) -> None:
        anchor = _anchor("alice", tracklet_id="t1", recognition_state="unrecognized")
        result = collect_evidence_identity_ids(
            entity_obs_ids=frozenset({"t1"}),
            entity_id="gt-1",
            face_anchors=[anchor],
            face_evidence=[],
            reid_likelihood=PosteriorDist({}),
        )
        assert result == frozenset()

    def test_reid_argmax_above_floor_included(self) -> None:
        result = collect_evidence_identity_ids(
            entity_obs_ids=frozenset(),
            entity_id="gt-1",
            face_anchors=[],
            face_evidence=[],
            reid_likelihood=PosteriorDist({"alice": 0.5, "UNKNOWN": 0.5}),
        )
        assert result == frozenset({"alice"})

    def test_reid_argmax_unknown_excluded(self) -> None:
        result = collect_evidence_identity_ids(
            entity_obs_ids=frozenset(),
            entity_id="gt-1",
            face_anchors=[],
            face_evidence=[],
            reid_likelihood=PosteriorDist({"UNKNOWN": 1.0}),
        )
        assert result == frozenset()

    def test_reid_argmax_at_or_below_floor_excluded(self) -> None:
        result = collect_evidence_identity_ids(
            entity_obs_ids=frozenset(),
            entity_id="gt-1",
            face_anchors=[],
            face_evidence=[],
            reid_likelihood=PosteriorDist({"alice": 0.3, "UNKNOWN": 0.7}),
        )
        assert result == frozenset()

    def test_result_is_subset_of_anchor_and_reid_ids(self) -> None:
        anchors = [
            _anchor("alice", tracklet_id="t1"),
            _anchor("bob", tracklet_id="t1", recognition_state="candidate"),
        ]
        reid = PosteriorDist({"carol": 0.9, "UNKNOWN": 0.1})
        result = collect_evidence_identity_ids(
            entity_obs_ids=frozenset({"t1"}),
            entity_id="gt-1",
            face_anchors=anchors,
            face_evidence=[],
            reid_likelihood=reid,
        )
        allowed = {"alice", "bob", "carol"}
        assert result <= frozenset(allowed)


# ---------------------------------------------------------------------------
# evaluate_commit: has_evidence must be identity-matched, not presence-only
# ---------------------------------------------------------------------------


class TestEvaluateCommitIdentityMatchedEvidence:
    def test_foreign_evidence_does_not_back_held_identity(self) -> None:
        """A PH holding 'amma' with only 'grandma' evidence must not be
        evidence_backed for amma, even though the posterior smoothing puts
        some mass on every identity (the F1 bug this milestone closes)."""
        entity = _make_gt(current_identity_id="amma")
        # Posterior still favors amma via the temporal prior + smoothing mass,
        # but the only real evidence this frame named grandma.
        posterior = PosteriorDist({"amma": 0.55, "grandma": 0.35, "UNKNOWN": 0.10})
        face_likelihood = PosteriorDist({"grandma": 0.9, "amma": 0.02, "UNKNOWN": 0.08})
        reid_likelihood = PosteriorDist({"UNKNOWN": 1.0})

        result = evaluate_commit(
            entity=entity,
            posterior=posterior,
            face_likelihood=face_likelihood,
            reid_likelihood=reid_likelihood,
            captured_at=_NOW,
            entity_quality=1.0,
            config=CommitPolicy(),
            enable_sticky_maintenance=False,
            enforce_quality_gate=False,
            enforce_flip_debounce=False,
            evidence_identity_ids=frozenset({"grandma"}),
        )

        assert result.new_id == "amma"  # held via maintenance window
        assert not result.has_evidence
        assert not result.evidence_backed

    def test_own_evidence_backs_held_identity(self) -> None:
        entity = _make_gt(current_identity_id="amma")
        posterior = PosteriorDist({"amma": 0.90, "UNKNOWN": 0.10})
        face_likelihood = PosteriorDist({"amma": 0.9, "UNKNOWN": 0.1})
        reid_likelihood = PosteriorDist({"UNKNOWN": 1.0})

        result = evaluate_commit(
            entity=entity,
            posterior=posterior,
            face_likelihood=face_likelihood,
            reid_likelihood=reid_likelihood,
            captured_at=_NOW,
            entity_quality=1.0,
            config=CommitPolicy(),
            enable_sticky_maintenance=False,
            enforce_quality_gate=False,
            enforce_flip_debounce=False,
            evidence_identity_ids=frozenset({"amma"}),
        )

        assert result.new_id == "amma"
        assert result.has_evidence
        assert result.evidence_backed

    def test_distribution_membership_alone_no_longer_counts(self) -> None:
        """Regression guard: has_evidence must not be derivable from mere
        distribution membership (the pre-M01 bug). top_id is present in
        both likelihood distributions (smoothing mass) but evidence_identity_ids
        is empty, so has_evidence must be False."""
        entity = _make_gt(current_identity_id=None)
        posterior = PosteriorDist({"amma": 0.94, "UNKNOWN": 0.06})
        face_likelihood = PosteriorDist({"amma": 0.02, "grandma": 0.9, "UNKNOWN": 0.08})
        reid_likelihood = PosteriorDist({"amma": 0.05, "UNKNOWN": 0.95})

        result = evaluate_commit(
            entity=entity,
            posterior=posterior,
            face_likelihood=face_likelihood,
            reid_likelihood=reid_likelihood,
            captured_at=_NOW,
            entity_quality=1.0,
            config=CommitPolicy(),
            enable_sticky_maintenance=False,
            enforce_quality_gate=False,
            enforce_flip_debounce=False,
            # evidence_identity_ids omitted -> defaults to frozenset
        )

        assert not result.has_evidence
        assert result.new_id is None  # no commit without qualifying evidence

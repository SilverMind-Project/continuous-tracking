"""Commit policy: typed evaluation of identity assignment and maintenance.

Provides the canonical, pure commit evaluation function used by
``IdentityResolver``. All state (entity, config) is passed explicitly so
the function is testable without the full resolver.

``CommitPolicy`` is imported from ``policy.py`` and re-exported here for
backward compatibility with existing imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from structlog import get_logger

from ...domain import IdentityResolvableEntity, PosteriorDist
from .policy import CommitPolicy

logger = get_logger(__name__)

__all__ = [
    "CommitDecision",
    "CommitEvaluation",
    "CommitPolicy",
    "compute_contradiction",
    "evaluate_commit",
]


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommitEvaluation:
    """Pure commit-rule result before side effects and metric emission.

    Returned by ``evaluate_commit``; caller builds ``IdentityDecision`` and
    emits metrics from these fields.
    """

    new_id: str | None
    evidence_backed: bool
    has_evidence: bool
    within_maintenance_window: bool
    effective_commit_prob: float
    effective_commit_margin: float
    quality_gate_blocked: bool
    flip_debounce_blocked: bool


@dataclass(frozen=True)
class CommitDecision:
    """Higher-level commit result (additive-path compat, kept for reference).

    Not returned by the canonical ``evaluate_commit``. Kept so external code
    that depended on ``CommitDecision`` continues to import cleanly; callers
    should migrate to ``CommitEvaluation``.
    """

    identity_id: str | None
    reason: str
    evidence_backed: bool
    evidence_summary: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Contradiction helper
# ---------------------------------------------------------------------------


def compute_contradiction(
    *,
    prev_id: str | None,
    face_likelihood: PosteriorDist,
    best_face_confidence: float | None,
    top_id: str,
    top_prob: float,
    margin: float,
    config: CommitPolicy,
) -> bool:
    """Return True when evidence strongly contradicts a held identity.

    Two contradiction paths (either one is sufficient):
    1. A *recognized* face anchor names a different identity at or above
       ``contradiction_face_confidence``. Candidate and unrecognized faces
       cannot contradict — they are too weak to overturn a held identity.
    2. Posterior argmax is a different non-UNKNOWN identity clearing both
       ``contradiction_posterior_prob`` and ``contradiction_posterior_margin``.

    Returns False when ``prev_id`` is None (nothing held, nothing to contradict).
    """
    if prev_id is None:
        return False

    if (
        best_face_confidence is not None
        and best_face_confidence >= config.contradiction_face_confidence
        and face_likelihood.distribution
    ):
        face_top = max(face_likelihood.distribution, key=face_likelihood.distribution.__getitem__)
        if face_top != "UNKNOWN" and face_top != prev_id:
            return True

    return bool(
        top_id != prev_id
        and top_id != "UNKNOWN"
        and top_prob >= config.contradiction_posterior_prob
        and margin >= config.contradiction_posterior_margin
    )


# ---------------------------------------------------------------------------
# Canonical commit evaluation
# ---------------------------------------------------------------------------


def evaluate_commit(
    entity: IdentityResolvableEntity,
    posterior: PosteriorDist,
    face_likelihood: PosteriorDist,
    reid_likelihood: PosteriorDist,
    captured_at: datetime,
    entity_quality: float,
    config: CommitPolicy,
    *,
    contradicted: bool = False,
    enable_sticky_maintenance: bool,
    enforce_quality_gate: bool,
    enforce_flip_debounce: bool,
) -> CommitEvaluation:
    """Evaluate the commit rule without mutating state or emitting metrics.

    This is the single canonical implementation. ``IdentityResolver._commit``
    delegates here after computing ``contradicted``.

    Maintenance window is anchored to ``entity.last_independent_identity_evidence_at``
    (the evidence clock), never to ``current_identity_committed_at``. Prior-only
    decisions that advance ``current_identity_committed_at`` without refreshing the
    evidence clock therefore do not extend the maintenance window.

    Flip-debounce uses ``current_identity_committed_at`` because it guards against
    rapid label changes, not evidence staleness.

    Args:
        entity: The entity being evaluated (provides identity history).
        posterior: Combined Bayesian posterior.
        face_likelihood: Face-only distribution (used for evidence detection).
        reid_likelihood: ReID-only distribution (used for evidence detection).
        captured_at: Frame wall-clock time.
        entity_quality: Rolling crop-quality EMA for the entity.
        config: Full commit policy configuration.
        contradicted: Whether a strong contradiction was detected upstream.
        enable_sticky_maintenance: Hold identity on weak evidence when no
            contradiction exists.
        enforce_quality_gate: Block new commits below ``min_quality_to_commit``.
        enforce_flip_debounce: Block rapid flips that don't clear dense thresholds.

    Returns:
        ``CommitEvaluation`` with all fields needed to build ``IdentityDecision``.
    """
    (top_id, top_prob), margin = posterior.top_with_margin()
    has_evidence = (
        top_id in face_likelihood.distribution or top_id in reid_likelihood.distribution
    )

    prev_id = entity.current_identity_id
    identity_unchanged = top_id == prev_id and prev_id is not None
    within_maintenance_window = False

    if identity_unchanged:
        evidence_ts = entity.last_independent_identity_evidence_at
        if evidence_ts is not None:
            age_s = (captured_at - evidence_ts).total_seconds()
            within_maintenance_window = age_s <= config.prior_maintenance_max_age_s
        # If evidence_ts is None: entity was never backed by independent evidence;
        # the maintenance window is not open.

    # Sticky maintenance: extend the window when identity dips (posterior says
    # UNKNOWN or weak other) but the window has not expired and no strong
    # contradiction exists.
    if (
        enable_sticky_maintenance
        and prev_id is not None
        and not within_maintenance_window
        and not contradicted
    ):
        evidence_ts = entity.last_independent_identity_evidence_at
        if evidence_ts is not None:
            age_s = (captured_at - evidence_ts).total_seconds()
            if age_s <= config.prior_maintenance_max_age_s:
                within_maintenance_window = True

    evidence_ok = has_evidence or within_maintenance_window

    dense_candidates = sum(1 for p in posterior.distribution.values() if p > 0.3)
    is_dense = dense_candidates >= 2
    effective_commit_prob = config.commit_prob_dense if is_dense else config.commit_prob
    effective_commit_margin = config.commit_margin_dense if is_dense else config.commit_margin

    evidence_backed = False
    if within_maintenance_window:
        new_id: str | None = prev_id
        evidence_backed = has_evidence
    elif evidence_ok and top_prob >= effective_commit_prob and margin >= effective_commit_margin:
        new_id = top_id if top_id != "UNKNOWN" else None
        evidence_backed = has_evidence
    else:
        new_id = None

    quality_gate_blocked = (
        new_id is not None
        and new_id != prev_id
        and entity_quality < config.min_quality_to_commit
    )
    if quality_gate_blocked and enforce_quality_gate:
        new_id = None
        evidence_backed = False

    flip_debounce_blocked = False
    if (
        prev_id is not None
        and new_id is not None
        and new_id != prev_id
        and entity.current_identity_committed_at is not None
    ):
        age_s = (captured_at - entity.current_identity_committed_at).total_seconds()
        if age_s <= config.flip_debounce_window_s:
            clears_dense = (
                top_prob >= config.commit_prob_dense and margin >= config.commit_margin_dense
            )
            flip_debounce_blocked = not clears_dense
            if flip_debounce_blocked and enforce_flip_debounce:
                new_id = prev_id
                evidence_backed = False

    return CommitEvaluation(
        new_id=new_id,
        evidence_backed=evidence_backed,
        has_evidence=has_evidence,
        within_maintenance_window=within_maintenance_window,
        effective_commit_prob=effective_commit_prob,
        effective_commit_margin=effective_commit_margin,
        quality_gate_blocked=quality_gate_blocked,
        flip_debounce_blocked=flip_debounce_blocked,
    )

"""Commit policy: decides when to assign, maintain, or demote an identity.

Extracted from ``IdentityResolver._commit()`` so it can be tested in
isolation without the full resolver infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from structlog import get_logger

from .evidence import IdentityEvidence
from .posterior import EvidencePosterior

logger = get_logger(__name__)


@dataclass(frozen=True)
class CommitPolicy:
    """Thresholds and rules for identity commit decisions."""

    commit_prob: float = 0.65
    commit_margin: float = 0.15
    commit_prob_dense: float = 0.80
    commit_margin_dense: float = 0.20
    prior_maintenance_max_age_s: float = 120.0
    face_commit_min_confidence: float = 0.70
    face_lock_maintenance_max_age_s: float = 300.0


@dataclass
class CommitDecision:
    """Output of the commit policy for one global track in one frame."""

    identity_id: str | None
    reason: str
    evidence_backed: bool
    evidence_summary: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FaceLockState:
    identity_id: str
    confidence: float
    locked_at: datetime


@dataclass
class CommitPolicyState:
    """Mutable state maintained across frames by the commit policy."""

    face_locks: dict[str, _FaceLockState] = field(default_factory=dict)
    revision_log: dict[str, list[datetime]] = field(default_factory=dict)


def evaluate_commit(
    posterior: EvidencePosterior,
    evidence_list: list[IdentityEvidence],
    previous_identity_id: str | None,
    committed_at: datetime | None,
    captured_at: datetime,
    global_track_id: str,
    state: CommitPolicyState,
    config: CommitPolicy,
) -> CommitDecision:
    """Evaluate whether to commit, maintain, or demote an identity.

    Args:
        posterior: The combined posterior from evidence.
        evidence_list: All evidence items for this GT in this frame.
        previous_identity_id: The GT's current committed identity (if any).
        committed_at: When the current identity was last evidence-backed.
        captured_at: Wall-clock time of the current frame.
        global_track_id: The GT being evaluated.
        state: Mutable state (face locks, revision log) maintained across frames.
        config: Commit thresholds.

    Returns:
        CommitDecision with the resulting identity_id and reason.
    """
    face_locked_id = _get_face_locked_identity(state, global_track_id, captured_at, config)

    # Check if any evidence can create a new identity.
    has_creating_evidence = any(ev.can_create_identity for ev in evidence_list)

    # Check if any evidence justifies maintaining the existing identity.
    identity_unchanged = (
        posterior.top_identity == previous_identity_id and previous_identity_id is not None
    )

    # Maintenance window: face lock or standard committed_at window.
    within_maintenance = False
    if identity_unchanged:
        if face_locked_id == previous_identity_id:
            within_maintenance = True
        elif committed_at is not None:
            age_s = (captured_at - committed_at).total_seconds()
            within_maintenance = age_s <= config.prior_maintenance_max_age_s

    # Dense scene detection: more than 2 identities with posterior > 0.3.
    dense_count = sum(1 for p in posterior.distribution.values() if p > 0.3)
    is_dense = dense_count >= 2
    effective_prob = config.commit_prob_dense if is_dense else config.commit_prob
    effective_margin = config.commit_margin_dense if is_dense else config.commit_margin

    # Face lock management.
    _manage_face_lock(state, global_track_id, posterior, evidence_list, captured_at, config)

    evidence_ok = has_creating_evidence or within_maintenance

    if within_maintenance and identity_unchanged:
        # Carry the existing identity forward.
        return CommitDecision(
            identity_id=previous_identity_id,
            reason="maintained_by_prior",
            evidence_backed=has_creating_evidence,
            evidence_summary=posterior.evidence_summary,
        )

    if (
        evidence_ok
        and posterior.top_probability >= effective_prob
        and posterior.margin >= effective_margin
        and posterior.top_identity != "UNKNOWN"
    ):
        new_id = posterior.top_identity
        reason = "committed_by_evidence"
        if posterior.face_evidence_present:
            p_str = f"p={posterior.top_probability:.3f}, margin={posterior.margin:.3f}"
            reason = f"committed_by_face ({p_str})"
        elif posterior.reid_evidence_present:
            p_str = f"p={posterior.top_probability:.3f}, margin={posterior.margin:.3f}"
            reason = f"committed_by_reid ({p_str})"
        else:
            reason = f"committed (p={posterior.top_probability:.3f}, margin={posterior.margin:.3f})"
        return CommitDecision(
            identity_id=new_id,
            reason=reason,
            evidence_backed=True,
            evidence_summary=posterior.evidence_summary,
        )

    # No commit: stay as previous identity or default to None (UNKNOWN).
    if previous_identity_id and identity_unchanged and within_maintenance:
        return CommitDecision(
            identity_id=previous_identity_id,
            reason="maintained_by_prior",
            evidence_backed=False,
            evidence_summary=posterior.evidence_summary,
        )

    return CommitDecision(
        identity_id=None,
        reason="insufficient_evidence",
        evidence_backed=False,
        evidence_summary=posterior.evidence_summary,
    )


def _get_face_locked_identity(
    state: CommitPolicyState,
    global_track_id: str,
    captured_at: datetime,
    config: CommitPolicy,
) -> str | None:
    lock = state.face_locks.get(global_track_id)
    if lock is None:
        return None
    age_s = (captured_at - lock.locked_at).total_seconds()
    if age_s > config.face_lock_maintenance_max_age_s:
        return None
    return lock.identity_id


def _manage_face_lock(
    state: CommitPolicyState,
    global_track_id: str,
    posterior: EvidencePosterior,
    evidence_list: list[IdentityEvidence],
    captured_at: datetime,
    config: CommitPolicy,
) -> None:
    """Set, refresh, or clear face locks based on evidence.

    Only evidence with ``can_set_face_lock=True`` can set or refresh a
    face lock.  Propagated hints and association-derived evidence cannot.
    """
    # Find the best face-lock-eligible evidence.
    best_face: IdentityEvidence | None = None
    for ev in evidence_list:
        if (
            ev.can_set_face_lock
            and ev.identity_id
            and ev.confidence >= config.face_commit_min_confidence
            and (best_face is None or ev.confidence > best_face.confidence)
        ):
            best_face = ev

    existing_lock = state.face_locks.get(global_track_id)

    if best_face is not None:
        if existing_lock is None or existing_lock.identity_id == best_face.identity_id:
            state.face_locks[global_track_id] = _FaceLockState(
                identity_id=best_face.identity_id,  # type: ignore[arg-type]
                confidence=best_face.confidence,
                locked_at=captured_at,
            )
        else:
            # Different identity at sufficient confidence: displace the lock.
            logger.info(
                "face_lock_displaced",
                global_track_id=global_track_id,
                old_identity=existing_lock.identity_id,
                new_identity=best_face.identity_id,
                new_confidence=round(best_face.confidence, 3),
            )
            state.face_locks[global_track_id] = _FaceLockState(
                identity_id=best_face.identity_id,  # type: ignore[arg-type]
                confidence=best_face.confidence,
                locked_at=captured_at,
            )

"""Pure conflict classification and duplicate contender ranking.

Functions here are stateless: they receive posterior distributions and
identity lists, classify conflicts, and rank contenders. No I/O, no
logging side-effects.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...domain import PosteriorDist
from .commit_policy import CommitEvaluation
from .types import IdentityConflict


@dataclass(frozen=True)
class ConflictClassification:
    """Result of classifying one commit evaluation for a conflict type."""

    conflict: IdentityConflict
    detail: str


@dataclass(frozen=True)
class DuplicateContender:
    """One entity that claims an identity during the duplicate-active guard."""

    entity_id: str
    identity_id: str
    top_probability: float
    direct_face_confidence: float


def classify_commit_conflict(evaluation: CommitEvaluation) -> ConflictClassification:
    """Map a ``CommitEvaluation`` to its primary conflict classification.

    Priority: quality_gate > flip_debounce > insufficient_evidence > none.
    """
    if evaluation.quality_gate_blocked:
        return ConflictClassification(
            conflict=IdentityConflict.QUALITY_GATE,
            detail="quality_gate_blocked (entity_quality < min_quality_to_commit)",
        )
    if evaluation.flip_debounce_blocked:
        return ConflictClassification(
            conflict=IdentityConflict.FLIP_DEBOUNCE,
            detail=(
                f"flip_debounce_blocked "
                f"(p={evaluation.effective_commit_prob:.3f}, "
                f"m={evaluation.effective_commit_margin:.3f})"
            ),
        )
    if evaluation.new_id is None and not evaluation.within_maintenance_window:
        return ConflictClassification(
            conflict=IdentityConflict.INSUFFICIENT_EVIDENCE,
            detail=(
                f"insufficient_evidence "
                f"(needs p>={evaluation.effective_commit_prob:.3f}, "
                f"margin>={evaluation.effective_commit_margin:.3f})"
            ),
        )
    return ConflictClassification(conflict=IdentityConflict.NONE, detail="")


def rank_duplicate_contenders(
    contenders: list[DuplicateContender],
) -> list[DuplicateContender]:
    """Rank contenders for the same identity by evidence strength.

    Primary sort: ``direct_face_confidence`` descending.
    Secondary sort: ``top_probability`` descending.

    Callers use position 0 as the winner when ``direct_face_confidence``
    is clearly dominant; otherwise the guard sets all contenders to Unknown.
    """
    return sorted(
        contenders,
        key=lambda c: (c.direct_face_confidence, c.top_probability),
        reverse=True,
    )


def is_strongly_dominant(
    contenders: list[DuplicateContender],
    *,
    min_direct_face_confidence: float,
) -> bool:
    """Return True when the top-ranked contender has strong direct-face evidence.

    A winner is declared only when its ``direct_face_confidence`` is at or
    above ``min_direct_face_confidence``. Tie → False → all contenders
    become Unknown.
    """
    if len(contenders) < 2:
        return False
    ranked = rank_duplicate_contenders(contenders)
    return ranked[0].direct_face_confidence >= min_direct_face_confidence


def classify_posterior_conflict(
    posterior: PosteriorDist,
    *,
    contradiction_posterior_prob: float,
    contradiction_posterior_margin: float,
    prev_id: str | None,
) -> IdentityConflict:
    """Classify a posterior as a strong contradiction or normal competition."""
    if prev_id is None:
        return IdentityConflict.NONE

    (top_id, top_prob), margin = posterior.top_with_margin()
    if (
        top_id != prev_id
        and top_id != "UNKNOWN"
        and top_prob >= contradiction_posterior_prob
        and margin >= contradiction_posterior_margin
    ):
        return IdentityConflict.STRONG_CONTRADICTION

    if top_prob < 0.5:
        return IdentityConflict.WEAK_POSTERIOR

    return IdentityConflict.NONE

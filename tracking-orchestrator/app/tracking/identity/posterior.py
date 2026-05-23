"""Posterior combiner for identity evidence.

Consumes a list of ``IdentityEvidence`` and returns a normalized
posterior distribution with entropy, top margin, and evidence summary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ...domain import PosteriorDist
from .evidence import EvidenceSource, IdentityEvidence

# Weight multipliers per evidence source.
SOURCE_WEIGHTS: dict[EvidenceSource, float] = {
    "direct_face": 3.0,
    "reid": 1.0,
    "temporal_prior": 0.6,
    "height_proxy": 1.5,
    "operator": 5.0,
    "association_hint": 0.5,
}


@dataclass(frozen=True)
class EvidencePosterior:
    """Normalized posterior from combining identity evidence."""

    distribution: dict[str, float]
    entropy: float
    top_identity: str
    top_probability: float
    margin: float
    face_evidence_present: bool
    reid_evidence_present: bool
    evidence_summary: dict[str, int] = field(default_factory=dict)


def combine_evidence(
    evidence_list: list[IdentityEvidence],
    known_identities: set[str],
    previous_identity_id: str | None = None,
    unknown_mass: float = 0.05,
) -> EvidencePosterior:
    """Combine a list of identity evidence into a normalized posterior.

    Args:
        evidence_list: Evidence items for one global track in one frame.
        known_identities: Set of enrolled identity IDs.
        previous_identity_id: The GT's current committed identity (if any).
        unknown_mass: Minimum probability mass reserved for UNKNOWN.

    Returns:
        EvidencePosterior with normalized distribution and diagnostics.
    """
    if not evidence_list and not previous_identity_id:
        return EvidencePosterior(
            distribution={"UNKNOWN": 1.0},
            entropy=0.0,
            top_identity="UNKNOWN",
            top_probability=1.0,
            margin=1.0,
            face_evidence_present=False,
            reid_evidence_present=False,
        )

    # Separate evidence by source for the summary.
    source_counts: dict[str, int] = {}
    face_present = False
    reid_present = False

    # Accumulate weighted scores per identity.
    scores: dict[str, float] = {}
    total_weight = 0.0

    for ev in evidence_list:
        source_counts[ev.source] = source_counts.get(ev.source, 0) + 1
        if ev.source == "direct_face":
            face_present = True
        if ev.source == "reid":
            reid_present = True

        weight = SOURCE_WEIGHTS.get(ev.source, 1.0)
        weighted_conf = ev.confidence * ev.quality * weight

        if ev.identity_id:
            scores[ev.identity_id] = scores.get(ev.identity_id, 0.0) + weighted_conf
        total_weight += weighted_conf

    # Add temporal prior for previous identity.
    if previous_identity_id and previous_identity_id in known_identities:
        prior_weight = SOURCE_WEIGHTS["temporal_prior"] * 0.5
        scores[previous_identity_id] = scores.get(previous_identity_id, 0.0) + prior_weight
        total_weight += prior_weight

    # Add UNKNOWN with minimum mass, plus any evidence that didn't match.
    unknown_score = unknown_mass * total_weight
    scores["UNKNOWN"] = unknown_score

    # Add small mass to known identities not yet scored (smoothing).
    for kid in known_identities:
        if kid not in scores:
            scores[kid] = 0.001 * total_weight

    # Normalize.
    total = sum(scores.values())
    distribution = {"UNKNOWN": 1.0} if total <= 0 else {k: v / total for k, v in scores.items()}

    # Compute entropy.
    entropy = 0.0
    for prob in distribution.values():
        if prob > 0:
            entropy -= prob * math.log2(prob)

    # Top identity and margin.
    sorted_items = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
    top_id, top_prob = sorted_items[0]
    margin = top_prob - sorted_items[1][1] if len(sorted_items) > 1 else top_prob

    return EvidencePosterior(
        distribution=distribution,
        entropy=entropy,
        top_identity=top_id,
        top_probability=top_prob,
        margin=margin,
        face_evidence_present=face_present,
        reid_evidence_present=reid_present,
        evidence_summary=source_counts,
    )


def to_domain_posterior(ep: EvidencePosterior) -> PosteriorDist:
    """Convert an EvidencePosterior to the legacy PosteriorDist domain type."""
    return PosteriorDist(distribution=dict(ep.distribution))

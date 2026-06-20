"""In-memory evidence snapshot for identity decisions.

Constructs ``DecisionEvidence`` from the evidence ledger and likelihood
distributions. Transient in M01 — no persistence touches this module yet.
Persistence arrives in M04.
"""

from __future__ import annotations

from ...domain import PosteriorDist
from .evidence import EvidenceSource, IdentityEvidence
from .types import DecisionEvidence, EvidenceContribution


def build_decision_evidence(
    evidence_list: list[IdentityEvidence],
    face_likelihood: PosteriorDist,
    reid_likelihood: PosteriorDist,
    entity_quality: float | None = None,
    face_model_version: str | None = None,
    reid_model_version: str | None = None,
) -> DecisionEvidence:
    """Build an in-memory ``DecisionEvidence`` from one frame's ledger.

    Args:
        evidence_list: Evidence items produced by the resolver for one entity.
        face_likelihood: Face-only posterior distribution.
        reid_likelihood: ReID-only posterior distribution.
        entity_quality: Rolling crop-quality EMA.
        face_model_version: ArcFace model version tag, if known.
        reid_model_version: ReID model version tag, if known.

    Returns:
        ``DecisionEvidence`` snapshot. Transient; caller owns lifetime.
    """
    direct_face_conf: float | None = None
    propagated_face_conf: float | None = None
    prior_conf: float | None = None

    contributions: list[EvidenceContribution] = []

    for ev in evidence_list:
        if ev.source == EvidenceSource.DIRECT_FACE:
            if direct_face_conf is None or ev.confidence > direct_face_conf:
                direct_face_conf = ev.confidence
        elif ev.source == EvidenceSource.ASSOCIATION_HINT:
            if propagated_face_conf is None or ev.confidence > propagated_face_conf:
                propagated_face_conf = ev.confidence
        elif ev.source == EvidenceSource.TEMPORAL_PRIOR:
            prior_conf = ev.confidence

        contributions.append(
            EvidenceContribution(
                source=str(ev.source),
                identity_id=ev.identity_id,
                weight=ev.quality,
                confidence=ev.confidence,
            )
        )

    # Top reid similarity from the likelihood distribution.
    reid_top: float | None = None
    if reid_likelihood.distribution:
        reid_top = max(reid_likelihood.distribution.values())

    # Height likelihood: take best non-UNKNOWN entry from the face likelihood
    # when height evidence would be present — placeholder for M04 wiring.
    height_ll: float | None = None
    if face_likelihood.distribution:
        candidates = {k: v for k, v in face_likelihood.distribution.items() if k != "UNKNOWN"}
        if candidates:
            height_ll = max(candidates.values())

    return DecisionEvidence(
        direct_face_confidence=direct_face_conf,
        propagated_face_confidence=propagated_face_conf,
        reid_top_similarity=reid_top,
        height_likelihood=height_ll,
        prior_confidence=prior_conf,
        entity_quality=entity_quality,
        top_contributions=tuple(
            sorted(contributions, key=lambda c: c.confidence, reverse=True)[:5]
        ),
        face_model_version=face_model_version,
        reid_model_version=reid_model_version,
    )

"""Identity evidence ledger entries.

Each evidence item carries a typed source, confidence, quality, and
enough provenance to distinguish direct ArcFace matches from propagated
hints or operator corrections.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class EvidenceSource(StrEnum):
    """Typed source of one identity evidence item.

    Wire values are stable; rename requires a migration. StrEnum membership
    means ``EvidenceSource.DIRECT_FACE == "direct_face"`` is True, so
    existing string comparisons continue to work unchanged.
    """

    DIRECT_FACE = "direct_face"
    REID = "reid"
    TEMPORAL_PRIOR = "temporal_prior"
    HEIGHT_PROXY = "height_proxy"
    OPERATOR = "operator"
    ASSOCIATION_HINT = "association_hint"
    CC_ASSERTION = "cc_assertion"


# Evidence sources that can justify a new identity assignment.
CAN_CREATE_IDENTITY: set[EvidenceSource] = {
    EvidenceSource.DIRECT_FACE,
    EvidenceSource.REID,
    EvidenceSource.OPERATOR,
}


@dataclass(frozen=True)
class IdentityEvidence:
    """One piece of identity evidence for a global track.

    The posterior combiner consumes a list of these.  Sources have
    different weights: ``direct_face`` has the highest weight,
    ``association_hint`` and ``temporal_prior`` cannot create a new
    identity, and ``temporal_prior`` can only maintain an existing one.
    """

    source: EvidenceSource
    identity_id: str | None
    confidence: float
    quality: float = 1.0
    captured_at: datetime | None = None
    tracklet_id: str | None = None
    metadata: Mapping[str, str | int | float | bool] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def direct_face(
        cls,
        identity_id: str,
        confidence: float,
        tracklet_id: str,
        captured_at: datetime,
        quality: float = 1.0,
    ) -> IdentityEvidence:
        """Create direct face evidence from an ArcFace service call."""
        return cls(
            source=EvidenceSource.DIRECT_FACE,
            identity_id=identity_id,
            confidence=confidence,
            quality=quality,
            captured_at=captured_at,
            tracklet_id=tracklet_id,
        )

    @classmethod
    def association_hint(
        cls,
        identity_id: str,
        confidence: float,
        tracklet_id: str,
        captured_at: datetime,
    ) -> IdentityEvidence:
        """Create an association hint (propagated or duplicate-view)."""
        return cls(
            source=EvidenceSource.ASSOCIATION_HINT,
            identity_id=identity_id,
            confidence=confidence,
            quality=0.5,  # reduced weight for non-direct evidence
            captured_at=captured_at,
            tracklet_id=tracklet_id,
        )

    @classmethod
    def reid(
        cls,
        identity_id: str,
        confidence: float,
        quality: float = 1.0,
    ) -> IdentityEvidence:
        """Create ReID-based evidence from gallery search."""
        return cls(
            source=EvidenceSource.REID,
            identity_id=identity_id,
            confidence=confidence,
            quality=quality,
        )

    @classmethod
    def temporal_prior(
        cls,
        identity_id: str | None,
        confidence: float,
    ) -> IdentityEvidence:
        """Create temporal prior evidence from previous assignment."""
        return cls(
            source=EvidenceSource.TEMPORAL_PRIOR,
            identity_id=identity_id,
            confidence=confidence,
            quality=0.3,  # prior alone is weak
        )

    @classmethod
    def cc_assertion(
        cls,
        identity_id: str,
        confidence: float,
        tracklet_id: str,
        captured_at: datetime,
        quality: float = 0.5,
    ) -> IdentityEvidence:
        """Create external evidence from a matched cc.identity_assertions anchor.

        Distinct source so the replay evaluator and reid-disagreement metrics
        can segment external evidence from native ArcFace matches
        (identity-continuity M09).
        """
        return cls(
            source=EvidenceSource.CC_ASSERTION,
            identity_id=identity_id,
            confidence=confidence,
            quality=quality,
            captured_at=captured_at,
            tracklet_id=tracklet_id,
        )

    @classmethod
    def operator_correction(
        cls,
        identity_id: str,
        confidence: float = 1.0,
    ) -> IdentityEvidence:
        """Create operator-applied identity correction evidence."""
        return cls(
            source=EvidenceSource.OPERATOR,
            identity_id=identity_id,
            confidence=confidence,
            quality=1.0,
        )

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    @property
    def can_create_identity(self) -> bool:
        """Return True when this evidence can justify a new identity assignment."""
        return self.source in CAN_CREATE_IDENTITY

    @property
    def can_advance_evidence_clock(self) -> bool:
        """Return True when this evidence qualifies to refresh the identity clock.

        Only direct face recognition and verified ReID advance the clock;
        propagated face (ASSOCIATION_HINT) and height proxy never do.
        """
        return self.source in {EvidenceSource.DIRECT_FACE, EvidenceSource.REID}

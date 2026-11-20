"""Central governance registry for face-anchor evidence grades.

Every governance rule that depends on where a :class:`FaceAnchor` came from
(``origin``) or how it was aggregated (``aggregation``) is declared once here.
Downstream resolver, commit-policy, tracker, dedup, and attachment code reads
``grade_of(anchor)`` instead of branching on raw origin or aggregation strings.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from app.domain import AnchorAggregation, AnchorOrigin, FaceAnchor


class EvidenceGradeName(StrEnum):
    """Closed vocabulary naming each registered evidence grade (also used as metric labels)."""

    NATIVE_FRAME = "native_frame"
    NATIVE_TEMPLATE = "native_template"
    CC_ASSERTION = "cc_assertion"
    WEARABLE = "wearable"


@dataclass(frozen=True)
class EvidenceGrade:
    """Governance flags for one ``(origin, aggregation)`` face-anchor kind."""

    name: EvidenceGradeName
    grants_authority: bool
    refreshes_evidence_clock: bool
    ranks_in_duplicate_guard: bool
    grants_duplicate_bypass: bool
    enters_association: bool
    enters_contradiction: bool
    propagates: bool
    requires_calibration: bool
    uses_own_capture_time: bool
    applies_missing_face_quality_default: bool
    attach_precedence: int
    likelihood_scale_key: str | None = None


NATIVE_FRAME = EvidenceGrade(
    name=EvidenceGradeName.NATIVE_FRAME,
    grants_authority=True,
    refreshes_evidence_clock=True,
    ranks_in_duplicate_guard=True,
    grants_duplicate_bypass=True,
    enters_association=True,
    enters_contradiction=True,
    propagates=True,
    requires_calibration=False,
    uses_own_capture_time=False,
    applies_missing_face_quality_default=True,
    attach_precedence=2,
    likelihood_scale_key=None,
)

NATIVE_TEMPLATE = EvidenceGrade(
    name=EvidenceGradeName.NATIVE_TEMPLATE,
    grants_authority=True,
    refreshes_evidence_clock=True,
    ranks_in_duplicate_guard=False,
    grants_duplicate_bypass=False,
    enters_association=False,
    enters_contradiction=False,
    propagates=False,
    requires_calibration=True,
    uses_own_capture_time=True,
    applies_missing_face_quality_default=True,
    attach_precedence=2,
    likelihood_scale_key=None,
)

CC_ASSERTION_FRAME = EvidenceGrade(
    name=EvidenceGradeName.CC_ASSERTION,
    grants_authority=False,
    refreshes_evidence_clock=False,
    ranks_in_duplicate_guard=False,
    grants_duplicate_bypass=False,
    enters_association=False,
    enters_contradiction=True,
    propagates=True,
    requires_calibration=False,
    uses_own_capture_time=False,
    applies_missing_face_quality_default=False,
    attach_precedence=1,
    likelihood_scale_key="cc_assertion_likelihood_scale",
)

WEARABLE = EvidenceGrade(
    name=EvidenceGradeName.WEARABLE,
    grants_authority=False,
    refreshes_evidence_clock=True,
    ranks_in_duplicate_guard=True,
    grants_duplicate_bypass=False,
    enters_association=False,
    enters_contradiction=False,
    propagates=False,
    requires_calibration=False,
    uses_own_capture_time=True,
    applies_missing_face_quality_default=False,
    attach_precedence=0,
    likelihood_scale_key=None,
)

# Convenience aliases for single-frame native and external assertion grades.
NATIVE = NATIVE_FRAME
CC_ASSERTION = CC_ASSERTION_FRAME

EVIDENCE_GRADES: Mapping[tuple[str, str], EvidenceGrade] = MappingProxyType(
    {
        (str(AnchorOrigin.NATIVE), str(AnchorAggregation.FRAME)): NATIVE_FRAME,
        (str(AnchorOrigin.NATIVE), str(AnchorAggregation.TEMPLATE)): NATIVE_TEMPLATE,
        (str(AnchorOrigin.CC_ASSERTION), str(AnchorAggregation.FRAME)): CC_ASSERTION_FRAME,
    }
)


def grade_of(anchor: FaceAnchor) -> EvidenceGrade:
    """Return the registered :class:`EvidenceGrade` for *anchor*; fail loudly if unknown."""
    key = (str(anchor.origin), str(anchor.aggregation))
    grade = EVIDENCE_GRADES.get(key)
    if grade is None:
        raise KeyError(
            f"Unknown FaceAnchor evidence grade for origin={anchor.origin!r}, "
            f"aggregation={anchor.aggregation!r}"
        )
    return grade

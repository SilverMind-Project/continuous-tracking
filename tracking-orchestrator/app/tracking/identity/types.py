"""Typed domain objects for identity decisions.

Provides enums and value objects that replace anonymous strings and dicts
across the identity subsystem. All types are frozen; no module-level
singletons. EvidenceSource is not redefined here — import it from
``evidence.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class IdentityAuthority(StrEnum):
    """Authority level that produced the effective identity.

    Values are stable wire strings; do not rename without a migration.
    Listed from weakest to strongest so comparisons work naturally.

    ``NONE`` and ``POSTERIOR`` were added by codebase-hardening M07 (F9):
    the producer previously left ordinary Bayesian commits with authority
    ``""`` and, worse, set the ArcFace-authority path's ``authority`` to the
    matched identity id rather than a level. ``UNKNOWN`` and ``HEIGHT_PROXY``
    are legacy/reserved members the current producer never emits.
    """

    NONE = "none"
    UNKNOWN = "unknown"
    TEMPORAL_PRIOR = "temporal_prior"
    POSTERIOR = "posterior"
    HEIGHT_PROXY = "height_proxy"
    REID_GALLERY = "reid_gallery"
    DIRECT_FACE = "direct_face"
    OPERATOR = "operator"


class IdentityConflict(StrEnum):
    """Typed conflict classification for one commit evaluation."""

    NONE = "none"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    QUALITY_GATE = "quality_gate_blocked"
    FLIP_DEBOUNCE = "flip_debounce_blocked"
    DUPLICATE_ACTIVE = "duplicate_active"
    STRONG_CONTRADICTION = "strong_contradiction"
    WEAK_POSTERIOR = "weak_posterior"


@dataclass(frozen=True)
class EvidenceContribution:
    """Summary of one evidence item's contribution to the posterior."""

    source: str  # EvidenceSource wire value
    identity_id: str | None
    weight: float
    confidence: float


@dataclass(frozen=True)
class DecisionEvidence:
    """In-memory snapshot of the evidence used for one identity decision.

    Transient in M01. Persistence arrives in M04.

    Carry this object alongside ``IdentityDecision`` so downstream
    consumers can inspect what drove the decision without re-running the
    resolver.
    """

    direct_face_confidence: float | None = None
    propagated_face_confidence: float | None = None
    reid_top_similarity: float | None = None
    height_likelihood: float | None = None
    prior_confidence: float | None = None
    entity_quality: float | None = None
    top_contributions: tuple[EvidenceContribution, ...] = field(default_factory=tuple)
    face_model_version: str | None = None
    reid_model_version: str | None = None

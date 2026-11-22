"""Identity resolution subsystem: evidence grades, channels, ledger, posterior, commit policy."""

from __future__ import annotations

from .channels import ChannelLikelihood, EvidenceChannel, ResolverInputs
from .commit_policy import (
    CommitEvaluation,
    CommitPolicy,
    compute_contradiction,
    evaluate_commit,
)
from .evidence import EvidenceSource, IdentityEvidence
from .evidence_grades import EvidenceGrade, EvidenceGradeName, grade_of
from .gallery_scoring import (
    GalleryScoringConfig,
    ScoredHit,
    aggregate_max_over_views,
    aggregate_mean,
    cap_votes,
    score_hits,
)
from .posterior import EvidencePosterior, combine_channels
from .types import IdentityAuthority

__all__ = [
    "ChannelLikelihood",
    "CommitEvaluation",
    "CommitPolicy",
    "EvidenceChannel",
    "EvidenceGrade",
    "EvidenceGradeName",
    "EvidencePosterior",
    "EvidenceSource",
    "GalleryScoringConfig",
    "IdentityAuthority",
    "IdentityEvidence",
    "ResolverInputs",
    "ScoredHit",
    "aggregate_max_over_views",
    "aggregate_mean",
    "cap_votes",
    "combine_channels",
    "compute_contradiction",
    "evaluate_commit",
    "grade_of",
    "score_hits",
]

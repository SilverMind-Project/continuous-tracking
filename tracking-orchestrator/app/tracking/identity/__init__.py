"""Identity resolution subsystem: evidence ledger, posterior, commit policy."""

from .commit_policy import (
    CommitEvaluation,
    CommitPolicy,
    compute_contradiction,
    evaluate_commit,
)
from .evidence import EvidenceSource, IdentityEvidence
from .gallery_scoring import (
    GalleryScoringConfig,
    ScoredHit,
    aggregate_max_over_views,
    aggregate_mean,
    cap_votes,
    score_hits,
)
from .posterior import EvidencePosterior, combine_posteriors
from .types import IdentityAuthority

__all__ = [
    "CommitEvaluation",
    "CommitPolicy",
    "EvidencePosterior",
    "EvidenceSource",
    "GalleryScoringConfig",
    "IdentityAuthority",
    "IdentityEvidence",
    "ScoredHit",
    "aggregate_max_over_views",
    "aggregate_mean",
    "cap_votes",
    "combine_posteriors",
    "compute_contradiction",
    "evaluate_commit",
    "score_hits",
]

"""Identity resolution subsystem: evidence ledger, posterior, commit policy."""

from .commit_policy import (
    CommitEvaluation,
    CommitPolicy,
    compute_contradiction,
    evaluate_commit,
)
from .evidence import CAN_CREATE_IDENTITY, EvidenceSource, IdentityEvidence
from .posterior import EvidencePosterior, combine_posteriors
from .types import IdentityAuthority

__all__ = [
    "CAN_CREATE_IDENTITY",
    "CommitEvaluation",
    "CommitPolicy",
    "EvidencePosterior",
    "EvidenceSource",
    "IdentityAuthority",
    "IdentityEvidence",
    "combine_posteriors",
    "compute_contradiction",
    "evaluate_commit",
]

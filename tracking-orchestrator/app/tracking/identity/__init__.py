"""Identity resolution subsystem: evidence ledger, posterior, commit policy."""

from .commit_policy import CommitDecision, CommitPolicy, CommitPolicyState, evaluate_commit
from .evidence import EvidenceSource, IdentityEvidence
from .gallery_governance import (
    GalleryEntryState,
    GalleryGovernanceConfig,
    GalleryGovernanceService,
)
from .posterior import EvidencePosterior, combine_evidence, to_domain_posterior

__all__ = [
    "CommitDecision",
    "CommitPolicy",
    "CommitPolicyState",
    "EvidencePosterior",
    "EvidenceSource",
    "GalleryEntryState",
    "GalleryGovernanceConfig",
    "GalleryGovernanceService",
    "IdentityEvidence",
    "combine_evidence",
    "evaluate_commit",
    "to_domain_posterior",
]

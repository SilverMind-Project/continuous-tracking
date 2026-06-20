"""Identity resolution subsystem: evidence ledger, posterior, commit policy."""

from .commit_policy import (
    CommitDecision,
    CommitEvaluation,
    CommitPolicy,
    compute_contradiction,
    evaluate_commit,
)
from .conflicts import (
    ConflictClassification,
    DuplicateContender,
    classify_commit_conflict,
    classify_posterior_conflict,
    rank_duplicate_contenders,
)
from .evidence import CAN_CREATE_IDENTITY, EvidenceSource, IdentityEvidence
from .gallery_governance import (
    GalleryEntryState,
    GalleryGovernanceConfig,
    GalleryGovernanceService,
)
from .policy import CommitPolicy  # noqa: F811 — re-exported from commit_policy too
from .posterior import EvidencePosterior, combine_evidence, combine_posteriors, to_domain_posterior
from .protocols import GalleryReadsProtocol, IdentityDecisionPersistenceProtocol
from .provenance import build_decision_evidence
from .types import (
    DecisionEvidence,
    EvidenceContribution,
    IdentityAuthority,
    IdentityConflict,
)

__all__ = [
    "CAN_CREATE_IDENTITY",
    "CommitDecision",
    "CommitEvaluation",
    "CommitPolicy",
    "ConflictClassification",
    "DecisionEvidence",
    "DuplicateContender",
    "EvidenceContribution",
    "EvidencePosterior",
    "EvidenceSource",
    "GalleryEntryState",
    "GalleryGovernanceConfig",
    "GalleryGovernanceService",
    "GalleryReadsProtocol",
    "IdentityAuthority",
    "IdentityConflict",
    "IdentityDecisionPersistenceProtocol",
    "IdentityEvidence",
    "build_decision_evidence",
    "classify_commit_conflict",
    "classify_posterior_conflict",
    "combine_evidence",
    "combine_posteriors",
    "compute_contradiction",
    "evaluate_commit",
    "rank_duplicate_contenders",
    "to_domain_posterior",
]

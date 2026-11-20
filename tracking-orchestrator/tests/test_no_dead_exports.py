"""Pins the ``app.tracking.identity`` package's re-export surface.

``__init__.py`` re-exports are how inert modules masquerade as API
(codebase-hardening M09, F11/F12). This test freezes the exact surviving
surface so any addition is a conscious, reviewed act rather than an
unnoticed re-export of a new module.
"""

from __future__ import annotations

import app.tracking.identity as identity_pkg

_EXPECTED_ALL = (
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
)


def test_identity_package_export_surface_is_pinned() -> None:
    assert tuple(identity_pkg.__all__) == _EXPECTED_ALL

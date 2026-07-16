"""Posterior combiner for identity evidence.

Consumes a list of ``IdentityEvidence`` and returns a normalized
posterior distribution with entropy, top margin, and evidence summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...domain import PosteriorDist


@dataclass(frozen=True)
class EvidencePosterior:
    """Normalized posterior from combining identity evidence."""

    distribution: dict[str, float]
    entropy: float
    top_identity: str
    top_probability: float
    margin: float
    face_evidence_present: bool
    reid_evidence_present: bool
    evidence_summary: dict[str, int] = field(default_factory=dict)


def combine_posteriors(
    prior: PosteriorDist,
    face: PosteriorDist,
    reid: PosteriorDist,
    height: PosteriorDist | None = None,
    *,
    face_weight_multiplier: float = 3.0,
    height_weight_multiplier: float = 1.5,
) -> PosteriorDist:
    """Combine prior, face likelihood, ReID likelihood, and optional height.

    Posterior = prior * face_likelihood * reid_likelihood * height_likelihood

    If any source is empty (no evidence), it is treated as uniform so it
    does not dilute evidence from the other sources.

    When a source is non-empty but missing an identity, a Laplace-style
    smoothing constant ``1 / (n + 1)`` is used instead of 1.0 to avoid
    penalising identities that *do* appear in that source.

    This is the resolver-canonical multiplicative combiner.
    """
    all_ids: set[str] = set(prior.distribution.keys())
    all_ids.update(face.distribution.keys())
    all_ids.update(reid.distribution.keys())
    if height is not None and height.distribution:
        all_ids.update(height.distribution.keys())

    if not all_ids:
        return PosteriorDist({"UNKNOWN": 1.0})

    def _w(dist: PosteriorDist, ident: str) -> float:
        if not dist.distribution:
            return 1.0
        if ident in dist.distribution:
            return dist.distribution[ident]
        n = len(dist.distribution)
        return 1.0 / (n + 1)

    combined: dict[str, float] = {}
    for ident in all_ids:
        fw = _w(face, ident)
        if ident in face.distribution:
            fw = fw * face_weight_multiplier

        hw = _w(height, ident) if height is not None else 1.0
        if height is not None and height.distribution and ident in height.distribution:
            hw = hw * height_weight_multiplier

        combined[ident] = _w(prior, ident) * fw * _w(reid, ident) * hw

    if not combined:
        return PosteriorDist({"UNKNOWN": 1.0})

    return PosteriorDist(combined)

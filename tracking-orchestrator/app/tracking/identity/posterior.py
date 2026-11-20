"""Posterior combiner for identity evidence channels.

Combines a temporal prior with an ordered sequence of :class:`ChannelLikelihood`
distributions into a single normalized :class:`PosteriorDist`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from app.domain import PosteriorDist
from app.tracking.identity.channels import ChannelLikelihood


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


def combine_channels(
    prior: PosteriorDist,
    channels: Sequence[ChannelLikelihood],
) -> PosteriorDist:
    """Combine a temporal prior with an ordered sequence of evidence channels.

    For each identity ``id`` in the union of ``prior`` and every non-empty
    channel distribution, multiplies ``prior(id)`` by each channel's factor in
    sequence order:

    - empty channel distribution: ``1.0`` (uniform, no dilution);
    - identity present in channel: ``dist[id] * channel.support_multiplier``;
    - identity absent from a non-empty channel: Laplace fill ``1 / (n + 1)``.

    Channel order is preserved during multiplication so floating-point results
    remain deterministic and match the canonical face -> gallery -> height order.
    """
    all_ids: set[str] = set(prior.distribution.keys())
    for ch in channels:
        if ch.distribution.distribution:
            all_ids.update(ch.distribution.distribution.keys())

    if not all_ids:
        return PosteriorDist({"UNKNOWN": 1.0})

    def _prior_weight(ident: str) -> float:
        if not prior.distribution:
            return 1.0
        if ident in prior.distribution:
            return prior.distribution[ident]
        return 1.0 / (len(prior.distribution) + 1)

    # Sorted so the combined distribution's key order, and anything that iterates it, does not
    # depend on set iteration order (and so on PYTHONHASHSEED).
    combined: dict[str, float] = {}
    for ident in sorted(all_ids):
        weight = _prior_weight(ident)
        for ch in channels:
            dist = ch.distribution.distribution
            if not dist:
                ch_w = 1.0
            elif ident in dist:
                ch_w = dist[ident] * ch.support_multiplier
            else:
                ch_w = 1.0 / (len(dist) + 1)
            weight = weight * ch_w
        combined[ident] = weight

    if not combined:
        return PosteriorDist({"UNKNOWN": 1.0})

    return PosteriorDist(combined)

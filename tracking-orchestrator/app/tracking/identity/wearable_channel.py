"""Wearable identity evidence channel for the Bayesian resolver.

Converts an accepted :class:`WearableEvidence` pairing into a full-support
power-weighted :class:`ChannelLikelihood`. Because every enrolled identity
plus ``UNKNOWN`` is populated in the distribution, the channel applies
``wearable.weight_multiplier`` as a log-likelihood power (``p ** weight``)
before ``PosteriorDist`` normalization and emits ``support_multiplier = 1.0``
so normalization cannot cancel the weight.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from datetime import datetime

from app.domain import (
    Identity,
    IdentityResolvableEntity,
    PosteriorDist,
    WearableEvidence,
    WearableEvidenceClass,
)
from app.tracking.identity.channels import ChannelLikelihood, ResolverInputs
from app.tracking.identity.evidence_grades import WEARABLE
from app.tracking.world.wearable_config import WearableConfig


def wearable_evidence_probability(evidence: WearableEvidence, cfg: WearableConfig) -> float:
    """Map an accepted wearable pairing's ``window_llr`` to ``p_w`` with class cap."""
    cap = (
        cfg.p_max if evidence.evidence_class == WearableEvidenceClass.TRAJECTORY else cfg.p_max_room
    )
    z = (evidence.window_llr - cfg.llr_mid) / cfg.llr_scale
    if z >= 40.0:
        raw = 1.0
    elif z <= -40.0:
        raw = 0.0
    else:
        raw = 1.0 / (1.0 + math.exp(-z))
    return min(cap, max(1e-6, raw))


class WearableChannel:
    """Evidence channel turning accepted wearable associations into likelihoods."""

    name: str = "wearable"

    def __init__(
        self,
        *,
        config: WearableConfig,
        identities_getter: Callable[[], Mapping[str, Identity]],
    ) -> None:
        self._config = config
        self._identities_getter = identities_getter

    def evaluate_wearable(
        self,
        entity: IdentityResolvableEntity,
        evidence: WearableEvidence | None,
    ) -> tuple[ChannelLikelihood, float]:
        """Build the wearable :class:`ChannelLikelihood` and scalar ``p_w`` for *entity*."""
        if evidence is None:
            return (
                ChannelLikelihood(
                    channel=self.name,
                    distribution=PosteriorDist({}),
                    support_multiplier=1.0,
                    evidence_at=None,
                ),
                0.0,
            )

        p_w = wearable_evidence_probability(evidence, self._config)
        identities = self._identities_getter()
        other_ids = [cid for cid in identities if cid != evidence.identity_id]
        other_slots = len(other_ids) + 1  # other enrolled identities + UNKNOWN
        p_other = max(1e-9, (1.0 - p_w) / other_slots)

        weight = self._config.weight_multiplier
        dist: dict[str, float] = {
            evidence.identity_id: p_w**weight,
            "UNKNOWN": p_other**weight,
        }
        for cid in other_ids:
            dist[cid] = p_other**weight

        if WEARABLE.refreshes_evidence_clock:
            ev_at = evidence.newest_fix_at
            clock_ids = frozenset({evidence.identity_id})
            ev_by_id: dict[str, datetime] = {evidence.identity_id: ev_at}
        else:
            ev_at = None
            clock_ids = frozenset()
            ev_by_id = {}

        return (
            ChannelLikelihood(
                channel=self.name,
                distribution=PosteriorDist(dist),
                support_multiplier=1.0,
                evidence_at=ev_at,
                clock_evidence_ids=clock_ids,
                evidence_at_by_identity=ev_by_id,
            ),
            p_w,
        )

    async def evaluate(
        self,
        entity: IdentityResolvableEntity,
        inputs: ResolverInputs,
    ) -> ChannelLikelihood:
        """Implement :class:`EvidenceChannel`."""
        likelihood, _ = self.evaluate_wearable(
            entity, inputs.wearable_evidence.get(entity.entity_id)
        )
        return likelihood

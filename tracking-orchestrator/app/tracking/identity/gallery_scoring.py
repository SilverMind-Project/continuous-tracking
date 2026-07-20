"""Single gallery-vote scorer shared by every ReID gallery query path.

Pure functions only: no I/O, no ``app.observability`` import (the backstop
counter and the logistic curve are injected by the caller). Every path that
queries the ReID gallery (multiview, single-query fallback, shadow) must
route hits through ``score_hits`` -> ``cap_votes`` -> one of the aggregate
functions. Inline scoring in the resolver is a defect (identity-continuity
M01, findings V1/V2).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from ...domain import GalleryEmbedding

_VERIFIED_STATE = "operator_verified"
_AUTO_VERIFIED_STATE = "auto_verified"


@dataclass(frozen=True)
class GalleryScoringConfig:
    """Trust, decay, and boost constants for gallery vote scoring."""

    verified_trust_multiplier: float = 2.0
    # Used from M02 onward; harmless today since no gallery entry can be in
    # the auto_verified state yet (the state does not exist until M02).
    auto_verified_trust_multiplier: float = 1.5
    recency_half_life_days: float = 7.0
    identified_entry_boost_min_sim: float = 0.65
    identified_entry_min_likelihood: float = 0.80


@dataclass(frozen=True)
class ScoredHit:
    """One gallery hit after trust, recency, and identified-entry scoring."""

    entry: GalleryEmbedding
    similarity: float
    logit: float
    trust_multiplier: float
    recency_factor: float
    weighted_logit: float
    # True when the identified-entry boost raised this hit's logit to the
    # floor. Downstream aggregation uses this to decide whether to spread
    # the non-match floor across every enrolled identity.
    boosted: bool


def _trust_multiplier(
    state: str,
    cfg: GalleryScoringConfig,
    on_nonvoting_state: Callable[[], None],
) -> float:
    """Map a gallery entry's review state to its vote trust multiplier.

    The pending/rejected-never-vote invariant is enforced by the gallery
    SQL filter (only operator_verified and auto_verified rows are ever
    returned to a resolver query); this multiplier never zeroes a vote so a
    query regression that leaks a non-voting row is loud (the backstop
    counter fires) rather than silently miscounted.
    """
    if state == _VERIFIED_STATE:
        return cfg.verified_trust_multiplier
    if state == _AUTO_VERIFIED_STATE:
        return cfg.auto_verified_trust_multiplier
    on_nonvoting_state()
    return 1.0


def _recency_factor(seen_at: datetime, now: datetime, half_life_days: float) -> float:
    seen = seen_at if seen_at.tzinfo else seen_at.replace(tzinfo=UTC)
    age_days = (now - seen).total_seconds() / 86400.0
    if age_days < 0:
        return 1.0
    return float(2.0 ** (-age_days / half_life_days))


def score_hits(
    hits: Sequence[tuple[GalleryEmbedding, float]],
    *,
    now: datetime,
    cfg: GalleryScoringConfig,
    logistic: Callable[[float], float],
    on_nonvoting_state: Callable[[], None],
) -> list[ScoredHit]:
    """Score raw gallery hits: logistic curve, identified-entry boost, trust, recency.

    The identified-entry boost is applied to the logit before trust and
    recency weighting, so both aggregation modes see identical per-hit
    boosted values and a verified boosted hit is not left unscaled by trust.
    """
    scored: list[ScoredHit] = []
    for entry, sim in hits:
        raw_logit = logistic(sim)
        logit = raw_logit
        boosted = False
        if (
            entry.identity_id is not None
            and sim >= cfg.identified_entry_boost_min_sim
            and raw_logit < cfg.identified_entry_min_likelihood
        ):
            logit = cfg.identified_entry_min_likelihood
            boosted = True

        trust = _trust_multiplier(entry.state, cfg, on_nonvoting_state)
        recency = _recency_factor(entry.seen_at, now, cfg.recency_half_life_days)
        weighted_logit = logit * trust * recency

        scored.append(
            ScoredHit(
                entry=entry,
                similarity=sim,
                logit=logit,
                trust_multiplier=trust,
                recency_factor=recency,
                weighted_logit=weighted_logit,
                boosted=boosted,
            )
        )
    return scored


def cap_votes(scored: Sequence[ScoredHit]) -> list[ScoredHit]:
    """Keep the strongest weighted hit per (identity, source_episode, camera, orientation).

    Prevents ten near-duplicate crops from one episode voting ten times.
    """
    best: dict[tuple[str | None, str, str, int], ScoredHit] = {}
    for hit in scored:
        entry = hit.entry
        key = (
            entry.identity_id,
            entry.source_episode_id or "",
            entry.camera_id or "",
            entry.orientation,
        )
        if key not in best or best[key].weighted_logit < hit.weighted_logit:
            best[key] = hit
    return list(best.values())


def _spread_non_match_floor(
    avg: dict[str, float],
    *,
    identities: Collection[str],
    cfg: GalleryScoringConfig,
) -> None:
    """Give every enrolled identity a non-zero floor once any hit was boosted.

    Without this, a boosted single-identity match would let PosteriorDist
    normalize to 100% for that identity even though no evidence about the
    other enrolled identities was gathered this round.
    """
    non_match_floor = (1.0 - cfg.identified_entry_min_likelihood) / max(len(identities), 1)
    for iid in identities:
        if iid not in avg:
            avg[iid] = non_match_floor
    if "UNKNOWN" not in avg:
        avg["UNKNOWN"] = non_match_floor


def aggregate_mean(
    capped: Sequence[ScoredHit],
    *,
    identities: Collection[str],
    cfg: GalleryScoringConfig,
) -> dict[str, float]:
    """Mean-of-weighted-logits aggregation (single-query fallback semantics)."""
    likelihood: dict[str, list[float]] = defaultdict(list)
    boosted = False
    for hit in capped:
        key = hit.entry.identity_id if hit.entry.identity_id else "UNKNOWN"
        likelihood[key].append(hit.weighted_logit)
        boosted = boosted or hit.boosted

    avg: dict[str, float] = {key: sum(scores) / len(scores) for key, scores in likelihood.items()}
    if not avg:
        return {}

    if boosted:
        _spread_non_match_floor(avg, identities=identities, cfg=cfg)

    return avg


def aggregate_max_over_views(
    capped: Sequence[ScoredHit],
    *,
    identities: Collection[str],
    cfg: GalleryScoringConfig,
) -> dict[str, float]:
    """Max-over-views aggregation with the UNKNOWN-complement guard (multiview semantics)."""
    likelihood: dict[str, list[float]] = defaultdict(list)
    boosted = False
    for hit in capped:
        key = hit.entry.identity_id if hit.entry.identity_id else "UNKNOWN"
        likelihood[key].append(hit.weighted_logit)
        boosted = boosted or hit.boosted

    avg: dict[str, float] = {key: max(scores) for key, scores in likelihood.items()}
    if not avg:
        return {}

    if boosted:
        _spread_non_match_floor(avg, identities=identities, cfg=cfg)

    # Single-identity normalization guard. Without residual UNKNOWN mass a
    # weak best match still normalizes to the only/nearest enrolled identity,
    # so a stranger whose body does not match anyone commits as that identity
    # (the clinical identity-leak this resolver must never produce). Hold the
    # complement of the best identity match strength on UNKNOWN: weak matches
    # resolve to UNKNOWN while strong matches (logit near 1) still commit.
    identity_logits = [v for k, v in avg.items() if k != "UNKNOWN"]
    if identity_logits:
        avg["UNKNOWN"] = max(avg.get("UNKNOWN", 0.0), 1.0 - max(identity_logits))

    return avg

"""Deterministic rebuild of PH appearance state from accepted observations.

M03 tasks 10-11 expose this as a *service* for the confirmed handoff / split /
merge operations that land in M06. It is intentionally NOT wired into the live
``WorldTracker`` step or the ``PHRepository.merge`` / ``.split`` endpoints in
this milestone — the no-cross-PR-dependency rule keeps M06's correction flow
the owner of when partition/rebuild runs. M06 calls these pure functions to:

- partition a corrected PH's accepted-observation history at the split/handoff
  boundary and rebuild each active side independently, and
- on merge, concatenate the source and target accepted-observation partitions
  and rebuild once — rather than directly averaging two PHs' ``gallery_mean``
  vectors, which would blend conflicting identities.

(Note: today's ``InMemoryPHRepository.merge`` already keeps the target's
appearance and closes the source — it performs no mean averaging — so there is
no averaging to *remove*; M06 replaces the keep-target behaviour with a rebuild.)

Rebuild is order-independent in the only way that matters: a fixed input
multiset of accepted observations always yields the same prototypes and gallery
mean, by replaying them through the same EMA helpers the live tracker uses, in a
deterministic orientation-then-time order. Pure: no I/O, no datetime.now.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

from ...domain import OrientationBin, ViewPrototype
from ..orientation import update_view_prototypes
from .appearance_policy import evaluate_appearance_update
from .config import WorldTrackerConfig
from .helpers import update_gallery_mean


@dataclass(frozen=True)
class AcceptedObservation:
    """The minimal appearance-bearing slice of a persisted observation.

    M06 builds these from the PH's observation history. ``captured_at`` gives a
    stable replay order; it is not otherwise interpreted here.
    """

    embedding: list[float]
    orientation: OrientationBin
    orientation_confidence: float
    quality: float
    captured_at: datetime


@dataclass(frozen=True)
class RebuiltAppearance:
    """Rebuilt PH-local appearance state."""

    gallery_mean: list[float] | None
    view_prototypes: tuple[ViewPrototype, ...]
    mean_quality: float
    accepted_count: int


def _replay_order(observations: Iterable[AcceptedObservation]) -> list[AcceptedObservation]:
    """Deterministic, order-independent replay order: by (orientation, time)."""
    return sorted(observations, key=lambda o: (int(o.orientation), o.captured_at))


def rebuild_appearance(
    observations: Sequence[AcceptedObservation],
    cfg: WorldTrackerConfig,
) -> RebuiltAppearance:
    """Rebuild gallery mean, view prototypes, and mean-quality from scratch.

    Each observation passes the same :func:`evaluate_appearance_update` policy
    the live tracker applies, so a rebuild can only ever contain embeddings the
    tracker itself would have accepted. The result is identical for any input
    ordering of the same observation multiset.
    """
    gallery_mean: list[float] | None = None
    prototypes: tuple[ViewPrototype, ...] = ()
    mean_quality = 0.0
    accepted = 0

    for obs in _replay_order(observations):
        decision = evaluate_appearance_update(
            embedding=obs.embedding,
            orientation=obs.orientation,
            orientation_confidence=obs.orientation_confidence,
            quality=obs.quality,
            existing_prototypes=prototypes,
            cfg=cfg,
        )
        if not decision.accept:
            continue
        gallery_mean = update_gallery_mean(gallery_mean, obs.embedding, accepted)
        prototypes = update_view_prototypes(
            prototypes, obs.orientation, obs.embedding, obs.orientation_confidence
        )
        mean_quality = obs.quality if accepted == 0 else 0.1 * obs.quality + 0.9 * mean_quality
        accepted += 1

    return RebuiltAppearance(
        gallery_mean=gallery_mean,
        view_prototypes=prototypes,
        mean_quality=mean_quality,
        accepted_count=accepted,
    )


def partition_and_rebuild(
    partitions: Sequence[Sequence[AcceptedObservation]],
    cfg: WorldTrackerConfig,
) -> list[RebuiltAppearance]:
    """Rebuild each side of a split/handoff independently.

    ``partitions`` is the corrected PH's accepted-observation history already
    split into the active sides (M06 owns the partitioning predicate). Each side
    is rebuilt from its own observations only — no appearance crosses the
    boundary. For PH merge, pass a single partition that concatenates the source
    and target histories.
    """
    return [rebuild_appearance(part, cfg) for part in partitions]

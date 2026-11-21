"""Closed-world prior-shaping channel for the Bayesian identity resolver.

Shapes priors using two physical constraints:
1. ``home_i``: down-weights members reported ``away`` by Cognitive Companion with
   high confidence unless a non-parked bound wearable tag was heard recently.
2. ``elim_ij``: down-weights an identity already held by another simultaneously
   observed, spatially distinct open PH with fresh non-room-wearable evidence.

This channel shapes priors only. Its evidence grade (``CLOSED_WORLD``) sets
``counts_as_commit_evidence = False`` and ``refreshes_evidence_clock = False``,
so it can never commit an identity or extend a maintenance clock on its own.
An explicit guest slot (``UNKNOWN``) keeps at least ``guest_floor`` (or
``guest_floor_visitors`` when visitors are present) of the normalized channel
mass so visitors are never forced onto an enrolled resident.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from cts_contracts.household import PresenceStatus

from app.domain import (
    CLOSED_WORLD_MODES,
    ClosedWorldMode,
    Identity,
    IdentityResolvableEntity,
    OpenPHOccupancy,
    OverlapGroup,
    PosteriorDist,
)
from app.observability import metrics as _metrics
from app.observability.metrics import ClosedWorldInactiveReason
from app.tracking.identity.channels import ChannelLikelihood, ResolverInputs
from app.tracking.identity.evidence_grades import CLOSED_WORLD


@dataclass(frozen=True)
class ClosedWorldConfig:
    """Configuration governing closed-world prior shaping.

    Dimensional fields declare their units (``_s`` seconds, ``_m`` metres in the
    shared household floor-plan frame).
    """

    # Rollout switch: "off" skips evaluation; "shadow" computes counterfactual
    # outcomes without changing live decisions; "enabled" joins the live posterior.
    mode: ClosedWorldMode = "off"

    # Factor applied to an enrolled member reported "away" with confidence >=
    # away_min_confidence and no fresh non-parked wearable fix. Bounded away
    # from zero so a left-behind phone slows commits rather than blocking them.
    away_factor: float = 0.3

    # Minimum PresenceService confidence required before treating "away" as
    # informative. Low-confidence or unknown presence remains neutral (1.0).
    away_min_confidence: float = 0.8

    # Lookback horizon (seconds) for a fresh non-parked bound wearable fix that
    # overrides a stale or phone-only CC "away" status back to neutral (1.0).
    wearable_home_window_s: float = 600.0

    # Factor applied to an identity held by another simultaneously observed,
    # spatially distinct open PH with fresh independent evidence.
    occupied_factor: float = 0.2

    # Maximum age (seconds, in capture time) of the other PH's independent
    # evidence for it to count as a fresh incumbent in joint elimination.
    occupancy_fresh_s: float = 60.0

    # Maximum capture-time difference (seconds) between two PHs' last_seen_at
    # timestamps for them to count as simultaneously observed in the current round.
    simultaneous_obs_max_skew_s: float = 1.0

    # Minimum floor-plane Euclidean distance (metres) separating two calibrated
    # open PHs without a CoPresenceLink to prove they are distinct people.
    min_elimination_distance_m: float = 2.0

    # Minimum normalized share reserved for UNKNOWN (the explicit guest slot)
    # under normal household conditions.
    guest_floor: float = 0.10

    # Elevated minimum UNKNOWN share when CC reports visitors present (active
    # named guest or a surfaced visitor cluster sighted within the last hour).
    guest_floor_visitors: float = 0.30

    # Age (seconds) after which a HouseholdPresence snapshot's per-member
    # statuses are treated as unknown so stale presence never penalizes a member.
    presence_stale_after_s: float = 180.0

    # Hard lower and upper clamps on per-identity factors so closed-world prior
    # shaping can never zero out a candidate or override direct biometric proof.
    factor_floor: float = 0.05
    factor_ceiling: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in CLOSED_WORLD_MODES:
            raise ValueError(
                f"closed_world.mode must be one of {CLOSED_WORLD_MODES}, got {self.mode!r}"
            )
        if not 0.0 < self.factor_floor <= self.factor_ceiling <= 1.0:
            raise ValueError("closed_world factor bounds must satisfy 0 < floor <= ceiling <= 1")
        if not self.factor_floor <= self.away_factor <= self.factor_ceiling:
            raise ValueError("closed_world.away_factor must lie within [factor_floor, ceiling]")
        if not self.factor_floor <= self.occupied_factor <= self.factor_ceiling:
            raise ValueError("closed_world.occupied_factor must lie within [factor_floor, ceiling]")
        if not 0.0 <= self.away_min_confidence <= 1.0:
            raise ValueError("closed_world.away_min_confidence must be in [0, 1]")
        if self.wearable_home_window_s < 0.0:
            raise ValueError("closed_world.wearable_home_window_s must be >= 0")
        if self.occupancy_fresh_s <= 0.0:
            raise ValueError("closed_world.occupancy_fresh_s must be positive")
        if self.simultaneous_obs_max_skew_s < 0.0:
            raise ValueError("closed_world.simultaneous_obs_max_skew_s must be >= 0")
        if self.min_elimination_distance_m <= 0.0:
            raise ValueError("closed_world.min_elimination_distance_m must be positive")
        if not 0.0 < self.guest_floor < 1.0:
            raise ValueError("closed_world.guest_floor must be in (0, 1)")
        if not self.guest_floor <= self.guest_floor_visitors < 1.0:
            raise ValueError("closed_world.guest_floor_visitors must be in [guest_floor, 1)")
        if self.presence_stale_after_s <= 0.0:
            raise ValueError("closed_world.presence_stale_after_s must be positive")


@dataclass(frozen=True)
class ClosedWorldEvaluation:
    """Result of evaluating :class:`ClosedWorldChannel` for one entity."""

    likelihood: ChannelLikelihood
    active: bool
    inactive_reason: ClosedWorldInactiveReason | None = None
    presence_stale: bool = False


def canonical_ph_pair(ph_id_a: str, ph_id_b: str) -> tuple[str, str]:
    """Return a canonical sorted ``(min_id, max_id)`` key for an unordered PH pair."""
    return (ph_id_a, ph_id_b) if ph_id_a <= ph_id_b else (ph_id_b, ph_id_a)


def rooms_or_cameras_overlap(
    occ_a: OpenPHOccupancy,
    occ_b: OpenPHOccupancy,
    overlap_groups: Sequence[OverlapGroup],
    camera_room_map: Mapping[str, str],
) -> bool:
    """Return True if *occ_a* and *occ_b* could view the same open space.

    Two PHs are treated as overlapping if any registered ``OverlapGroup`` links
    a camera in ``occ_a``'s room to a camera in ``occ_b``'s room, or links any
    camera observed by ``occ_a`` to any camera observed by ``occ_b``.
    """
    if not overlap_groups:
        return False

    cams_a: set[str] = set(occ_a.active_cameras)
    if occ_a.last_seen_camera:
        cams_a.add(occ_a.last_seen_camera)
    cams_b: set[str] = set(occ_b.active_cameras)
    if occ_b.last_seen_camera:
        cams_b.add(occ_b.last_seen_camera)

    room_a = occ_a.room_id
    room_b = occ_b.room_id

    for group in overlap_groups:
        group_cams = set(group.camera_ids)
        if not group_cams:
            continue
        if (cams_a & group_cams) and (cams_b & group_cams):
            return True
        if room_a and room_b:
            group_rooms = {camera_room_map[cam] for cam in group_cams if cam in camera_room_map}
            if room_a in group_rooms and room_b in group_rooms:
                return True
    return False


def are_phs_spatially_distinct(
    occ_a: OpenPHOccupancy,
    occ_b: OpenPHOccupancy,
    *,
    min_elimination_distance_m: float,
    copresence_pairs: frozenset[tuple[str, str]],
    overlap_groups: Sequence[OverlapGroup],
    camera_room_map: Mapping[str, str],
) -> bool:
    """Return True only when *occ_a* and *occ_b* are proven to be distinct people.

    Branch 1 (calibrated floor distance): both PHs have calibrated floor
    coordinates, no active ``CoPresenceLink`` connects them, and their
    floor-plane Euclidean distance is at least *min_elimination_distance_m*.

    Branch 2 (non-overlapping rooms): both PHs have known, different rooms and
    no registered ``OverlapGroup`` links those rooms or their cameras.
    """
    if (
        occ_a.floor_calibrated
        and occ_b.floor_calibrated
        and canonical_ph_pair(occ_a.ph_id, occ_b.ph_id) not in copresence_pairs
    ):
        dist_m = math.hypot(
            occ_a.floor_x_m - occ_b.floor_x_m,
            occ_a.floor_y_m - occ_b.floor_y_m,
        )
        if math.isfinite(dist_m) and dist_m >= min_elimination_distance_m:
            return True

    return bool(
        occ_a.room_id
        and occ_b.room_id
        and occ_a.room_id != occ_b.room_id
        and not rooms_or_cameras_overlap(occ_a, occ_b, overlap_groups, camera_room_map)
    )


def _empty_likelihood(channel_name: str) -> ChannelLikelihood:
    return ChannelLikelihood(
        channel=channel_name,
        distribution=PosteriorDist({}),
        support_multiplier=1.0,
        evidence_at=None,
        clock_evidence_ids=frozenset(),
        counts_as_commit_evidence=CLOSED_WORLD.counts_as_commit_evidence,
    )


class ClosedWorldChannel:
    """Prior-shaping channel combining household presence and joint elimination."""

    name: str = "closed_world"

    def __init__(
        self,
        *,
        config: ClosedWorldConfig,
        identities_getter: Callable[[], Mapping[str, Identity]],
    ) -> None:
        self._config = config
        self._identities_getter = identities_getter

    def evaluate_closed_world(
        self,
        entity: IdentityResolvableEntity,
        inputs: ResolverInputs,
    ) -> ClosedWorldEvaluation:
        """Compute the closed-world likelihood and gate status for *entity*."""
        presence = inputs.household_presence
        if presence is None:
            _metrics.metrics.cts_closed_world_inactive_total.labels(
                reason=str(ClosedWorldInactiveReason.NO_PRESENCE)
            ).inc()
            return ClosedWorldEvaluation(
                likelihood=_empty_likelihood(self.name),
                active=False,
                inactive_reason=ClosedWorldInactiveReason.NO_PRESENCE,
            )

        if not presence.roster_complete:
            _metrics.metrics.cts_closed_world_inactive_total.labels(
                reason=str(ClosedWorldInactiveReason.ROSTER_INCOMPLETE)
            ).inc()
            return ClosedWorldEvaluation(
                likelihood=_empty_likelihood(self.name),
                active=False,
                inactive_reason=ClosedWorldInactiveReason.ROSTER_INCOMPLETE,
            )

        identities = self._identities_getter()
        if not identities:
            _metrics.metrics.cts_closed_world_inactive_total.labels(
                reason=str(ClosedWorldInactiveReason.NO_IDENTITIES)
            ).inc()
            return ClosedWorldEvaluation(
                likelihood=_empty_likelihood(self.name),
                active=False,
                inactive_reason=ClosedWorldInactiveReason.NO_IDENTITIES,
            )

        cfg = self._config
        age_s = (inputs.captured_at - presence.published_at).total_seconds()
        presence_stale = age_s > cfg.presence_stale_after_s
        members_by_id = {} if presence_stale else presence.by_member_id()

        self_occ = inputs.open_ph_occupancies.get(entity.entity_id)
        eliminated_identities = self._eliminated_identities(
            entity_id=entity.entity_id,
            self_occ=self_occ,
            inputs=inputs,
        )

        raw: dict[str, float] = {}
        for ident_id in sorted(identities.keys()):
            home_factor = 1.0
            if ident_id not in inputs.wearable_present_identities:
                mp = members_by_id.get(ident_id)
                if (
                    mp is not None
                    and mp.status == PresenceStatus.AWAY
                    and mp.confidence >= cfg.away_min_confidence
                ):
                    home_factor = cfg.away_factor

            elim_factor = cfg.occupied_factor if ident_id in eliminated_identities else 1.0

            home_clamped = max(cfg.factor_floor, min(cfg.factor_ceiling, home_factor))
            elim_clamped = max(cfg.factor_floor, min(cfg.factor_ceiling, elim_factor))
            raw[ident_id] = max(
                cfg.factor_floor, min(cfg.factor_ceiling, home_clamped * elim_clamped)
            )

        # Base UNKNOWN factor is 1.0 (unpenalized); post-normalization share is
        # floored at effective_guest_floor so visitors retain explicit mass.
        raw["UNKNOWN"] = max(1.0, cfg.guest_floor)
        total = sum(raw.values())
        normalized = {k: v / total for k, v in raw.items()}

        effective_guest_floor = (
            cfg.guest_floor_visitors if presence.visitors_present else cfg.guest_floor
        )
        p_unknown = normalized["UNKNOWN"]
        if p_unknown < effective_guest_floor:
            resident_mass = 1.0 - p_unknown
            if resident_mass > 0.0:
                scale = (1.0 - effective_guest_floor) / resident_mass
                for k in list(normalized.keys()):
                    if k != "UNKNOWN":
                        normalized[k] *= scale
            normalized["UNKNOWN"] = effective_guest_floor

        likelihood = ChannelLikelihood(
            channel=self.name,
            distribution=PosteriorDist(normalized),
            support_multiplier=1.0,
            evidence_at=None,
            clock_evidence_ids=frozenset(),
            counts_as_commit_evidence=CLOSED_WORLD.counts_as_commit_evidence,
        )
        return ClosedWorldEvaluation(
            likelihood=likelihood,
            active=True,
            inactive_reason=None,
            presence_stale=presence_stale,
        )

    def _eliminated_identities(
        self,
        *,
        entity_id: str,
        self_occ: OpenPHOccupancy | None,
        inputs: ResolverInputs,
    ) -> set[str]:
        """Return identities held by another simultaneously observed, spatially distinct PH."""
        if self_occ is None or self_occ.coasting:
            return set()

        cfg = self._config
        eliminated: set[str] = set()
        for other_occ in inputs.open_ph_occupancies.values():
            if other_occ.ph_id == entity_id:
                continue
            if not other_occ.identity_id or not other_occ.elim_eligible:
                continue
            if other_occ.coasting:
                continue
            if other_occ.last_independent_evidence_at is None:
                continue
            evidence_age_s = (
                inputs.captured_at - other_occ.last_independent_evidence_at
            ).total_seconds()
            if evidence_age_s > cfg.occupancy_fresh_s:
                continue
            obs_skew_s = abs((self_occ.last_seen_at - other_occ.last_seen_at).total_seconds())
            if obs_skew_s > cfg.simultaneous_obs_max_skew_s:
                continue
            if are_phs_spatially_distinct(
                self_occ,
                other_occ,
                min_elimination_distance_m=cfg.min_elimination_distance_m,
                copresence_pairs=inputs.copresence_pairs,
                overlap_groups=inputs.overlap_groups,
                camera_room_map=inputs.camera_room_map,
            ):
                eliminated.add(other_occ.identity_id)
        return eliminated

    async def evaluate(
        self,
        entity: IdentityResolvableEntity,
        inputs: ResolverInputs,
    ) -> ChannelLikelihood:
        """Implement :class:`EvidenceChannel`."""
        return self.evaluate_closed_world(entity, inputs).likelihood

"""Evidence-channel protocol and built-in identity channels (face, gallery, height).

Every evidence source consumed by :class:`IdentityResolver` emits a
:class:`ChannelLikelihood`. Adding a new kind of identity evidence (such as a
wearable or appearance-classifier channel) means adding a class implementing
:class:`EvidenceChannel` plus a config block, without widening the combiner
signature or adding inline origin checks.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Literal, Protocol

from structlog import get_logger

from app.domain import (
    FaceAnchor,
    HouseholdPresenceSnapshot,
    Identity,
    IdentityHeightProfile,
    IdentityResolvableEntity,
    OpenPHOccupancy,
    OverlapGroup,
    PosteriorDist,
    WearableEvidence,
)
from app.tracking.identity.commit_policy import (
    _evidence_face_anchors,
    _reid_evidence_identities,
    collect_evidence_identity_ids,
    identity_evidence_time,
)
from app.tracking.identity.evidence_grades import EvidenceGrade, grade_of

if TYPE_CHECKING:
    from app.inference.evidence import FaceEvidence

logger = get_logger(__name__)

FaceQualityMode = Literal["off", "shadow", "enabled"]


class FaceChannelSettings(Protocol):
    """Read-only configuration slice required by :class:`FaceChannel`."""

    @property
    def face_weight_multiplier(self) -> float: ...

    @property
    def candidate_face_weight_multiplier(self) -> float: ...

    @property
    def face_present_unknown_unknown_mass(self) -> float: ...

    @property
    def propagated_face_weight_multiplier(self) -> float: ...

    @property
    def cc_assertion_likelihood_scale(self) -> float: ...

    @property
    def uncalibrated_face_max_likelihood(self) -> float: ...

    @property
    def face_quality_mode(self) -> FaceQualityMode: ...

    @property
    def face_quality_missing_default(self) -> float: ...

    @property
    def face_evidence_policy(self) -> str: ...

    @property
    def frontality_full_yaw_deg(self) -> float: ...

    @property
    def frontality_zero_yaw_deg(self) -> float: ...

    @property
    def frontality_min_factor(self) -> float: ...


@dataclass(frozen=True)
class ChannelLikelihood:
    """Likelihood distribution produced by one :class:`EvidenceChannel`.

    Note on ``support_multiplier``: in :func:`combine_channels`, multiplying every
    entry present in ``distribution`` by ``support_multiplier`` only changes the
    normalized posterior when some identities in the combined universe are absent
    from ``distribution`` (receiving the unmultiplied Laplace fill
    ``1 / (len(distribution) + 1)``). When ``distribution`` already populates
    every enrolled identity plus ``UNKNOWN``, scalar multiplication cancels on
    normalization. Any new channel (such as wearable or household-classifier
    channels) that populates all identities plus ``UNKNOWN`` must apply its
    weight as a log-likelihood power (``p ** weight_multiplier``) or pre-softmax
    logit scale before constructing ``PosteriorDist`` and set
    ``support_multiplier = 1.0``.

    ``evidence_at`` reports the newest capture time of the evidence this channel
    used (when it is allowed to refresh the independent-evidence clock). Channels
    that can report per-identity evidence timestamps (such as multi-anchor face
    evidence) populate ``evidence_at_by_identity`` alongside ``clock_evidence_ids``.
    """

    channel: str
    distribution: PosteriorDist
    support_multiplier: float = 1.0
    evidence_at: datetime | None = None
    clock_evidence_ids: frozenset[str] = frozenset()
    evidence_at_by_identity: Mapping[str, datetime] = field(default_factory=dict)
    counts_as_commit_evidence: bool = True


@dataclass(frozen=True)
class VoteWindow:
    """Capture-time gallery vote window ``(as_of - max_age_s, as_of]``.

    Built once per ``resolve()`` call from the round's ``captured_at``. Using one
    window for repository bounds and recency decay keeps replay deterministic and
    prevents rows minted later in a batch from voting for an earlier context.
    """

    as_of: datetime
    max_age_s: float | None = None

    @property
    def seen_after(self) -> datetime | None:
        if self.max_age_s is None:
            return None
        return self.as_of - timedelta(seconds=self.max_age_s)

    @property
    def seen_before(self) -> datetime:
        return self.as_of


@dataclass(frozen=True)
class ResolverInputs:
    """Per-round evidence shared by all channels. Epoch: one ``resolve()`` call.

    ``captured_at`` is the capture time of the round (never wall-clock ingest time).
    ``face_anchors`` and ``face_evidence`` carry the post-propagation inputs used
    for likelihood evaluation and commit eligibility; ``raw_face_anchors`` and
    ``raw_face_evidence`` carry the pre-propagation inputs used for the
    independent-evidence clock.
    """

    captured_at: datetime
    face_anchors: tuple[FaceAnchor, ...]
    face_evidence: tuple[FaceEvidence, ...]
    ph_heights_m: Mapping[str, float]
    ph_qualities: Mapping[str, float]
    ph_height_sigmas_m: Mapping[str, float] = field(default_factory=dict)
    raw_face_anchors: tuple[FaceAnchor, ...] = ()
    raw_face_evidence: tuple[FaceEvidence, ...] = ()
    wearable_evidence: Mapping[str, WearableEvidence] = field(default_factory=dict)
    vote_window: VoteWindow | None = None
    open_ph_occupancies: Mapping[str, OpenPHOccupancy] = field(default_factory=dict)
    copresence_pairs: frozenset[tuple[str, str]] = frozenset()
    overlap_groups: tuple[OverlapGroup, ...] = ()
    camera_room_map: Mapping[str, str] = field(default_factory=dict)
    household_presence: HouseholdPresenceSnapshot | None = None
    wearable_present_identities: frozenset[str] = frozenset()


class EvidenceChannel(Protocol):
    """Protocol implemented by every identity evidence channel."""

    name: str

    async def evaluate(
        self,
        entity: IdentityResolvableEntity,
        inputs: ResolverInputs,
    ) -> ChannelLikelihood: ...


@dataclass(frozen=True)
class FaceChannelResult:
    """Detailed face-channel evaluation carrying likelihood plus contradiction score."""

    likelihood: ChannelLikelihood
    best_face_similarity: float | None


class FaceChannel:
    """Face-anchor evidence channel (native frame, template, and external anchors)."""

    name: str = "face"

    def __init__(
        self,
        *,
        config: FaceChannelSettings,
        identities_getter: Callable[[], Mapping[str, Identity]],
    ) -> None:
        self._config = config
        self._identities_getter = identities_getter

    def likelihood_scale(self, grade: EvidenceGrade) -> float:
        """Look up the configured likelihood scale for *grade* (1.0 when none is configured)."""
        if grade.likelihood_scale_key is None:
            return 1.0
        return float(getattr(self._config, grade.likelihood_scale_key))

    def p_face(self, confidence: float, quality: float) -> float:
        """Face likelihood weight for a recognized anchor from its match score and quality.

        An uncalibrated tanh heuristic, not a fitted probability: 0.7 * score + 0.3 * quality is
        squashed around 0.5 and capped at 0.99. At (0.5, 0.5) -> 0.5; at (0.95, 1.0) -> ~0.96.
        ``quality`` is :meth:`effective_face_quality`'s value.
        """
        combined = 0.7 * confidence + 0.3 * quality
        return min(0.99, 0.5 + 0.5 * math.tanh(4 * (combined - 0.5)))

    def frontality_factor(self, yaw_deg: float) -> float:
        """Linear frontality ramp based on absolute yaw in degrees."""
        abs_yaw = abs(yaw_deg)
        if abs_yaw <= self._config.frontality_full_yaw_deg:
            return 1.0
        if abs_yaw >= self._config.frontality_zero_yaw_deg:
            return self._config.frontality_min_factor
        frac = (abs_yaw - self._config.frontality_full_yaw_deg) / (
            self._config.frontality_zero_yaw_deg - self._config.frontality_full_yaw_deg
        )
        return 1.0 - frac * (1.0 - self._config.frontality_min_factor)

    def effective_face_quality(self, anchor: FaceAnchor, mode: FaceQualityMode) -> float:
        """The quality a recognized face anchor is weighed with; the only face-quality switch.

        ``anchor.quality`` is the evidence-path weight (1.0 for a native anchor, attenuated on
        propagation, the assertion's own quality for an external anchor). With the mode enabled
        it is multiplied by the service-measured face quality. Grades that declare
        ``applies_missing_face_quality_default`` fall back to ``face_quality_missing_default``
        when the service sent no score; external grades are not discounted again.
        """
        if mode == "off":
            return anchor.quality
        if anchor.face_quality is not None:
            return anchor.quality * anchor.face_quality
        if grade_of(anchor).applies_missing_face_quality_default:
            return anchor.quality * self._config.face_quality_missing_default
        return anchor.quality

    def recognized_face_likelihood(
        self,
        anchor: FaceAnchor,
        *,
        source: str,
        face_quality_mode: FaceQualityMode,
        calibrated: bool,
    ) -> float:
        """Likelihood weight of one recognized face anchor; the only place either weighting lives.

        Legacy weighting uses the uncalibrated ``p_face`` heuristic of match score and effective
        quality. Calibrated weighting uses ``calibrated_confidence`` directly (quality is already
        one of its fitted features), capping uncalibrated faces at
        ``uncalibrated_face_max_likelihood``. Frontality and grade/propagation multipliers apply in
        both.
        """
        weight_mult = (
            self._config.propagated_face_weight_multiplier if source == "propagated" else 1.0
        )
        weight_mult *= self.likelihood_scale(grade_of(anchor))
        frontality = self.frontality_factor(anchor.yaw_deg)
        if calibrated and anchor.calibrated_confidence is not None:
            base = anchor.calibrated_confidence
        else:
            base = self.p_face(
                anchor.match_score, self.effective_face_quality(anchor, face_quality_mode)
            )
            if calibrated:
                base = min(base, self._config.uncalibrated_face_max_likelihood)
        return base * weight_mult * frontality

    def collect_clock_evidence(
        self,
        entity: IdentityResolvableEntity,
        *,
        face_anchors: Sequence[FaceAnchor],
        face_evidence: Sequence[FaceEvidence],
        round_time: datetime,
        calibrated_faces_only: bool,
    ) -> tuple[frozenset[str], dict[str, datetime]]:
        """Collect pre-propagation face identities and newest capture times for the clock."""
        entity_obs_ids = frozenset(entity.observation_ids)
        times_by_id: dict[str, datetime] = {}
        for fa in _evidence_face_anchors(
            entity_obs_ids=entity_obs_ids,
            entity_id=entity.entity_id,
            face_anchors=face_anchors,
            face_evidence=face_evidence,
            recognized_only=True,
            calibrated_faces_only=calibrated_faces_only,
        ):
            ts = fa.captured_at if grade_of(fa).uses_own_capture_time else round_time
            prev = times_by_id.get(fa.person_id)
            if prev is None or ts > prev:
                times_by_id[fa.person_id] = ts
        return frozenset(times_by_id.keys()), times_by_id

    def evaluate_anchors(
        self,
        entity: IdentityResolvableEntity,
        face_anchors: Sequence[FaceAnchor],
        face_evidence: Sequence[FaceEvidence] | None = None,
        *,
        face_quality_mode: FaceQualityMode = "off",
        calibrated: bool = False,
        raw_face_anchors: Sequence[FaceAnchor] | None = None,
        raw_face_evidence: Sequence[FaceEvidence] | None = None,
        round_time: datetime | None = None,
    ) -> FaceChannelResult:
        """Build the face channel likelihood and best contradiction score for *entity*."""
        entity_obs_ids = set(entity.observation_ids)
        ev_by_detection: dict[str, FaceEvidence] = {}
        if face_evidence:
            for fe in face_evidence:
                if fe.detection_id:
                    ev_by_detection[fe.detection_id] = fe

        relevant_anchors = [
            fa
            for fa in face_anchors
            if fa.tracklet_id in entity_obs_ids
            or fa.tracklet_id == entity.entity_id
            or fa.detection_id in entity_obs_ids
        ]

        clock_ids: frozenset[str] = frozenset()
        clock_times: dict[str, datetime] = {}
        if round_time is not None and raw_face_anchors is not None:
            clock_ids, clock_times = self.collect_clock_evidence(
                entity,
                face_anchors=raw_face_anchors,
                face_evidence=raw_face_evidence or (),
                round_time=round_time,
                calibrated_faces_only=calibrated,
            )
        newest_evidence_at = max(clock_times.values(), default=None)

        if not relevant_anchors:
            logger.debug(
                "face_no_match_for_entity",
                entity_id=entity.entity_id,
                obs_count=len(entity_obs_ids),
                face_anchor_count=len(face_anchors),
            )
            return FaceChannelResult(
                likelihood=ChannelLikelihood(
                    channel=self.name,
                    distribution=PosteriorDist({}),
                    support_multiplier=self._config.face_weight_multiplier,
                    evidence_at=newest_evidence_at,
                    clock_evidence_ids=clock_ids,
                    evidence_at_by_identity=clock_times,
                ),
                best_face_similarity=None,
            )

        recognized_anchors = [fa for fa in relevant_anchors if fa.recognition_state == "recognized"]
        candidate_anchors = [fa for fa in relevant_anchors if fa.recognition_state == "candidate"]
        unrecognized_anchors = [
            fa for fa in relevant_anchors if fa.recognition_state == "unrecognized"
        ]

        likelihood: dict[str, float] = {}
        best_recognized_conf: float | None = None

        def _source(fa: FaceAnchor) -> str:
            ev = ev_by_detection.get(fa.detection_id)
            return ev.source if ev and ev.source else "direct"

        def _likelihood(fa: FaceAnchor) -> float:
            return self.recognized_face_likelihood(
                fa,
                source=_source(fa),
                face_quality_mode=face_quality_mode,
                calibrated=calibrated or grade_of(fa).requires_calibration,
            )

        best: FaceAnchor | None = None
        if recognized_anchors:
            if calibrated or any(grade_of(fa).requires_calibration for fa in recognized_anchors):
                best = max(recognized_anchors, key=_likelihood)
            else:
                best = max(
                    recognized_anchors,
                    key=lambda fa: (
                        fa.match_score * self.effective_face_quality(fa, face_quality_mode)
                    ),
                )
            best_recognized_conf = best.match_score if grade_of(best).enters_contradiction else None
            p_face = _likelihood(best)

            if best.person_id:
                likelihood[best.person_id] = p_face

            logger.debug(
                "face_anchor_matched",
                entity_id=entity.entity_id,
                person_id=best.person_id,
                confidence=round(best.match_score, 3),
                source=_source(best),
                likelihood=round(p_face, 4),
                calibrated=calibrated,
                anchor_count=len(relevant_anchors),
            )

        candidate_ratio = self._config.candidate_face_weight_multiplier / max(
            self._config.face_weight_multiplier, 0.01
        )
        for fa in candidate_anchors:
            if not fa.person_id or fa.person_id == "unknown":
                continue
            frontality = self.frontality_factor(fa.yaw_deg)
            candidate_weight = fa.similarity * frontality * candidate_ratio
            existing = likelihood.get(fa.person_id, 0.0)
            likelihood[fa.person_id] = max(existing, candidate_weight)
            logger.debug(
                "face_candidate_matched",
                entity_id=entity.entity_id,
                person_id=fa.person_id,
                similarity=round(fa.similarity, 3),
                frontality_factor=round(frontality, 3),
                candidate_ratio=round(candidate_ratio, 3),
                candidate_weight=round(candidate_weight, 4),
            )

        if unrecognized_anchors:
            unknown_mass = self._config.face_present_unknown_unknown_mass
            likelihood["UNKNOWN"] = likelihood.get("UNKNOWN", 0.0) + unknown_mass
            logger.debug(
                "face_unrecognized_marker",
                entity_id=entity.entity_id,
                unrecognized_count=len(unrecognized_anchors),
                unknown_mass_added=unknown_mass,
            )

        spread_remainder = (
            best is not None
            and (
                not calibrated
                or any(fa.calibrated_confidence is not None for fa in recognized_anchors)
            )
            and not (grade_of(best).requires_calibration and best.calibrated_confidence is None)
        )
        if likelihood:
            total_face = sum(likelihood.values())
            remainder = max(0.0, 1.0 - total_face)
            if remainder > 0:
                if spread_remainder:
                    candidates = list(self._identities_getter().keys())
                    if candidates:
                        per_id = remainder / (len(candidates) + 1)
                        for cid in candidates:
                            if cid not in likelihood:
                                likelihood[cid] = per_id
                        likelihood["UNKNOWN"] = likelihood.get("UNKNOWN", 0.0) + per_id
                    else:
                        likelihood["UNKNOWN"] = likelihood.get("UNKNOWN", 0.0) + remainder
                else:
                    likelihood["UNKNOWN"] = likelihood.get("UNKNOWN", 0.0) + remainder

        return FaceChannelResult(
            likelihood=ChannelLikelihood(
                channel=self.name,
                distribution=PosteriorDist(likelihood),
                support_multiplier=self._config.face_weight_multiplier,
                evidence_at=newest_evidence_at,
                clock_evidence_ids=clock_ids,
                evidence_at_by_identity=clock_times,
            ),
            best_face_similarity=best_recognized_conf,
        )

    def evaluate_detailed(
        self,
        entity: IdentityResolvableEntity,
        inputs: ResolverInputs,
        *,
        face_quality_mode: FaceQualityMode | None = None,
        calibrated: bool | None = None,
    ) -> FaceChannelResult:
        """Evaluate the face channel from :class:`ResolverInputs`."""
        resolved_quality_mode: FaceQualityMode = (
            face_quality_mode
            if face_quality_mode is not None
            else ("enabled" if self._config.face_quality_mode == "enabled" else "off")
        )
        resolved_calibrated = (
            calibrated
            if calibrated is not None
            else (self._config.face_evidence_policy == "calibrated")
        )
        return self.evaluate_anchors(
            entity,
            inputs.face_anchors,
            inputs.face_evidence,
            face_quality_mode=resolved_quality_mode,
            calibrated=resolved_calibrated,
            raw_face_anchors=inputs.raw_face_anchors,
            raw_face_evidence=inputs.raw_face_evidence,
            round_time=inputs.captured_at,
        )

    async def evaluate(
        self,
        entity: IdentityResolvableEntity,
        inputs: ResolverInputs,
    ) -> ChannelLikelihood:
        """Implement :class:`EvidenceChannel` using live policy settings."""
        return self.evaluate_detailed(entity, inputs).likelihood

    def collect_commit_evidence_ids(
        self,
        entity: IdentityResolvableEntity,
        *,
        face_anchors: Sequence[FaceAnchor],
        face_evidence: Sequence[FaceEvidence],
        reid_likelihood: PosteriorDist,
        calibrated_faces_only: bool,
    ) -> frozenset[str]:
        """Return identities with commit-eligible face or ReID evidence."""
        return collect_evidence_identity_ids(
            entity_obs_ids=frozenset(entity.observation_ids),
            entity_id=entity.entity_id,
            face_anchors=face_anchors,
            face_evidence=face_evidence,
            reid_likelihood=reid_likelihood,
            recognized_only=False,
            calibrated_faces_only=calibrated_faces_only,
        )

    def evidence_time_for_identity(
        self,
        *,
        identity_id: str,
        round_time: datetime,
        entity: IdentityResolvableEntity,
        face_anchors: Sequence[FaceAnchor],
        face_evidence: Sequence[FaceEvidence],
        reid_likelihood: PosteriorDist,
        calibrated_faces_only: bool,
    ) -> datetime | None:
        """Return the newest clock-qualifying evidence capture time for *identity_id*."""
        return identity_evidence_time(
            identity_id=identity_id,
            round_time=round_time,
            entity_obs_ids=frozenset(entity.observation_ids),
            entity_id=entity.entity_id,
            face_anchors=face_anchors,
            face_evidence=face_evidence,
            reid_likelihood=reid_likelihood,
            recognized_only=True,
            calibrated_faces_only=calibrated_faces_only,
        )


class GalleryChannel:
    """Verified ReID gallery evidence channel."""

    name: str = "gallery"

    def __init__(
        self,
        *,
        query_fn: Callable[[IdentityResolvableEntity, VoteWindow], Awaitable[PosteriorDist]],
        support_multiplier: float = 1.0,
    ) -> None:
        self._query_fn = query_fn
        self._support_multiplier = support_multiplier

    def wrap(self, distribution: PosteriorDist, *, captured_at: datetime) -> ChannelLikelihood:
        """Wrap an already-queried ReID distribution into a :class:`ChannelLikelihood`."""
        reid_ids = _reid_evidence_identities(distribution)
        if reid_ids:
            return ChannelLikelihood(
                channel=self.name,
                distribution=distribution,
                support_multiplier=self._support_multiplier,
                evidence_at=captured_at,
                clock_evidence_ids=reid_ids,
                evidence_at_by_identity=dict.fromkeys(reid_ids, captured_at),
            )
        return ChannelLikelihood(
            channel=self.name,
            distribution=distribution,
            support_multiplier=self._support_multiplier,
            evidence_at=None,
        )

    async def evaluate(
        self,
        entity: IdentityResolvableEntity,
        inputs: ResolverInputs,
    ) -> ChannelLikelihood:
        """Query the ReID gallery for *entity* and return its :class:`ChannelLikelihood`."""
        window = inputs.vote_window or VoteWindow(as_of=inputs.captured_at)
        dist = await self._query_fn(entity, window)
        return self.wrap(dist, captured_at=inputs.captured_at)


class HeightChannel:
    """Anthropometric standing-height channel (corroborating; never refreshes clock).

    Reads PH standing height and uncertainty keyed by ``entity.entity_id`` in metres and
    compares against stored per-identity height profiles in metres. Applies
    ``height_weight_multiplier`` as a log-likelihood power (``exp(-0.5 * w * z^2)``,
    floored at 0.01) with ``support_multiplier = 1.0`` so the weight scales resident-vs-resident
    separation. Includes ``UNKNOWN`` at a neutral baseline (0.5) so mismatched profiles never
    normalize above ``UNKNOWN``, while leaving unprofiled residents absent (Laplace fill).
    """

    name: str = "height"
    _UNKNOWN_BASELINE: float = 0.5
    _MIN_LIKELIHOOD: float = 0.01

    def __init__(
        self,
        *,
        height_profiles_getter: Callable[[], Mapping[str, IdentityHeightProfile]],
        height_weight_multiplier: float,
        default_ph_sigma_m: float = 0.08,
    ) -> None:
        self._height_profiles_getter = height_profiles_getter
        self._height_weight_multiplier = height_weight_multiplier
        self._default_ph_sigma_m = default_ph_sigma_m

    def evaluate_heights(
        self,
        entity: IdentityResolvableEntity,
        ph_heights_m: Mapping[str, float],
        ph_height_sigmas_m: Mapping[str, float] | None = None,
    ) -> ChannelLikelihood:
        """Build height likelihood against per-identity standing height profiles (metres)."""
        ph_height_m = ph_heights_m.get(entity.entity_id)
        if ph_height_m is None or not math.isfinite(ph_height_m):
            return ChannelLikelihood(
                channel=self.name,
                distribution=PosteriorDist({}),
                support_multiplier=1.0,
                evidence_at=None,
            )

        profiles = self._height_profiles_getter()
        if not profiles:
            return ChannelLikelihood(
                channel=self.name,
                distribution=PosteriorDist({}),
                support_multiplier=1.0,
                evidence_at=None,
            )

        raw_sigma_ph = (
            ph_height_sigmas_m.get(entity.entity_id) if ph_height_sigmas_m is not None else None
        )
        sigma_ph = (
            raw_sigma_ph
            if (raw_sigma_ph is not None and math.isfinite(raw_sigma_ph) and raw_sigma_ph > 0.0)
            else self._default_ph_sigma_m
        )

        likelihood: dict[str, float] = {}
        weight = max(0.0, float(self._height_weight_multiplier))
        for identity_id, profile in profiles.items():
            sigma_comb = math.hypot(sigma_ph, profile.height_sigma_m)
            if sigma_comb <= 0.0 or not math.isfinite(sigma_comb):
                continue
            z = (ph_height_m - profile.height_m) / sigma_comb
            p = max(self._MIN_LIKELIHOOD, math.exp(-0.5 * weight * z * z))
            likelihood[identity_id] = p

        if not likelihood:
            return ChannelLikelihood(
                channel=self.name,
                distribution=PosteriorDist({}),
                support_multiplier=1.0,
                evidence_at=None,
            )

        likelihood["UNKNOWN"] = self._UNKNOWN_BASELINE
        return ChannelLikelihood(
            channel=self.name,
            distribution=PosteriorDist(likelihood),
            support_multiplier=1.0,
            evidence_at=None,
        )

    async def evaluate(
        self,
        entity: IdentityResolvableEntity,
        inputs: ResolverInputs,
    ) -> ChannelLikelihood:
        """Implement :class:`EvidenceChannel`."""
        return self.evaluate_heights(entity, inputs.ph_heights_m, inputs.ph_height_sigmas_m)

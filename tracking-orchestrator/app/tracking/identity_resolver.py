"""Identity resolver: Bayesian posterior over identities + retroactive revision.

For each GlobalTrack, the resolver maintains a posterior distribution over
the set {person_ids} U {UNKNOWN}. The posterior is updated from three
evidence sources:

1. **Face anchors** (strong): from the face ID service.
2. **Gallery similarities** (medium): from GalleryRepository k-NN search.
3. **Temporal prior** (weak): from the previous identity assignment.

The commit rule decides whether to assign an identity or keep the track
as UNKNOWN:
- top_probability >= commit_prob AND margin >= commit_margin

When a decision revises a previous identity, the retroactive revision
protocol walks backward through the tracklets in the GlobalTrack and
emits IdentityRevision messages.

This module is pure logic — no I/O. It operates on domain types and
returns decisions/revisions. Persistence and publishing are handled by
the pipeline.
"""

from __future__ import annotations

import math
import random
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict, get_args

import numpy as np
from structlog import get_logger

if TYPE_CHECKING:
    from ..pipeline.gallery_cache import GalleryCache

from ..domain import (
    FACE_EVIDENCE_POLICIES,
    AppearanceFeature,
    FaceAnchor,
    FaceEvidencePolicy,
    GalleryEmbedding,
    GallerySeedSource,
    HouseholdPresenceSnapshot,
    HouseholdPresenceSource,
    Identity,
    IdentityDecision,
    IdentityHeightProfile,
    IdentityResolvableEntity,
    IdentityRevision,
    ImageOnlyEvaluation,
    OpenPHOccupancy,
    OverlapGroup,
    PosteriorDist,
    ResolveOutcome,
    ViewPrototype,
    WearableEvidence,
    WearableEvidenceClass,
)
from ..inference.evidence import FaceEvidence
from ..observability import metrics
from ..observability.metrics import (
    ClosedWorldShadowOutcome,
    HouseholdClassifierShadowOutcome,
    WearableShadowOutcome,
)
from ..storage.base import VOTING_STATES, GalleryRepository
from ..storage.height_profiles import IdentityHeightProfileRepository
from .identity.channels import (
    ChannelLikelihood,
    EvidenceChannel,
    FaceChannel,
    GalleryChannel,
    HeightChannel,
    ResolverInputs,
    VoteWindow,
)
from .identity.closed_world_channel import ClosedWorldChannel, ClosedWorldConfig
from .identity.commit_policy import (
    CommitEvaluation,
    compute_contradiction,
)
from .identity.commit_policy import (
    evaluate_commit as _evaluate_commit_pure,
)
from .identity.evidence import IdentityEvidence
from .identity.evidence_grades import (
    CLOSED_WORLD,
    HOUSEHOLD_CLASSIFIER,
    NATIVE_TEMPLATE,
    WEARABLE,
    EvidenceGrade,
    grade_of,
)
from .identity.gallery_scoring import (
    GalleryScoringConfig,
    aggregate_max_over_views,
    aggregate_mean,
    cap_votes,
    score_hits,
)
from .identity.household_classifier import (
    HouseholdClassifierChannel,
    HouseholdClassifierConfig,
    HouseholdClassifierEvaluator,
)
from .identity.policy import CommitPolicy as _CommitPolicy
from .identity.posterior import EvidencePosterior, combine_channels
from .identity.types import IdentityAuthority
from .identity.wearable_channel import WearableChannel
from .world.wearable_config import WearableConfig

logger = get_logger(__name__)


_CommitEvaluation = CommitEvaluation


class _GallerySearchKwargs(TypedDict):
    """Kwargs shape returned by ``IdentityResolver._gallery_search_kwargs``.

    The provenance key ``exclude_seed_sources`` appears only for an image-only
    evaluation arm; the live vote path never narrows by provenance.
    """

    seen_after: datetime | None
    seen_before: datetime
    states: frozenset[str] | None
    exclude_seed_sources: NotRequired[frozenset[str]]


# Rows a wearable taught: the image-only control arm votes without any of them.
_WEARABLE_TAUGHT_SOURCES: frozenset[str] = frozenset(
    {str(GallerySeedSource.WEARABLE_ATTESTED), str(GallerySeedSource.WEARABLE_ROOM_ATTESTED)}
)


@dataclass(frozen=True)
class _GalleryArm:
    """Provenance filter for one image-only evaluation arm's gallery reads.

    ``exclude_seed_sources`` applies to every read, including the PH's own rows that build its
    query embedding: in the control arm's world wearable rows were never minted. ``seen_before``
    applies to the votes only: it stops rows from the current stretch voting for themselves,
    while the PH's recent rows still describe what it looks like now.
    """

    exclude_seed_sources: frozenset[str] = frozenset()
    seen_before: datetime | None = None


@dataclass(frozen=True)
class _UnheldEntity:
    """An entity viewed as if it held no identity, for image-only evaluation.

    A held identity may itself come from the wearable, and the temporal prior and sticky
    maintenance would carry it into the "image-only" decision. Both arms therefore decide from
    this round's image evidence alone, which keeps them paired and free of the wearable.
    """

    inner: IdentityResolvableEntity

    @property
    def entity_id(self) -> str:
        return self.inner.entity_id

    @property
    def observation_ids(self) -> list[str]:
        return self.inner.observation_ids

    @property
    def camera_ids(self) -> list[str]:
        return self.inner.camera_ids

    @property
    def current_identity_id(self) -> str | None:
        return None

    @property
    def current_identity_committed_at(self) -> datetime | None:
        return None

    @property
    def last_independent_identity_evidence_at(self) -> datetime | None:
        return None

    @property
    def last_seen_at(self) -> datetime:
        return self.inner.last_seen_at

    @property
    def started_at(self) -> datetime:
        return self.inner.started_at

    @property
    def view_prototypes(self) -> tuple[ViewPrototype, ...]:
        return self.inner.view_prototypes


def _sampled(rate: float) -> bool:
    """The one sample gate for shadow work that issues I/O: off unless the rate is positive."""
    return rate > 0.0 and random.random() < rate


@dataclass(frozen=True)
class _PendingIdentityDecision:
    """Identity decision plus batch-scoped evidence used by guardrails."""

    entity: IdentityResolvableEntity
    decision: IdentityDecision
    direct_face_confidence: float
    wearable_confidence: float = 0.0


FaceQualityMode = Literal["off", "shadow", "enabled"]
HeightMode = Literal["off", "shadow", "enabled"]


@dataclass(frozen=True)
class ResolverConfig:
    """Configuration for the identity resolver."""

    # Probability threshold to commit an identity assignment.
    # With prior_weight=0.6, a strong face anchor (p_face≈0.788) and
    # N identities, the posterior for the face-matched identity is
    # roughly prior_weight * p_face / (prior_weight + (1-prior_weight)/N).
    # For N=2 this is ~0.706; for N=1 it is ~0.813.  commit_prob=0.65
    # lets strong evidence commit in typical deployments (N≤5) while
    # still guarding against weak evidence.
    commit_prob: float = 0.65

    # Margin (top - second) required to commit.
    commit_margin: float = 0.15

    # ReID similarity midpoint for the logistic likelihood curve.
    reid_decision_sim: float = 0.70

    # Revision horizon in seconds. Revisions only apply backward within
    # this window from the revision time.
    revision_horizon_s: float = 600.0

    # Maximum revisions per PH per minute (rate limiting).
    max_revisions_per_ph_per_minute: int = 3

    # Unknown mass: minimum probability for the UNKNOWN state.
    # Prevents weak-but-best matches from crowding out "we do not know".
    unknown_mass: float = 0.05

    # Prior weight for the temporal prior (before evidence is combined).
    # Must be > 0.5 / n_identities to ensure the current identity beats
    # uniform smoothing. With 10 identities, 0.5/10 = 0.05, so 0.6 is
    # safe for any reasonable number of identities.
    prior_weight: float = 0.6

    # Multiplier applied to face likelihood weights before combining with
    # ReID and prior.  Face evidence (ArcFace) is much more reliable than
    # body ReID (SOLIDER) in multi-generational households where clothing
    # similarities can confuse the ReID embedder.  A value of 3.0 means
    # face evidence carries 3x the weight of ReID evidence.
    face_weight_multiplier: float = 3.0

    # Multiplier applied to *propagated* face evidence.  Synthetic face
    # anchors created by cross-GT propagation are not direct ArcFace matches
    # and must carry less weight.  Default 0.5 means propagated evidence
    # has half the weight of direct face evidence.
    propagated_face_weight_multiplier: float = 0.5

    # Log-likelihood power applied to standing-height evidence when height_mode is enabled.
    # Standing height is a corroborating clothing-independent cue (never grants authority or
    # refreshes the evidence clock).
    height_weight_multiplier: float = 1.5

    # Rollout mode for single-view standing height evidence:
    # "off": ignored; "shadow": evaluates HeightChannel and counts disagreements in
    # cts_identity_shadow_mismatch_total{feature="height"} without altering live decisions;
    # "enabled": includes HeightChannel in the live Bayesian posterior.
    height_mode: HeightMode = "shadow"

    # --- Evidence quality gates ---
    # Minimum PH crop-quality EMA needed for a new identity commit.
    # CropQuality maxes out at 0.60 without pose (0.30 area + 0.30 detector
    # confidence) and 0.80 with pose, so 0.35 blocks clearly poor crops while
    # allowing ordinary no-pose detections to commit.
    min_quality_to_commit: float = 0.35

    # Quality gate — now enabled by default.
    enable_quality_gate: bool = True

    # Higher commit threshold used in dense scenes (≥ 2 candidate
    # identities with posterior > 0.3).  Prevents confident-but-wrong
    # commits when two enrolled people are in the same room.
    commit_prob_dense: float = 0.80

    # Larger margin required in dense scenes.  When two identities have
    # posteriors like [0.55, 0.40, 0.05], the narrow 0.15 gap should
    # refuse to commit — the resolver must wait for stronger evidence.
    commit_margin_dense: float = 0.20

    # --- Identity flip debounce ---
    # Reversals inside this window must clear dense-scene thresholds.  This
    # dampens A -> B -> A oscillation caused by weak or marginal evidence.
    flip_debounce_window_s: float = 10.0

    # Shadow flag for the flip debounce.  When false, live behavior is unchanged
    # and cts_identity_shadow_mismatch_total{feature="flip_debounce"} counts
    # decisions the debounce would have changed.
    enable_flip_debounce: bool = False

    # Duplicate-active identity guard — authoritative.
    enable_duplicate_active_identity_guard: bool = True

    # Minimum direct-face raw cosine similarity required to bypass the duplicate-active
    # identity guard.  This is stricter than face_commit_min_similarity because
    # the invariant is about two simultaneously visible PHs sharing one person.
    duplicate_identity_direct_face_min_similarity: float = 0.90

    # Sticky maintenance — hold a committed identity within the maintenance
    # window unless strongly contradicted, even when the posterior argmax is
    # UNKNOWN or a weakly-supported other identity.  Encodes the product-owner
    # choice to "favor continuity and stability" under weak evidence.
    enable_sticky_maintenance: bool = False

    # A different identity's recognized face at or above this raw cosine similarity
    # contradicts the held identity during sticky maintenance.
    contradiction_face_similarity: float = 0.70

    # --- Rich face evidence ---
    # Multiplier applied to candidate (grey-zone) face evidence in _combine.
    # Sits below face_weight_multiplier (3.0) and above neutral (1.0).
    # A candidate face for an identity contributes a weak positive scaled by
    # the already-low raw similarity, so the effective contribution is small.
    candidate_face_weight_multiplier: float = 1.0

    # Mass added to the UNKNOWN entry of the face likelihood when a face
    # region is present but unrecognized.  Small (0.10) and applied to
    # UNKNOWN only — never penalizes a held identity.
    face_present_unknown_unknown_mass: float = 0.10

    # Yaw angle (degrees) at or below which frontality factor = 1.0.
    # A face looking directly at the camera is fully weighted.
    frontality_full_yaw_deg: float = 15.0

    # Yaw angle (degrees) at or above which frontality factor = frontality_min_factor.
    # A face in near-profile or beyond is down-weighted.
    frontality_zero_yaw_deg: float = 60.0

    # Floor of the frontality factor for extreme yaw angles.
    # An off-axis recognized face is still positive but down-weighted,
    # reducing false commits from glancing matches.
    frontality_min_factor: float = 0.3

    # A different non-UNKNOWN identity clearing both these dense-scene
    # thresholds contradicts the held identity during sticky maintenance.
    contradiction_posterior_prob: float = 0.80
    contradiction_posterior_margin: float = 0.20

    # Maximum age (seconds) of last_independent_identity_evidence_at before
    # prior-only maintenance expires.  Measured from the last qualifying
    # independent evidence (direct face or ReID), not from the last label write.
    prior_maintenance_max_age_s: float = 30.0

    # Minimum cosine similarity for applying the face-confirmed entry
    # likelihood boost.  Below this value the raw logistic is used
    # unchanged.  At or above it, the logistic is raised to
    # identified_entry_min_likelihood because face recognition already
    # confirmed the identity of that gallery entry — the lower raw
    # similarity only reflects a viewpoint change (front→back), not a
    # different person.
    identified_entry_boost_min_sim: float = 0.65

    # Likelihood floor applied to face-confirmed gallery entries whose
    # similarity exceeds identified_entry_boost_min_sim.
    # Without this floor, a back-facing query finding front-facing
    # alice entries at sim≈0.73 produces logistic≈0.65, which after
    # _combine smoothing collapses to posterior≈0.35 — below
    # commit_prob.  The floor ensures the posterior crosses commit_prob
    # even when only alice's entries appear in the top-k results.
    identified_entry_min_likelihood: float = 0.80

    # --- Gallery vote scoring (unified across every gallery query path) ---
    # Trust multiplier for operator_verified gallery entries before aggregation.
    gallery_verified_trust_multiplier: float = 2.0

    # Trust multiplier for auto_verified gallery entries (identity-continuity
    # Calibrated high-confidence face matches minted
    # directly at candidate creation, below operator_verified's full trust.
    gallery_auto_verified_trust_multiplier: float = 1.5

    # Trust multiplier for auto_verified rows minted from a trajectory-attested wearable
    # association: below a face-minted row, because the wearable never saw the face.
    gallery_wearable_trust_multiplier: float = 1.25

    # Exponential recency half-life in days, no floor. Default 2.0
    # (shortened from the original 7.0).
    gallery_recency_half_life_days: float = 2.0

    # Hard vote-age cutoff in seconds: gallery entries older than this
    # contribute zero vote weight on every path (,
    # decision D4). None means no cutoff (rollback state; production default
    # is 43200s/12h via settings.yaml). SOLIDER-REID embeddings are
    # clothing-dominated, so a stale entry describes yesterday's outfit; the
    # cutoff encodes the wardrobe-change step function that recency decay
    # alone cannot.
    gallery_vote_max_age_s: float | None = None

    # --- Embedding coherence (sticky ReID) ---
    # When enabled, if a GlobalTrack's last N gallery entries are all
    # mutually similar (embedding drift ≤ 1 - embedding_coherence_min_sim),
    # the best-matching identity gets a likelihood boost.  This prevents
    # identity loss when the person's pose is stable but the gallery search
    # returns scores just below the commit threshold.
    # Disabled by default — turn on via ResolverConfig(enable_embedding_coherence_boost=True).
    enable_embedding_coherence_boost: bool = False

    # Number of consecutive gallery entries to inspect for coherence.
    embedding_coherence_window: int = 5

    # Minimum cosine similarity between every consecutive pair in the window.
    # If any pair falls below this, coherence is not declared.
    embedding_coherence_min_sim: float = 0.70

    # Multiplier applied to the best-matching identity's likelihood when
    # coherence is active.  Capped at 0.99 after multiplication.
    embedding_coherence_boost: float = 2.0

    # Fraction of frames where the coherence-boost shadow comparison runs.
    # 0.0 = disabled (no shadow query).  Set > 0 to measure what enabling
    # coherence boost would change.
    coherence_shadow_sample_rate: float = 0.0

    # Fraction of frames where the multiview-gallery shadow comparison runs
    # (only evaluated when enable_multiview_gallery is False and the entity
    # has view prototypes).  0.0 = disabled (no shadow query).  Set > 0 to
    # measure what flipping enable_multiview_gallery off would change.
    multiview_shadow_sample_rate: float = 0.0

    # Minimum face anchor confidence to drive a commit via the posterior path.
    face_commit_min_similarity: float = 0.70

    # Calibrated ArcFace confidence required for direct-face authority.
    # Fails closed: authority never fires when calibrated_confidence is None.
    arcface_authority_calibrated_confidence: float = 0.80

    # --- External identity evidence  ---
    # Multiplier applied to FaceAnchor.origin == "cc_assertion" likelihood
    # weight in _from_face_anchors. Same recognizer as native ArcFace, but a
    # cross-fleet spatial match (room or floor-distance gated) is weaker
    # association than a same-frame native anchor; this is the knob shadow
    # data calibrates.
    cc_assertion_likelihood_scale: float = 0.5

    # Rollout mode for cc.identity_assertions matching: "off" preserves
    # pre-behavior exactly (matcher not consulted); "shadow" runs the
    # matcher and records outcome metrics without injecting anchors;
    # "enabled" injects matched anchors as evidence. Flip is a config
    # change, never a code change (shadow-before-authority house rule).
    cc_assertion_mode: str = "shadow"

    # Confidence multiplier for a room-matched (no floor point) external
    # assertion, applied on top of its calibrated confidence.
    room_match_confidence_scale: float = 0.8

    # Conservative defaults for external-assertion metadata the wire does
    # not carry (has_yaw/has_quality unset). Never 1.0 quality or 0.0 yaw:
    # those would make external evidence look better than native evidence.
    cc_assertion_default_quality: float = 0.5
    cc_assertion_default_yaw_deg: float = 60.0

    # --- Face image quality ---
    # Whether the service-measured face quality (FaceAnchor.face_quality) weighs recognized face
    # evidence. "off": the anchor's evidence-path quality only (pre-existing behavior). "shadow":
    # live as "off", plus a quality-weighted evaluation that only counts disagreements.
    # "enabled": recognized-anchor weight is evidence-path quality times face quality.
    face_quality_mode: FaceQualityMode = "off"
    # Face quality assumed for a native anchor whose service sent no score, once the mode is not
    # "off". Conservative, like cc_assertion_default_quality: never 1.0, which would rank an
    # unmeasured face above a measured good one.
    face_quality_missing_default: float = 0.5

    def __post_init__(self) -> None:
        if self.face_quality_mode not in get_args(FaceQualityMode):
            raise ValueError(
                f"resolver.face_quality_mode must be one of {get_args(FaceQualityMode)}, "
                f"got {self.face_quality_mode!r}"
            )
        if not 0.0 <= self.face_quality_missing_default <= 1.0:
            raise ValueError("resolver.face_quality_missing_default must be in [0, 1]")
        if self.face_evidence_policy not in FACE_EVIDENCE_POLICIES:
            raise ValueError(
                f"face evidence policy must be one of {FACE_EVIDENCE_POLICIES}, "
                f"got {self.face_evidence_policy!r}"
            )
        if not 0.0 <= self.uncalibrated_face_max_likelihood <= 1.0:
            raise ValueError("resolver.uncalibrated_face_max_likelihood must be in [0, 1]")

    # --- Cross-GT face propagation ---
    # Minimum gallery cosine similarity for propagating a face-confirmed identity
    # to an adjacent GlobalTrack that has no face anchor of its own. Propagation
    # also requires src_face_confidence * gallery_similarity to clear
    # face_commit_min_similarity, so 0.72 rejects weak ReID similarity while
    # preserving high-confidence doorway handoffs.
    cross_gt_face_propagation_threshold: float = 0.72

    # Maximum number of adjacent GlobalTracks to propagate face identity to per
    # resolve call.  Caps the gallery query overhead for busy scenes.
    cross_gt_face_propagation_max_gts: int = 4

    # --- Multi-view gallery query ---
    # When enabled, _from_gallery builds per-orientation queries from the PH's
    # view_prototypes and takes the max-over-views logistic similarity per
    # identity.  Ships shadow-first (off by default).
    enable_multiview_gallery: bool = False

    # Minimum orientation confidence for seeding a gallery entry from a
    # face-recognized PH's non-frontal observation.
    seed_orientation_min_confidence: float = 0.5

    # --- Unknown-segment backfill ---
    # On a qualifying first calibrated-face commit of a previously Unknown PH,
    # CTS can automatically backfill the PH's Unknown history to the newly
    # committed identity (inferred revision range + retroactive relabel of
    # identity-NULL trajectory/dwell rows). Staged rollout: off, then shadow
    # (metrics/logs only), then enabled. See UnknownBackfillService.
    enable_unknown_backfill: bool = False
    backfill_shadow: bool = True
    # Maximum span (seconds) a backfill may relabel backward from the commit,
    # regardless of how much earlier the PH's Unknown history extends.
    backfill_max_range_s: float = 14400.0
    # Wearable trigger for the same backfill: a trajectory-attested association backfills the
    # PH's identity-NULL rows over the attested segment. Waits on the owner's confirmation that
    # attested intervals may be backfilled; shares backfill_shadow and backfill_max_range_s.
    enable_wearable_backfill: bool = False

    # --- Face evidence policy ---
    # Set from face_id.evidence_policy (app/main.py), the same value FaceIdentityStage emits by,
    # so emission and weighting cannot disagree. "legacy": a recognized face weighs by _p_face.
    # "shadow": live as "legacy", plus a calibrated evaluation per PH that only counts outcomes.
    # "calibrated": a recognized face weighs by its calibrated confidence; a face without one
    # weighs by _p_face capped at uncalibrated_face_max_likelihood and, like every face that is
    # not a calibrated recognized one, corroborates only (never commit or clock evidence).
    face_evidence_policy: FaceEvidencePolicy = "legacy"
    # Cap on the likelihood weight of a recognized face without a calibrated confidence (a
    # resident outside the calibration artifact's coverage, a propagated or external anchor).
    # Below the 0.80 authority threshold and close to an even bet, so it can tip a posterior that
    # other evidence already supports but cannot carry one alone.
    uncalibrated_face_max_likelihood: float = 0.55


class IdentityResolver:
    """Bayesian identity resolver with retroactive revision.

    Usage::

        resolver = IdentityResolver(
            gallery_repo=gallery_repo,
            config=ResolverConfig(),
        )

        outcome = await resolver.resolve(
            hypotheses=active_phs,
            new_face_anchors=face_anchors,
            captured_at=frame.capture_time,
        )

        for decision in outcome.decisions:
            # Update the global track's identity
            ...

        for revision in outcome.revisions:
            # Emit to tracking.revisions stream
            ...
    """

    # How long (seconds) to reuse the cached enrolled-identity list before
    # re-querying the DB.  Enrollment/removal is rare (a few times per week in
    # production), so 10 s strikes the right balance: a newly-enrolled person
    # affects the prior only weakly anyway (the face anchor drives the first
    # commit), while the DB query saved per frame is meaningful at 5+ Hz with
    # multiple active PHs.
    _IDENTITY_LIST_TTL_S: float = 10.0

    def __init__(
        self,
        gallery_repo: GalleryRepository,
        identities: list[Identity] | None = None,
        config: ResolverConfig | None = None,
        gallery_cache: GalleryCache | None = None,
        wearable_config: WearableConfig | None = None,
        height_profile_repo: IdentityHeightProfileRepository | None = None,
        height_profiles: Sequence[IdentityHeightProfile] | None = None,
        closed_world_config: ClosedWorldConfig | None = None,
        household_presence_source: HouseholdPresenceSource | None = None,
        household_classifier_config: HouseholdClassifierConfig | None = None,
        household_classifier_evaluator: HouseholdClassifierEvaluator | None = None,
    ) -> None:
        self._gallery_repo = gallery_repo
        self._height_profile_repo = height_profile_repo
        self._config = config or ResolverConfig()
        self._gallery_cache = gallery_cache
        self._wearable_config = wearable_config or WearableConfig()
        self._closed_world_config = closed_world_config or ClosedWorldConfig()
        self._household_presence_source = household_presence_source
        self._household_classifier_config = (
            household_classifier_config or HouseholdClassifierConfig()
        )
        self._household_classifier_evaluator = household_classifier_evaluator
        # Known identities for display names
        self._identities: dict[str, Identity] = {
            ident.identity_id: ident for ident in identities or []
        }
        self._height_profiles: dict[str, IdentityHeightProfile] = {
            profile.identity_id: profile for profile in height_profiles or []
        }
        # Timestamps for the TTL-based identity-list cache.
        self._identities_loaded_at: datetime | None = (
            None  # None forces an immediate load on first resolve
        )
        # Revision rate limiter: global_track_id -> list of revision timestamps
        self._revision_log: dict[str, list[datetime]] = defaultdict(list)
        # Evidence channels in canonical order:
        # face -> gallery -> height -> wearable -> closed_world -> household_classifier.
        self._face_channel = FaceChannel(
            config=self._config,
            identities_getter=lambda: self._identities,
        )
        self._gallery_channel = GalleryChannel(
            query_fn=self._from_gallery,
            support_multiplier=1.0,
        )
        self._height_channel = HeightChannel(
            height_profiles_getter=lambda: self._height_profiles,
            height_weight_multiplier=self._config.height_weight_multiplier,
        )
        self._wearable_channel = WearableChannel(
            config=self._wearable_config,
            identities_getter=lambda: self._identities,
        )
        self._closed_world_channel = ClosedWorldChannel(
            config=self._closed_world_config,
            identities_getter=lambda: self._identities,
        )
        self._household_classifier_channel = HouseholdClassifierChannel(
            config=self._household_classifier_config,
            evaluator=self._household_classifier_evaluator,
        )
        self._channels: tuple[EvidenceChannel, ...] = (
            self._face_channel,
            self._gallery_channel,
            self._height_channel,
            self._wearable_channel,
            self._closed_world_channel,
            self._household_classifier_channel,
        )

    def likelihood_scale(self, grade: EvidenceGrade) -> float:
        """Look up the configured likelihood scale for *grade*."""
        return self._face_channel.likelihood_scale(grade)

    def invalidate_identity_cache(self) -> None:
        """Invalidate the cached enrolled-identity roster, height profiles, and gallery cache."""
        self._identities_loaded_at = None
        if self._height_profile_repo is not None:
            self._height_profiles = {}
        if self._gallery_cache is not None:
            self._gallery_cache.invalidate()

    async def resolve(
        self,
        hypotheses: Sequence[IdentityResolvableEntity],
        new_face_anchors: list[FaceAnchor],
        captured_at: datetime,
        ph_heights: dict[str, float] | None = None,
        ph_qualities: dict[str, float] | None = None,
        face_evidence: list[FaceEvidence] | None = None,
        open_ph_identities: Mapping[str, str] | None = None,
        wearable_evidence: Mapping[str, WearableEvidence] | None = None,
        ph_height_sigmas: dict[str, float] | None = None,
        open_ph_occupancies: Mapping[str, OpenPHOccupancy] | None = None,
        copresence_pairs: frozenset[tuple[str, str]] | None = None,
        overlap_groups: Sequence[OverlapGroup] | None = None,
        camera_room_map: Mapping[str, str] | None = None,
        wearable_present_identities: frozenset[str] | None = None,
        household_presence: HouseholdPresenceSnapshot | None = None,
        ph_appearance_features: Mapping[str, AppearanceFeature] | None = None,
    ) -> ResolveOutcome:
        """Resolve identities for a batch of tracked entities.

        Args:
            hypotheses: active entities (PHs or GlobalTracks) to resolve.
            new_face_anchors: face anchors from this frame.
            captured_at: wall-clock time of the current frame.
            ph_heights: optional entity_id → height_m mapping.
            ph_qualities: optional entity_id → rolling crop quality mapping.
            open_ph_identities: optional ph_id → committed identity_id for every
                currently-open PH, including PHs not observed this frame. Feeds
                the duplicate-active-identity guard so an incumbent holder that
                is momentarily undetected still protects its identity.
            face_evidence: optional typed FaceEvidence records with source
                metadata. When provided, direct evidence receives normal
                weight and propagated evidence receives reduced weight.

        Returns:
            ResolveOutcome with decisions and any revisions to emit.
        """
        # Refresh the known-identity list on first call and after the TTL
        # expires.  Enrollment is rare (a few times per week); a 10-second
        # staleness window eliminates the per-frame DB query at 5+ Hz with
        # multiple active PHs without meaningfully delaying new enrolments
        # (the face anchor drives the first commit, not the prior list).
        now_dt = captured_at
        if self._identities_loaded_at is None or (
            (now_dt - self._identities_loaded_at).total_seconds() >= self._IDENTITY_LIST_TTL_S
        ):
            enrolled = await self._gallery_repo.list_identities(active_only=True)
            self._identities = {ident.identity_id: ident for ident in enrolled}
            if self._height_profile_repo is not None:
                loaded_profiles = await self._height_profile_repo.list_all()
                self._height_profiles = {
                    p.identity_id: p for p in loaded_profiles if p.identity_id in self._identities
                }
            self._identities_loaded_at = now_dt

        # Shadow-only anchors feed their shadow evaluation and nothing else: from here on every
        # live path (propagation, likelihood, authority, the evidence sets, duplicate ranking)
        # sees only the live anchors. Frame anchors below the legacy floor feed the calibrated
        # shadow; face templates in shadow mode feed the template shadow.
        shadow_face_anchors = [
            fa for fa in new_face_anchors if fa.shadow_only and grade_of(fa) is not NATIVE_TEMPLATE
        ]
        template_shadow_anchors = [
            fa for fa in new_face_anchors if fa.shadow_only and grade_of(fa) is NATIVE_TEMPLATE
        ]
        new_face_anchors = [fa for fa in new_face_anchors if not fa.shadow_only]
        policy = self._config.face_evidence_policy
        live_calibrated = policy == "calibrated"

        # Propagate face anchors from face-evidenced entities to similar adjacent
        # entities that share the same physical space but weren't merged by the
        # cross-camera associator.
        augmented_anchors = await self._propagate_face_anchors(hypotheses, new_face_anchors)
        augmented_evidence = self._augment_face_evidence(
            hypotheses, face_evidence or [], augmented_anchors
        )

        vote_window = VoteWindow(
            as_of=captured_at,
            max_age_s=self._config.gallery_vote_max_age_s,
        )
        effective_presence = (
            household_presence
            if household_presence is not None
            else (
                self._household_presence_source.latest()
                if self._household_presence_source is not None
                else None
            )
        )
        inputs = ResolverInputs(
            captured_at=captured_at,
            face_anchors=tuple(augmented_anchors),
            face_evidence=tuple(augmented_evidence),
            ph_heights_m=ph_heights or {},
            ph_qualities=ph_qualities or {},
            ph_height_sigmas_m=ph_height_sigmas or {},
            raw_face_anchors=tuple(new_face_anchors),
            raw_face_evidence=tuple(face_evidence or ()),
            wearable_evidence=wearable_evidence or {},
            vote_window=vote_window,
            open_ph_occupancies=open_ph_occupancies or {},
            copresence_pairs=copresence_pairs or frozenset(),
            overlap_groups=tuple(overlap_groups or ()),
            camera_room_map=camera_room_map or {},
            household_presence=effective_presence,
            wearable_present_identities=wearable_present_identities or frozenset(),
            ph_appearance_features=ph_appearance_features or {},
        )

        wearable_mode = self._wearable_config.mode
        height_mode = self._config.height_mode
        closed_world_mode = self._closed_world_config.mode
        classifier_mode = self._household_classifier_config.mode
        empty_height_ch = ChannelLikelihood("height", PosteriorDist({}), 1.0)
        pending_decisions: list[_PendingIdentityDecision] = []
        outcome_image_only: list[ImageOnlyEvaluation] = []

        for entity in hypotheses:
            prior = self._build_prior(entity, captured_at)
            live_quality_mode: FaceQualityMode = (
                "enabled" if self._config.face_quality_mode == "enabled" else "off"
            )
            face_res = self._face_channel.evaluate_detailed(
                entity,
                inputs,
                face_quality_mode=live_quality_mode,
                calibrated=live_calibrated,
            )
            gallery_ch = await self._gallery_channel.evaluate(entity, inputs)
            raw_height_ch = (
                await self._height_channel.evaluate(entity, inputs)
                if height_mode != "off"
                else empty_height_ch
            )
            height_ch = raw_height_ch if height_mode == "enabled" else empty_height_ch
            if height_mode == "enabled" and height_ch.distribution.distribution:
                metrics.metrics.height_evidence_frames_total.inc()

            raw_wearable_ev = inputs.wearable_evidence.get(entity.entity_id)
            live_wearable_ev = raw_wearable_ev if wearable_mode == "enabled" else None
            wearable_ch, wearable_prob = self._wearable_channel.evaluate_wearable(
                entity, live_wearable_ev
            )
            cw_eval = (
                self._closed_world_channel.evaluate_closed_world(entity, inputs)
                if closed_world_mode != "off"
                else None
            )
            hc_eval = (
                self._household_classifier_channel.evaluate_classifier(entity, inputs)
                if classifier_mode != "off"
                else None
            )
            channel_list: list[ChannelLikelihood] = [face_res.likelihood, gallery_ch, height_ch]
            if live_wearable_ev is not None:
                channel_list.append(wearable_ch)
            if closed_world_mode == "enabled" and cw_eval is not None and cw_eval.active:
                channel_list.append(cw_eval.likelihood)
            if classifier_mode == "enabled" and hc_eval is not None and hc_eval.active:
                channel_list.append(hc_eval.likelihood)
            channels: tuple[ChannelLikelihood, ...] = tuple(channel_list)
            face_likelihood = face_res.likelihood.distribution
            best_face_conf = face_res.best_face_similarity
            reid_likelihood = gallery_ch.distribution
            height_likelihood = height_ch.distribution
            posterior = combine_channels(prior, channels)
            entity_quality = inputs.ph_qualities.get(entity.entity_id, 1.0)

            # Identities with real (non-smoothed) face/ReID/wearable/classifier support this frame,
            # for commit eligibility. Built once and reused across the live
            # evaluation and every shadow evaluation below so shadows compare
            # like against like (same evidence set, different posterior/reid).
            # Prior-shaping channels with counts_as_commit_evidence=False (such as
            # CLOSED_WORLD) are intentionally excluded so they can never commit alone.
            base_commit_evidence_ids = self._face_channel.collect_commit_evidence_ids(
                entity,
                face_anchors=augmented_anchors,
                face_evidence=augmented_evidence,
                reid_likelihood=reid_likelihood,
                calibrated_faces_only=live_calibrated,
            )
            commit_evidence_ids = base_commit_evidence_ids
            if live_wearable_ev is not None and WEARABLE.counts_as_commit_evidence:
                commit_evidence_ids = commit_evidence_ids | frozenset(
                    {live_wearable_ev.identity_id}
                )
            if (
                classifier_mode == "enabled"
                and hc_eval is not None
                and hc_eval.active
                and HOUSEHOLD_CLASSIFIER.counts_as_commit_evidence
                and hc_eval.likelihood.commit_evidence_ids
            ):
                commit_evidence_ids = commit_evidence_ids | hc_eval.likelihood.commit_evidence_ids
            assert not CLOSED_WORLD.counts_as_commit_evidence

            # Shared "live" evaluation for both shadow comparisons below, computed
            # once instead of once per shadow block. contradicted=False matches
            # the shadows' existing semantics.
            shadow_live_eval = self._evaluate_commit(
                entity,
                posterior,
                face_likelihood,
                reid_likelihood,
                captured_at,
                entity_quality,
                contradicted=False,
                enable_sticky_maintenance=self._config.enable_sticky_maintenance,
                enforce_quality_gate=self._config.enable_quality_gate,
                enforce_flip_debounce=self._config.enable_flip_debounce,
                evidence_identity_ids=commit_evidence_ids,
            )

            # Shadow multiview gallery query -- sampled, never on by default.
            multiview_rate = self._config.multiview_shadow_sample_rate
            if (
                not self._config.enable_multiview_gallery
                and entity.view_prototypes
                and multiview_rate > 0.0
                and random.random() < multiview_rate
            ):
                await self._shadow_gallery_compare(
                    entity,
                    prior,
                    face_likelihood,
                    height_likelihood,
                    reid_likelihood,
                    shadow_live_eval,
                    captured_at,
                    entity_quality,
                    commit_evidence_ids,
                    feature="multiview_gallery",
                    vote_window=vote_window,
                    enable_multiview=True,
                )

            # Shadow coherence-boost gallery query -- sampled. Default rate 0.0
            # means no shadow query.
            coherence_rate = self._config.coherence_shadow_sample_rate
            if (
                not self._config.enable_embedding_coherence_boost
                and coherence_rate > 0.0
                and random.random() < coherence_rate
            ):
                await self._shadow_gallery_compare(
                    entity,
                    prior,
                    face_likelihood,
                    height_likelihood,
                    reid_likelihood,
                    shadow_live_eval,
                    captured_at,
                    entity_quality,
                    commit_evidence_ids,
                    feature="coherence_boost",
                    vote_window=vote_window,
                    enable_coherence_boost=True,
                )

            # Build identity evidence ledger for this entity.
            evidence_items = self._build_evidence_ledger(
                entity,
                face_likelihood,
                reid_likelihood,
                augmented_evidence,
                best_face_conf,
                captured_at,
                wearable_evidence=live_wearable_ev,
                wearable_probability=wearable_prob,
            )

            arcface_authority = self._check_arcface_authority(entity, augmented_anchors)
            # Face-quality shadow: pure CPU over this frame's anchors, so it is not sample-gated.
            # ArcFace authority decides without the posterior, so face quality cannot change it.
            if self._config.face_quality_mode == "shadow" and arcface_authority is None:
                self._shadow_face_quality(
                    entity,
                    augmented_anchors,
                    augmented_evidence,
                    prior,
                    posterior,
                    face_likelihood,
                    best_face_conf,
                    reid_likelihood,
                    height_likelihood,
                    captured_at,
                    entity_quality,
                    commit_evidence_ids,
                    calibrated=live_calibrated,
                )
            decision = self._commit(
                entity,
                posterior,
                face_likelihood,
                reid_likelihood,
                captured_at,
                best_face_conf,
                entity_quality=entity_quality,
                arcface_authority=arcface_authority,
                evidence_identity_ids=commit_evidence_ids,
                wearable_evidence=live_wearable_ev,
            )
            # Wearable shadow counterfactual: evaluate with WearableChannel added
            # without altering the live decision.
            if wearable_mode == "shadow" and raw_wearable_ev is not None:
                self._shadow_wearable(
                    entity=entity,
                    wearable_evidence=raw_wearable_ev,
                    prior=prior,
                    base_channels=channels,
                    face_likelihood=face_likelihood,
                    reid_likelihood=reid_likelihood,
                    best_face_conf=best_face_conf,
                    captured_at=captured_at,
                    entity_quality=entity_quality,
                    commit_evidence_ids=commit_evidence_ids,
                    arcface_authority=arcface_authority,
                    live_identity_id=decision.identity_id,
                )
            # Height shadow counterfactual: pure CPU over cached height profiles;
            # counts disagreements without changing live decisions.
            if (
                height_mode == "shadow"
                and arcface_authority is None
                and raw_height_ch.distribution.distribution
            ):
                shadow_height_channels = (
                    (face_res.likelihood, gallery_ch, raw_height_ch, wearable_ch)
                    if live_wearable_ev is not None
                    else (face_res.likelihood, gallery_ch, raw_height_ch)
                )
                height_shadow_eval = self._evaluate_channels_counterfactual(
                    entity,
                    prior,
                    shadow_height_channels,
                    face_likelihood=face_likelihood,
                    reid_likelihood=reid_likelihood,
                    best_face_similarity=best_face_conf,
                    captured_at=captured_at,
                    entity_quality=entity_quality,
                    evidence_identity_ids=commit_evidence_ids,
                    check_contradiction=True,
                )
                self._shadow_mismatch("height", decision.identity_id, height_shadow_eval)
            # Closed-world shadow counterfactual: pure CPU over cached presence and open PHs;
            # counts outcomes without altering live decisions or commit evidence eligibility.
            if (
                closed_world_mode == "shadow"
                and arcface_authority is None
                and cw_eval is not None
                and cw_eval.active
            ):
                cw_shadow_channels = (*channels, cw_eval.likelihood)
                cw_shadow_eval = self._evaluate_channels_counterfactual(
                    entity,
                    prior,
                    cw_shadow_channels,
                    face_likelihood=face_likelihood,
                    reid_likelihood=reid_likelihood,
                    best_face_similarity=best_face_conf,
                    captured_at=captured_at,
                    entity_quality=entity_quality,
                    evidence_identity_ids=commit_evidence_ids,
                    check_contradiction=True,
                )
                self._record_closed_world_shadow_outcome(
                    entity,
                    live_identity_id=decision.identity_id,
                    shadow_identity_id=cw_shadow_eval.new_id,
                )
            # Household appearance classifier shadow counterfactual: pure CPU over
            # the validated JSON artifact; records outcomes without altering live decisions.
            if (
                classifier_mode == "shadow"
                and arcface_authority is None
                and hc_eval is not None
                and hc_eval.active
            ):
                hc_shadow_channels = (*channels, hc_eval.likelihood)
                hc_shadow_evidence_ids = (
                    commit_evidence_ids | hc_eval.likelihood.commit_evidence_ids
                )
                hc_shadow_eval = self._evaluate_channels_counterfactual(
                    entity,
                    prior,
                    hc_shadow_channels,
                    face_likelihood=face_likelihood,
                    reid_likelihood=reid_likelihood,
                    best_face_similarity=best_face_conf,
                    captured_at=captured_at,
                    entity_quality=entity_quality,
                    evidence_identity_ids=hc_shadow_evidence_ids,
                    check_contradiction=True,
                )
                self._record_household_classifier_shadow_outcome(
                    entity,
                    live_identity_id=decision.identity_id,
                    shadow_identity_id=hc_shadow_eval.new_id,
                )
            # Calibrated-policy shadow: pure CPU over this frame's anchors, so not sample-gated.
            if policy == "shadow" and self._has_recognized_face(
                entity, augmented_anchors + shadow_face_anchors
            ):
                self._shadow_calibrated_face(
                    entity,
                    augmented_anchors + shadow_face_anchors,
                    augmented_evidence,
                    prior,
                    reid_likelihood,
                    height_likelihood,
                    captured_at,
                    entity_quality,
                    live_identity_id=decision.identity_id,
                    face_quality_mode=live_quality_mode,
                )
            # Template shadow: the live decision again with this round's shadow templates added.
            entity_templates = self._entity_anchors(entity, template_shadow_anchors)
            if entity_templates:
                counterfactual = self._counterfactual_identity(
                    entity,
                    augmented_anchors + entity_templates,
                    augmented_evidence,
                    prior,
                    reid_likelihood,
                    height_likelihood,
                    captured_at,
                    entity_quality,
                    calibrated=live_calibrated,
                    face_quality_mode=live_quality_mode,
                )
                self._record_shadow_outcome(
                    entity,
                    live_identity_id=decision.identity_id,
                    shadow_identity_id=counterfactual,
                    feature="face_template",
                )
            # Image-only evaluation: wearable truth scores what the cameras alone decide.
            if (
                raw_wearable_ev is not None
                and raw_wearable_ev.attested
                and _sampled(self._wearable_config.evaluation_sample_rate)
            ):
                classifier_arm_likelihood = (
                    hc_eval.likelihood
                    if (classifier_mode != "off" and hc_eval is not None and hc_eval.active)
                    else None
                )
                outcome_image_only.append(
                    await self._image_only_evaluation(
                        entity,
                        raw_wearable_ev,
                        augmented_anchors,
                        augmented_evidence,
                        height_likelihood,
                        captured_at,
                        entity_quality,
                        window=vote_window,
                        calibrated=live_calibrated,
                        face_quality_mode=live_quality_mode,
                        classifier_likelihood=classifier_arm_likelihood,
                    )
                )
            # Attach evidence summary.
            ep = EvidencePosterior(
                distribution=posterior.distribution,
                entropy=posterior.entropy(),
                top_identity=decision.identity_id or "UNKNOWN",
                top_probability=posterior.top_identity()[1],
                margin=posterior.top_with_margin()[1],
                face_evidence_present=any(ev.source == "direct_face" for ev in evidence_items),
                reid_evidence_present=any(ev.source == "reid" for ev in evidence_items),
                evidence_summary={
                    s: sum(1 for ev in evidence_items if ev.source == s)
                    for s in {ev.source for ev in evidence_items}
                },
            )
            # Clock refresh only for evidence that names the *decided* identity,
            # assembled across channels allowed to refresh the clock.
            has_qualifying_evidence, evidence_at = self._resolve_clock_evidence(
                entity,
                decision,
                channels=channels,
            )
            # Use the *original* anchors/evidence (pre-propagation) so the
            # duplicate-guard bypass only trusts a PH's own direct face.
            direct_face_confidence = self._direct_face_similarity(
                entity,
                new_face_anchors,
                face_evidence or [],
                identity_id=decision.identity_id,
            )
            room_wearable_only = bool(
                decision.identity_id is not None
                and live_wearable_ev is not None
                and live_wearable_ev.identity_id == decision.identity_id
                and live_wearable_ev.evidence_class == WearableEvidenceClass.ROOM
                and decision.identity_id not in base_commit_evidence_ids
            )
            decision = IdentityDecision(
                ph_id=decision.ph_id,
                identity_id=decision.identity_id,
                posterior=decision.posterior,
                revises_previous=decision.revises_previous,
                previous_identity_id=decision.previous_identity_id,
                reason=decision.reason,
                evidence_backed=decision.evidence_backed and has_qualifying_evidence,
                evidence={
                    "sources": ep.evidence_summary,
                    "direct_face_confidence": direct_face_confidence,
                    "posterior_entropy": ep.entropy,
                    "room_wearable_only": room_wearable_only,
                },
                decision_id=decision.decision_id,
                inferred_identity_id=decision.inferred_identity_id,
                effective_identity_id=decision.effective_identity_id,
                authority=decision.authority,
                decision_source=decision.decision_source,
                conflict=decision.conflict,
                last_independent_evidence_at_unix_ns=decision.last_independent_evidence_at_unix_ns,
                config_hash=decision.config_hash,
                model_set_version=decision.model_set_version,
                evidence_at=evidence_at,
            )
            wearable_guard_conf = (
                wearable_prob
                if (
                    WEARABLE.ranks_in_duplicate_guard
                    and live_wearable_ev is not None
                    and decision.identity_id == live_wearable_ev.identity_id
                    and wearable_prob >= self._wearable_config.duplicate_guard_min_probability
                )
                else 0.0
            )
            pending_decisions.append(
                _PendingIdentityDecision(
                    entity=entity,
                    decision=decision,
                    direct_face_confidence=direct_face_confidence,
                    wearable_confidence=wearable_guard_conf,
                )
            )

        outcome = ResolveOutcome()
        outcome.image_only = outcome_image_only
        final_decisions = self._apply_duplicate_active_identity_guard(
            pending_decisions, open_ph_identities or {}
        )
        entity_by_id = {item.entity.entity_id: item.entity for item in pending_decisions}

        for decision in final_decisions:
            if decision.revises_previous:
                revision = self._build_revision(entity_by_id[decision.ph_id], decision, captured_at)
                if revision is not None:
                    outcome.revisions.append(revision)
            if (
                decision.identity_id is not None
                and decision.revises_previous
                and await self._has_cross_camera_reid_assist(
                    entity_by_id[decision.ph_id], decision.identity_id, window=vote_window
                )
            ):
                metrics.metrics.reid_cross_camera_assist_total.inc()

            outcome.decisions.append(decision)

        return outcome

    def _apply_duplicate_active_identity_guard(
        self,
        pending: list[_PendingIdentityDecision],
        open_ph_identities: Mapping[str, str],
    ) -> list[IdentityDecision]:
        """Apply or shadow-count the duplicate-active-identity guard."""
        blocked = self._duplicate_active_blocks(pending, open_ph_identities)

        if not self._config.enable_duplicate_active_identity_guard:
            if blocked:
                metrics.metrics.identity_shadow_mismatch_total.labels(
                    feature="duplicate_active_identity"
                ).inc(len(blocked))
            return [item.decision for item in pending]

        if blocked:
            metrics.metrics.identity_duplicate_active_blocks_total.inc(len(blocked))
        final = [self._block_decision(item, blocked) for item in pending]
        self._assert_no_duplicate_active(final, open_ph_identities)
        return final

    @staticmethod
    def _assert_no_duplicate_active(
        final: list[IdentityDecision],
        open_ph_identities: Mapping[str, str],
    ) -> None:
        """Page-now invariant: at most one active PH may hold a household identity.

        Runs after the guard so it catches a guard escape (e.g. two PHs each with
        strong direct face for the same identity, which the holder rule does not
        block). Counts each identity held by more than one active PH; the value
        must stay zero.
        """
        holders: dict[str, set[str]] = defaultdict(set)
        for decision in final:
            if decision.identity_id is not None:
                holders[decision.identity_id].add(decision.ph_id)
        batch_ph_ids = {d.ph_id for d in final}
        for ph_id, identity_id in open_ph_identities.items():
            if ph_id not in batch_ph_ids:
                holders[identity_id].add(ph_id)
        breaches = sum(1 for phs in holders.values() if len(phs) > 1)
        if breaches:
            metrics.metrics.identity_duplicate_active_breach_total.inc(breaches)

    def _duplicate_active_blocks(
        self,
        pending: list[_PendingIdentityDecision],
        open_ph_identities: Mapping[str, str],
    ) -> dict[str, str]:
        """Return {ph_id: blocked_identity_id} for duplicate new assignments.

        An identity may stay with PHs that already hold it (an incumbent open
        PH, this frame or any other) or that carry strong direct recognized face
        evidence.  A *new* assignment to an already-occupied identity is blocked;
        when several new assignments compete for an unoccupied identity, only the
        strongest posterior survives.
        """
        threshold = self._config.duplicate_identity_direct_face_min_similarity
        batch_ph_ids = {item.decision.ph_id for item in pending}

        # Identities already occupied by an open PH that is not in this batch.
        external: set[str] = {
            identity_id
            for ph_id, identity_id in open_ph_identities.items()
            if ph_id not in batch_ph_ids
        }

        by_identity: dict[str, list[_PendingIdentityDecision]] = defaultdict(list)
        for item in pending:
            if item.decision.identity_id is not None:
                by_identity[item.decision.identity_id].append(item)

        blocked: dict[str, str] = {}
        for identity_id, items in by_identity.items():
            holders = {
                item.decision.ph_id
                for item in items
                if item.direct_face_confidence >= threshold
                or (
                    item.decision.previous_identity_id == identity_id
                    and not item.decision.revises_previous
                )
            }
            candidates = [item for item in items if item.decision.ph_id not in holders]
            if not candidates:
                continue

            occupied = bool(holders) or identity_id in external
            if not occupied:
                # No incumbent: rank candidates and pick a winner unless tied.
                # Rank: evidence_backed (True > False), direct_face_confidence,
                # posterior_prob, margin, evidence freshness (more recent = better),
                # then ph_id for stable ordering only.
                def _rank_key(
                    item: _PendingIdentityDecision,
                ) -> tuple[bool, float, float, float, float, float, str]:
                    top_prob = item.decision.posterior.top_identity()[1]
                    margin = item.decision.posterior.top_with_margin()[1]
                    ev_at = item.entity.last_independent_identity_evidence_at
                    freshness = ev_at.timestamp() if ev_at is not None else 0.0
                    return (
                        item.decision.evidence_backed,
                        item.direct_face_confidence,
                        item.wearable_confidence,
                        top_prob,
                        margin,
                        freshness,
                        item.decision.ph_id,  # stable ordering; not a semantic tie-break
                    )

                ranked = sorted(candidates, key=_rank_key, reverse=True)

                # Detect effective tie: top two match on all meaningful criteria.
                def _tied(a: _PendingIdentityDecision, b: _PendingIdentityDecision) -> bool:
                    return (
                        a.decision.evidence_backed == b.decision.evidence_backed
                        and abs(a.direct_face_confidence - b.direct_face_confidence) < 1e-6
                        and abs(a.wearable_confidence - b.wearable_confidence) < 1e-6
                        and abs(
                            a.decision.posterior.top_identity()[1]
                            - b.decision.posterior.top_identity()[1]
                        )
                        < 1e-6
                        and abs(
                            a.decision.posterior.top_with_margin()[1]
                            - b.decision.posterior.top_with_margin()[1]
                        )
                        < 1e-6
                        and a.entity.last_independent_identity_evidence_at
                        == b.entity.last_independent_identity_evidence_at
                    )

                if len(ranked) >= 2 and _tied(ranked[0], ranked[1]):
                    # Tie → block all contenders.
                    candidates_to_block = ranked
                else:
                    # One clear winner; block the rest.
                    candidates_to_block = ranked[1:]

                candidates = candidates_to_block

            for item in candidates:
                blocked[item.decision.ph_id] = identity_id

        return blocked

    @staticmethod
    def _block_decision(
        item: _PendingIdentityDecision,
        blocked: Mapping[str, str],
    ) -> IdentityDecision:
        """Return the decision, demoted to UNKNOWN if the guard blocked it."""
        identity_id = blocked.get(item.decision.ph_id)
        if identity_id is None:
            return item.decision
        return replace(
            item.decision,
            identity_id=None,
            revises_previous=False,
            reason=(
                f"duplicate_active_identity_blocked: {identity_id} "
                f"(direct_face_confidence={item.direct_face_confidence:.3f})"
            ),
            evidence_backed=False,
            authority=str(IdentityAuthority.NONE),
            evidence_at=None,
        )

    @staticmethod
    def _resolve_clock_evidence(
        entity: IdentityResolvableEntity,
        decision: IdentityDecision,
        *,
        channels: Sequence[ChannelLikelihood],
    ) -> tuple[bool, datetime | None]:
        """Determine clock-refresh qualification and newest evidence time across *channels*.

        Collects ``clock_evidence_ids`` and per-identity (or channel-level) ``evidence_at``
        across every channel. The clock never moves backwards for an identity the PH already
        holds.
        """
        if decision.identity_id is None:
            return False, None
        decided = decision.identity_id
        clock_ids: set[str] = set()
        times: list[datetime] = []
        for ch in channels:
            clock_ids.update(ch.clock_evidence_ids)
            if decided in ch.evidence_at_by_identity:
                times.append(ch.evidence_at_by_identity[decided])
            elif ch.evidence_at is not None and decided in ch.clock_evidence_ids:
                times.append(ch.evidence_at)
        has_qualifying = decided in clock_ids
        if not (decision.evidence_backed and has_qualifying):
            return has_qualifying, None
        evidence_at = max(times, default=None)
        previous = entity.last_independent_identity_evidence_at
        if (
            evidence_at is not None
            and previous is not None
            and entity.current_identity_id == decided
            and previous > evidence_at
        ):
            return has_qualifying, previous
        return has_qualifying, evidence_at

    # ------------------------------------------------------------------
    # ArcFace authority
    # ------------------------------------------------------------------

    def _check_arcface_authority(
        self,
        entity: IdentityResolvableEntity,
        face_anchors: list[FaceAnchor],
    ) -> str | None:
        """Check for qualifying authoritative ArcFace anchors for this entity.

        Returns:
        - None        if no qualifying anchor (calibrated_confidence is None or below threshold).
        - "CONFLICT"  if two qualifying anchors name different identities.
        - identity_id if exactly one qualifying identity found.

        Fails closed: calibrated_confidence=None (the default) never creates authority.
        """
        threshold = self._config.arcface_authority_calibrated_confidence
        entity_obs = set(entity.observation_ids)
        qualifying: dict[str, float] = {}  # identity_id → best calibrated_confidence
        for fa in face_anchors:
            if not grade_of(fa).grants_authority:
                continue
            if not (
                fa.tracklet_id in entity_obs
                or fa.tracklet_id == entity.entity_id
                or fa.detection_id in entity_obs
            ):
                continue
            if fa.recognition_state != "recognized":
                continue
            if fa.calibrated_confidence is None or fa.calibrated_confidence < threshold:
                continue
            # A template whose member faces were recognized as someone else votes, but never
            # decides on its own.
            if not fa.authority_eligible:
                continue
            prev_best = qualifying.get(fa.person_id, 0.0)
            if fa.calibrated_confidence > prev_best:
                qualifying[fa.person_id] = fa.calibrated_confidence
        if not qualifying:
            return None
        if len(qualifying) == 1:
            return next(iter(qualifying))
        return "CONFLICT"

    # ------------------------------------------------------------------
    # Prior construction
    # ------------------------------------------------------------------

    def _build_prior(
        self, entity: IdentityResolvableEntity, captured_at: datetime
    ) -> PosteriorDist:
        """Build the temporal prior from the previous identity assignment.

        The prior is a mixture of:
        - The previous identity (if any), weighted by prior_weight.
        - A uniform distribution over all known identities, weighted by (1 - prior_weight).
        - An UNKNOWN mass that grows with the time since last_seen.
        """
        prior_weight = self._config.prior_weight

        if entity.current_identity_id:
            # Strong prior on the previous identity.
            prior: dict[str, float] = {
                entity.current_identity_id: prior_weight,
            }
            # Add small mass to other known identities.
            other_weight = (1 - prior_weight) / max(len(self._identities), 1)
            for ident_id in self._identities:
                if ident_id != entity.current_identity_id:
                    prior[ident_id] = other_weight
        else:
            # No previous identity: uniform over known identities.
            uniform_weight = 1.0 / max(len(self._identities), 1)
            prior = dict.fromkeys(self._identities, uniform_weight)

        # Add UNKNOWN mass.
        prior["UNKNOWN"] = self._config.unknown_mass

        return PosteriorDist(prior)

    # ------------------------------------------------------------------
    # Face anchor likelihood
    # ------------------------------------------------------------------

    def _from_face_anchors(
        self,
        entity: IdentityResolvableEntity,
        face_anchors: list[FaceAnchor],
        face_evidence: list[FaceEvidence] | None = None,
        *,
        face_quality_mode: FaceQualityMode = "off",
        calibrated: bool = False,
    ) -> tuple[PosteriorDist, float | None]:
        """Delegate face-anchor likelihood evaluation to :class:`FaceChannel`."""
        res = self._face_channel.evaluate_anchors(
            entity,
            face_anchors,
            face_evidence,
            face_quality_mode=face_quality_mode,
            calibrated=calibrated,
        )
        return res.likelihood.distribution, res.best_face_similarity

    def _recognized_face_likelihood(
        self,
        anchor: FaceAnchor,
        *,
        source: str,
        face_quality_mode: FaceQualityMode,
        calibrated: bool,
    ) -> float:
        """Delegate recognized-face weighting to :class:`FaceChannel`."""
        return self._face_channel.recognized_face_likelihood(
            anchor,
            source=source,
            face_quality_mode=face_quality_mode,
            calibrated=calibrated,
        )

    @staticmethod
    def _has_recognized_face(entity: IdentityResolvableEntity, anchors: list[FaceAnchor]) -> bool:
        """Whether any recognized anchor in ``anchors`` belongs to this entity."""
        return any(
            fa.recognition_state == "recognized"
            for fa in IdentityResolver._entity_anchors(entity, anchors)
        )

    @staticmethod
    def _entity_anchors(
        entity: IdentityResolvableEntity, anchors: list[FaceAnchor]
    ) -> list[FaceAnchor]:
        """The anchors that belong to this entity."""
        entity_obs_ids = set(entity.observation_ids)
        return [
            fa
            for fa in anchors
            if fa.tracklet_id in entity_obs_ids
            or fa.tracklet_id == entity.entity_id
            or fa.detection_id in entity_obs_ids
        ]

    def _evaluate_channels_counterfactual(
        self,
        entity: IdentityResolvableEntity,
        prior: PosteriorDist,
        channels: Sequence[ChannelLikelihood],
        *,
        face_likelihood: PosteriorDist,
        reid_likelihood: PosteriorDist,
        best_face_similarity: float | None,
        captured_at: datetime,
        entity_quality: float,
        evidence_identity_ids: frozenset[str],
        check_contradiction: bool = True,
    ) -> CommitEvaluation:
        """Evaluate a channel sequence into a :class:`CommitEvaluation` for shadow comparisons."""
        posterior = combine_channels(prior, channels)
        contradicted = (
            self._contradicted(entity, posterior, face_likelihood, best_face_similarity)
            if check_contradiction
            else False
        )
        return self._evaluate_commit(
            entity,
            posterior,
            face_likelihood,
            reid_likelihood,
            captured_at,
            entity_quality,
            contradicted=contradicted,
            enable_sticky_maintenance=self._config.enable_sticky_maintenance,
            enforce_quality_gate=self._config.enable_quality_gate,
            enforce_flip_debounce=self._config.enable_flip_debounce,
            evidence_identity_ids=evidence_identity_ids,
        )

    def _counterfactual_identity(
        self,
        entity: IdentityResolvableEntity,
        face_anchors: list[FaceAnchor],
        face_evidence: list[FaceEvidence],
        prior: PosteriorDist,
        reid_likelihood: PosteriorDist,
        height_likelihood: PosteriorDist,
        captured_at: datetime,
        entity_quality: float,
        *,
        calibrated: bool,
        face_quality_mode: FaceQualityMode,
        extra_channels: Sequence[ChannelLikelihood] = (),
    ) -> str | None:
        """The identity ``_commit`` would decide from these anchors, purely for measurement.

        Same order as ``_commit``: ArcFace authority, then the channel likelihoods combined
        via :func:`combine_channels`, then commit evidence under the same rule, with its own
        contradiction. Compared with the live decision before the duplicate guard, which
        applies identically to both.
        """
        authority = self._check_arcface_authority(entity, face_anchors)
        if authority == "CONFLICT":
            return None
        if authority is not None:
            return authority
        face_res = self._face_channel.evaluate_anchors(
            entity,
            face_anchors,
            face_evidence,
            face_quality_mode=face_quality_mode,
            calibrated=calibrated,
        )
        channels: tuple[ChannelLikelihood, ...] = (
            face_res.likelihood,
            self._gallery_channel.wrap(reid_likelihood, captured_at=captured_at),
            ChannelLikelihood("height", height_likelihood, 1.0),
            *extra_channels,
        )
        evidence_ids = self._face_channel.collect_commit_evidence_ids(
            entity,
            face_anchors=face_anchors,
            face_evidence=face_evidence,
            reid_likelihood=reid_likelihood,
            calibrated_faces_only=calibrated,
        )
        for extra in extra_channels:
            if extra.counts_as_commit_evidence and extra.commit_evidence_ids:
                evidence_ids = evidence_ids | extra.commit_evidence_ids
        return self._evaluate_channels_counterfactual(
            entity,
            prior,
            channels,
            face_likelihood=face_res.likelihood.distribution,
            reid_likelihood=reid_likelihood,
            best_face_similarity=face_res.best_face_similarity,
            captured_at=captured_at,
            entity_quality=entity_quality,
            evidence_identity_ids=evidence_ids,
            check_contradiction=True,
        ).new_id

    async def _image_only_evaluation(
        self,
        entity: IdentityResolvableEntity,
        wearable_evidence: WearableEvidence,
        face_anchors: list[FaceAnchor],
        face_evidence: list[FaceEvidence],
        height_likelihood: PosteriorDist,
        captured_at: datetime,
        entity_quality: float,
        *,
        window: VoteWindow,
        calibrated: bool,
        face_quality_mode: FaceQualityMode,
        classifier_likelihood: ChannelLikelihood | None = None,
    ) -> ImageOnlyEvaluation:
        """Decide this PH without the wearable channel, for scoring against the tag.

        Control: no wearable-taught gallery row exists (what the cameras would know untaught).
        Treatment: wearable-taught rows vote too. Classifier (when active): treatment plus the
        household appearance classifier channel. In all arms no row seen in the last exclusion
        window votes, so rows minted from this stretch cannot vote for themselves. All arms see
        this round's faces and height and decide as for a PH holding no identity
        (:class:`_UnheldEntity`).
        """
        unheld = _UnheldEntity(entity)
        prior = self._build_prior(unheld, captured_at)
        cutoff = captured_at - timedelta(seconds=self._wearable_config.evaluation_exclusion_s)
        control_reid = await self._from_gallery(
            unheld,
            window,
            arm=_GalleryArm(exclude_seed_sources=_WEARABLE_TAUGHT_SOURCES, seen_before=cutoff),
        )
        treatment_reid = await self._from_gallery(
            unheld,
            window,
            arm=_GalleryArm(seen_before=cutoff),
        )
        control_id = self._counterfactual_identity(
            unheld,
            face_anchors,
            face_evidence,
            prior,
            control_reid,
            height_likelihood,
            captured_at,
            entity_quality,
            calibrated=calibrated,
            face_quality_mode=face_quality_mode,
        )
        treatment_id = self._counterfactual_identity(
            unheld,
            face_anchors,
            face_evidence,
            prior,
            treatment_reid,
            height_likelihood,
            captured_at,
            entity_quality,
            calibrated=calibrated,
            face_quality_mode=face_quality_mode,
        )
        classifier_id: str | None = None
        classifier_evaluated = classifier_likelihood is not None
        if classifier_likelihood is not None:
            classifier_id = self._counterfactual_identity(
                unheld,
                face_anchors,
                face_evidence,
                prior,
                treatment_reid,
                height_likelihood,
                captured_at,
                entity_quality,
                calibrated=calibrated,
                face_quality_mode=face_quality_mode,
                extra_channels=(classifier_likelihood,),
            )
        return ImageOnlyEvaluation(
            ph_id=entity.entity_id,
            truth_identity_id=wearable_evidence.identity_id,
            attestation_grade=wearable_evidence.attestation_grade,
            control_identity_id=control_id,
            treatment_identity_id=treatment_id,
            captured_at=captured_at,
            classifier_identity_id=classifier_id,
            classifier_evaluated=classifier_evaluated,
        )

    @staticmethod
    def _record_shadow_outcome(
        entity: IdentityResolvableEntity,
        *,
        live_identity_id: str | None,
        shadow_identity_id: str | None,
        feature: Literal["calibrated_face", "face_template"],
    ) -> None:
        """Count how a shadow decision compares with live; log each disagreement for review."""
        if shadow_identity_id == live_identity_id:
            outcome = "agrees"
        elif live_identity_id is None:
            outcome = "would_commit_new"
        elif shadow_identity_id is None:
            outcome = "would_become_unknown"
        else:
            outcome = "would_change_identity"
        counter = (
            metrics.metrics.cts_face_shadow_outcomes_total
            if feature == "calibrated_face"
            else metrics.metrics.cts_face_template_shadow_outcomes_total
        )
        counter.labels(outcome=outcome).inc()
        if outcome == "agrees":
            return
        metrics.metrics.identity_shadow_mismatch_total.labels(feature=feature).inc()
        # The rollout review reads these events to judge each disagreement against keyframes.
        event_name = (
            "calibrated_face_shadow_mismatch"
            if feature == "calibrated_face"
            else "face_template_shadow_mismatch"
        )
        logger.info(
            event_name,
            entity_id=entity.entity_id,
            outcome=outcome,
            live_identity_id=live_identity_id,
            shadow_identity_id=shadow_identity_id,
        )

    @staticmethod
    def _record_closed_world_shadow_outcome(
        entity: IdentityResolvableEntity,
        *,
        live_identity_id: str | None,
        shadow_identity_id: str | None,
    ) -> None:
        """Count closed-world shadow outcomes against live; log each disagreement for review."""
        if shadow_identity_id == live_identity_id:
            outcome = ClosedWorldShadowOutcome.AGREES
        elif live_identity_id is None and shadow_identity_id is not None:
            outcome = ClosedWorldShadowOutcome.WOULD_NAME_UNKNOWN
        else:
            outcome = ClosedWorldShadowOutcome.WOULD_CHANGE
        metrics.metrics.cts_closed_world_shadow_outcomes_total.labels(outcome=str(outcome)).inc()
        if outcome == ClosedWorldShadowOutcome.AGREES:
            return
        metrics.metrics.identity_shadow_mismatch_total.labels(feature="closed_world").inc()
        logger.info(
            "closed_world_shadow_mismatch",
            entity_id=entity.entity_id,
            outcome=str(outcome),
            live_identity_id=live_identity_id,
            shadow_identity_id=shadow_identity_id,
        )

    @staticmethod
    def _record_household_classifier_shadow_outcome(
        entity: IdentityResolvableEntity,
        *,
        live_identity_id: str | None,
        shadow_identity_id: str | None,
    ) -> None:
        """Count household classifier shadow outcomes against live; log disagreements."""
        if shadow_identity_id == live_identity_id:
            outcome = HouseholdClassifierShadowOutcome.AGREES
        elif live_identity_id is None and shadow_identity_id is not None:
            outcome = HouseholdClassifierShadowOutcome.WOULD_NAME_UNKNOWN
        else:
            outcome = HouseholdClassifierShadowOutcome.WOULD_CHANGE
        metrics.metrics.cts_household_classifier_shadow_outcomes_total.labels(
            outcome=str(outcome)
        ).inc()
        if outcome == HouseholdClassifierShadowOutcome.AGREES:
            return
        metrics.metrics.identity_shadow_mismatch_total.labels(feature="household_classifier").inc()
        logger.info(
            "household_classifier_shadow_mismatch",
            entity_id=entity.entity_id,
            outcome=str(outcome),
            live_identity_id=live_identity_id,
            shadow_identity_id=shadow_identity_id,
        )

    def _shadow_calibrated_face(
        self,
        entity: IdentityResolvableEntity,
        face_anchors: list[FaceAnchor],
        face_evidence: list[FaceEvidence],
        prior: PosteriorDist,
        reid_likelihood: PosteriorDist,
        height_likelihood: PosteriorDist,
        captured_at: datetime,
        entity_quality: float,
        *,
        live_identity_id: str | None,
        face_quality_mode: FaceQualityMode,
    ) -> None:
        """Decide this entity as the "calibrated" policy would, purely for measurement.

        ``face_anchors`` are the live anchors plus the shadow-only ones the stage emitted below
        the legacy floor.
        """
        shadow_identity_id = self._counterfactual_identity(
            entity,
            face_anchors,
            face_evidence,
            prior,
            reid_likelihood,
            height_likelihood,
            captured_at,
            entity_quality,
            calibrated=True,
            face_quality_mode=face_quality_mode,
        )
        self._record_shadow_outcome(
            entity,
            live_identity_id=live_identity_id,
            shadow_identity_id=shadow_identity_id,
            feature="calibrated_face",
        )

    def _direct_face_similarity(
        self,
        entity: IdentityResolvableEntity,
        face_anchors: list[FaceAnchor],
        face_evidence: list[FaceEvidence],
        *,
        identity_id: str | None,
    ) -> float:
        """Return the best direct recognized face match score (raw similarity) for this entity/id.

        Only grades whose governance allows duplicate-guard ranking and direct-face
        bypass qualify; external assertions and multi-frame templates are excluded.
        """
        if identity_id is None:
            return 0.0

        entity_obs_ids = set(entity.observation_ids)
        matched_ids = entity_obs_ids | {entity.entity_id}

        if face_evidence:
            return max(
                (
                    fe.match_score
                    for fe in face_evidence
                    if fe.source == "direct"
                    and fe.recognition_state == "recognized"
                    and fe.person_id == identity_id
                    and (fe.tracklet_id in matched_ids or fe.detection_id in entity_obs_ids)
                ),
                default=0.0,
            )

        return max(
            (
                fa.match_score
                for fa in face_anchors
                if fa.recognition_state == "recognized"
                and grade_of(fa).ranks_in_duplicate_guard
                and grade_of(fa).grants_duplicate_bypass
                and fa.person_id == identity_id
                and (fa.tracklet_id in matched_ids or fa.detection_id in entity_obs_ids)
            ),
            default=0.0,
        )

    def _effective_face_quality(self, anchor: FaceAnchor, mode: FaceQualityMode) -> float:
        """Delegate effective face quality calculation to :class:`FaceChannel`."""
        return self._face_channel.effective_face_quality(anchor, mode)

    def _shadow_face_quality(
        self,
        entity: IdentityResolvableEntity,
        face_anchors: list[FaceAnchor],
        face_evidence: list[FaceEvidence],
        prior: PosteriorDist,
        posterior: PosteriorDist,
        face_likelihood: PosteriorDist,
        best_face_conf: float | None,
        reid_likelihood: PosteriorDist,
        height_likelihood: PosteriorDist,
        captured_at: datetime,
        entity_quality: float,
        evidence_identity_ids: frozenset[str],
        *,
        calibrated: bool,
    ) -> None:
        """Evaluate this entity with quality-weighted face evidence, purely for measurement.

        Quality can change both the face likelihood and which recognized anchor is the best one,
        and the best anchor's match score feeds the sticky-maintenance contradiction check, so
        each side is evaluated with its own contradiction, exactly as the live commit does.
        """
        shadow_face_res = self._face_channel.evaluate_anchors(
            entity,
            face_anchors,
            face_evidence,
            face_quality_mode="enabled",
            calibrated=calibrated,
        )
        shadow_face = shadow_face_res.likelihood.distribution
        shadow_best_face_conf = shadow_face_res.best_face_similarity
        if (
            shadow_face.distribution == face_likelihood.distribution
            and shadow_best_face_conf == best_face_conf
        ):
            return
        shadow_channels = (
            shadow_face_res.likelihood,
            self._gallery_channel.wrap(reid_likelihood, captured_at=captured_at),
            ChannelLikelihood("height", height_likelihood, 1.0),
        )
        live_eval = self._evaluate_commit(
            entity,
            posterior,
            face_likelihood,
            reid_likelihood,
            captured_at,
            entity_quality,
            contradicted=self._contradicted(entity, posterior, face_likelihood, best_face_conf),
            enable_sticky_maintenance=self._config.enable_sticky_maintenance,
            enforce_quality_gate=self._config.enable_quality_gate,
            enforce_flip_debounce=self._config.enable_flip_debounce,
            evidence_identity_ids=evidence_identity_ids,
        )
        shadow_eval = self._evaluate_channels_counterfactual(
            entity,
            prior,
            shadow_channels,
            face_likelihood=shadow_face,
            reid_likelihood=reid_likelihood,
            best_face_similarity=shadow_best_face_conf,
            captured_at=captured_at,
            entity_quality=entity_quality,
            evidence_identity_ids=evidence_identity_ids,
            check_contradiction=True,
        )
        if shadow_eval.new_id != live_eval.new_id:
            # The rollout review reads these events to judge each disagreement.
            logger.info(
                "face_quality_shadow_mismatch",
                entity_id=entity.entity_id,
                live_identity_id=live_eval.new_id,
                shadow_identity_id=shadow_eval.new_id,
            )
        self._shadow_mismatch("face_quality", live_eval.new_id, shadow_eval)

    def _contradicted(
        self,
        entity: IdentityResolvableEntity,
        posterior: PosteriorDist,
        face_likelihood: PosteriorDist,
        best_face_confidence: float | None,
    ) -> bool:
        """The sticky-maintenance contradiction check, as ``_commit`` applies it."""
        (top_id, top_prob), margin = posterior.top_with_margin()
        return compute_contradiction(
            prev_id=entity.current_identity_id,
            face_likelihood=face_likelihood,
            best_face_confidence=best_face_confidence,
            top_id=top_id,
            top_prob=top_prob,
            margin=margin,
            config=self._to_commit_policy(),
        )

    def _p_face(self, confidence: float, quality: float) -> float:
        """Delegate uncalibrated face heuristic to :class:`FaceChannel`."""
        return self._face_channel.p_face(confidence, quality)

    def _frontality_factor(self, yaw_deg: float) -> float:
        """Delegate yaw frontality ramp to :class:`FaceChannel`."""
        return self._face_channel.frontality_factor(yaw_deg)

    # ------------------------------------------------------------------
    # ReID likelihood from gallery
    # ------------------------------------------------------------------

    async def _from_gallery(
        self,
        entity: IdentityResolvableEntity,
        window: VoteWindow,
        *,
        enable_coherence_boost: bool | None = None,
        enable_multiview: bool = False,
        arm: _GalleryArm | None = None,
    ) -> PosteriorDist:
        """Build likelihood from gallery k-NN search.

        Queries the gallery for similar embeddings to the entity's
        gallery entries. Maps results to per-identity scores using a
        calibrated logistic curve.

        When *enable_multiview* is True (or the live config flag is on)
        and the entity has view_prototypes, one query is issued per
        orientation bin and the MAX logistic similarity per identity is
        taken across all per-view results.  Falls back to the single
        mean-of-embeddings query when no prototypes are available.
        """
        use_multiview = enable_multiview or self._config.enable_multiview_gallery
        prototypes = entity.view_prototypes if use_multiview else ()

        if prototypes:
            return await self._from_gallery_multiview(entity, prototypes, window, arm=arm)

        # --- Single-query path (original behaviour) ---
        try:
            if arm is not None and arm.exclude_seed_sources:
                recent = await self._gallery_repo.list_gallery_entries_for_tracklets(
                    tracklet_ids=set(entity.observation_ids),
                    limit=20,
                    exclude_seed_sources=arm.exclude_seed_sources,
                )
            else:
                recent = await self._gallery_repo.list_gallery_entries_for_tracklets(
                    tracklet_ids=set(entity.observation_ids),
                    limit=20,
                )
        except Exception:
            logger.warning(
                "reid_gallery_lookup_failed",
                entity_id=entity.entity_id,
                exc_info=True,
            )
            return PosteriorDist({})

        if not recent:
            logger.debug(
                "reid_no_gallery_entries",
                entity_id=entity.entity_id,
                obs_count=len(entity.observation_ids),
            )
            return PosteriorDist({})

        # Compute mean embedding from recent gallery entries.
        embs = np.array([e.embedding for e in recent], dtype=np.float32)
        query = np.mean(embs, axis=0).tolist()

        # Embedding coherence check.
        use_coherence_boost = (
            self._config.enable_embedding_coherence_boost
            if enable_coherence_boost is None
            else enable_coherence_boost
        )
        coherence_active = False
        if use_coherence_boost and len(embs) >= 2:
            coherence_slice = embs[-self._config.embedding_coherence_window :]
            norms = np.linalg.norm(coherence_slice, axis=1, keepdims=True)
            normed = coherence_slice / np.maximum(norms, 1e-8)
            consecutive_sims = [float(normed[i] @ normed[i + 1]) for i in range(len(normed) - 1)]
            coherence_active = (
                bool(consecutive_sims)
                and min(consecutive_sims) >= self._config.embedding_coherence_min_sim
            )

        try:
            similar = await self._gallery_repo.search_similar(
                embedding=query,
                limit=20,
                **self._gallery_search_kwargs(window, arm),
            )
        except Exception:
            logger.warning(
                "reid_search_failed",
                entity_id=entity.entity_id,
                query_dim=len(query),
                exc_info=True,
            )
            return PosteriorDist({})

        metrics.metrics.reid_gallery_hits_total.labels(path="fallback").inc(len(similar))

        if not similar:
            logger.debug(
                "reid_no_similar_matches",
                entity_id=entity.entity_id,
                query_entries=len(recent),
            )
            return PosteriorDist({})

        return self._score_gallery_hits(similar, window, coherence_active)

    async def _from_gallery_multiview(
        self,
        entity: IdentityResolvableEntity,
        prototypes: tuple[ViewPrototype, ...],
        window: VoteWindow,
        *,
        arm: _GalleryArm | None = None,
    ) -> PosteriorDist:
        """Per-view gallery query with max-over-views aggregation.

        One search_similar call per orientation bin that has a qualified
        prototype.  Results are collected across all views and the MAX
        logistic similarity per identity is taken.
        """

        # Minimum prototype count to qualify for a query.
        _min_proto_count = 2
        _max_queries = 4

        # Collect per-view results.
        all_hits: list[tuple[GalleryEmbedding, float]] = []  # (entry, similarity)
        queries_run = 0

        for p in prototypes:
            if queries_run >= _max_queries:
                break
            if p.count < _min_proto_count:
                continue
            queries_run += 1

            query_emb = list(p.embedding)
            try:
                similar = await self._gallery_repo.search_similar(
                    embedding=query_emb,
                    limit=20,
                    **self._gallery_search_kwargs(window, arm),
                )
            except Exception:
                logger.warning(
                    "reid_multiview_search_failed",
                    entity_id=entity.entity_id,
                    orientation=int(p.orientation),
                    exc_info=True,
                )
                continue

            metrics.metrics.reid_gallery_hits_total.labels(path="multiview").inc(len(similar))
            all_hits.extend(similar)

        if not all_hits:
            logger.debug(
                "reid_multiview_no_matches",
                entity_id=entity.entity_id,
                queries_run=queries_run,
                prototype_count=len(prototypes),
            )
            return PosteriorDist({})

        cfg = self._gallery_scoring_config()
        scored = score_hits(
            all_hits,
            now=window.as_of,
            cfg=cfg,
            logistic=self._logistic,
            on_nonvoting_state=self._on_nonvoting_gallery_vote,
        )
        capped = cap_votes(scored)
        avg = aggregate_max_over_views(
            capped,
            identities=self._identities.keys(),
            cfg=cfg,
        )

        if not avg:
            return PosteriorDist({})

        top_reid = max(avg.items(), key=lambda x: x[1])
        logger.debug(
            "reid_multiview_top_match",
            entity_id=entity.entity_id,
            top_identity=top_reid[0],
            top_score=round(top_reid[1], 4),
            candidate_count=len(avg),
            queries_run=queries_run,
            total_hits=len(all_hits),
        )

        return PosteriorDist(avg)

    def _gallery_scoring_config(self) -> GalleryScoringConfig:
        """Build the pure scorer's config from the live ResolverConfig."""
        return GalleryScoringConfig(
            verified_trust_multiplier=self._config.gallery_verified_trust_multiplier,
            auto_verified_trust_multiplier=self._config.gallery_auto_verified_trust_multiplier,
            wearable_trust_multiplier=self._config.gallery_wearable_trust_multiplier,
            recency_half_life_days=self._config.gallery_recency_half_life_days,
            identified_entry_boost_min_sim=self._config.identified_entry_boost_min_sim,
            identified_entry_min_likelihood=self._config.identified_entry_min_likelihood,
        )

    def _gallery_search_kwargs(
        self, window: VoteWindow, arm: _GalleryArm | None = None
    ) -> _GallerySearchKwargs:
        """Kwargs every vote-path ``search_similar`` call must pass.

        Centralizes the capture-time vote window ``(seen_after, seen_before]`` and the
        voting-states filter so a future call site cannot forget either or read wall clock.
        An image-only evaluation *arm* tightens ``seen_before`` via ``min(arm.seen_before,
        window.as_of)`` and adds its provenance filter here, never at a call site.
        """
        seen_before = window.seen_before
        if arm is not None and arm.seen_before is not None:
            seen_before = min(arm.seen_before, window.as_of)
        kwargs: _GallerySearchKwargs = {
            "seen_after": window.seen_after,
            "seen_before": seen_before,
            "states": VOTING_STATES,
        }
        if arm is not None:
            kwargs["exclude_seed_sources"] = arm.exclude_seed_sources
        return kwargs

    @staticmethod
    def _on_nonvoting_gallery_vote() -> None:
        """Backstop counter: a non-operator_verified/auto_verified row reached scoring.

        Governance invariant: only operator_verified and auto_verified entries
        may vote. The gallery SQL enforces this; this counter is the
        observability backstop that pages if a non-verified vector ever
        reaches the vote (e.g. a future query regression). Keep it
        detection-only so it cannot mask a breach by silently changing scores.
        """
        metrics.metrics.reid_rejected_vector_vote_attempts_total.inc()

    def _score_gallery_hits(
        self,
        similar: list[tuple[GalleryEmbedding, float]],
        window: VoteWindow,
        coherence_active: bool = False,
    ) -> PosteriorDist:
        """Score gallery search hits into a PosteriorDist with trust-aware scoring."""
        cfg = self._gallery_scoring_config()
        scored = score_hits(
            similar,
            now=window.as_of,
            cfg=cfg,
            logistic=self._logistic,
            on_nonvoting_state=self._on_nonvoting_gallery_vote,
        )
        capped = cap_votes(scored)
        avg = aggregate_mean(capped, identities=self._identities.keys(), cfg=cfg)

        if not avg:
            return PosteriorDist({})

        coherence_boosted_identity: str | None = None
        if coherence_active and avg:
            best_key = max(avg, key=lambda k: avg[k])
            if best_key != "UNKNOWN":
                avg[best_key] = min(0.99, avg[best_key] * self._config.embedding_coherence_boost)
                coherence_boosted_identity = best_key

        top_reid = max(avg.items(), key=lambda x: x[1])
        logger.debug(
            "reid_top_match",
            entity_id="single_query",
            top_identity=top_reid[0],
            top_score=round(top_reid[1], 4),
            candidate_count=len(avg),
            gallery_entries_searched=len(similar),
            face_entry_boosted=any(hit.boosted for hit in capped),
            coherence_boosted=coherence_boosted_identity,
        )

        return PosteriorDist(avg)

    async def _has_cross_camera_reid_assist(
        self,
        entity: IdentityResolvableEntity,
        identity_id: str,
        *,
        window: VoteWindow,
    ) -> bool:
        """Return True when the committed identity's top ReID hit is cross-camera."""
        try:
            recent = await self._gallery_repo.list_gallery_entries_for_tracklets(
                tracklet_ids=set(entity.observation_ids),
                limit=20,
            )
        except Exception:
            logger.warning(
                "reid_cross_camera_assist_lookup_failed",
                entity_id=entity.entity_id,
                exc_info=True,
            )
            return False

        if not recent:
            return False

        embs = np.array([e.embedding for e in recent], dtype=np.float32)
        query = np.mean(embs, axis=0).tolist()
        try:
            similar = await self._gallery_repo.search_similar(
                embedding=query,
                limit=20,
                **self._gallery_search_kwargs(window),
            )
        except Exception:
            logger.warning(
                "reid_cross_camera_assist_search_failed",
                entity_id=entity.entity_id,
                exc_info=True,
            )
            return False

        metrics.metrics.reid_gallery_hits_total.labels(path="shadow").inc(len(similar))

        for entry, _sim in similar:
            if entry.identity_id != identity_id:
                continue
            return bool(
                entry.camera_id and entity.camera_ids and entry.camera_id not in entity.camera_ids
            )
        return False

    def _logistic(self, x: float) -> float:
        """Calibrated logistic curve for ReID similarity -> likelihood.

        Midpoint at reid_decision_sim (default 0.70). At similarity=0.9,
        returns ~0.8; at similarity=0.5, returns ~0.2.
        """
        s = self._config.reid_decision_sim
        # Simple logistic: 1 / (1 + exp(-k * (x - s)))
        k = 10.0  # steepness
        return 1.0 / (1.0 + math.exp(-k * (x - s)))

    # ------------------------------------------------------------------
    # Height likelihood
    # ------------------------------------------------------------------

    def _from_height(
        self,
        entity: IdentityResolvableEntity,
        ph_heights: dict[str, float],
        ph_height_sigmas: dict[str, float] | None = None,
    ) -> PosteriorDist:
        """Delegate height likelihood evaluation to :class:`HeightChannel`."""
        return self._height_channel.evaluate_heights(
            entity, ph_heights, ph_height_sigmas
        ).distribution

    # ------------------------------------------------------------------
    # Posterior combination
    # ------------------------------------------------------------------

    def _combine(
        self,
        prior: PosteriorDist,
        face: PosteriorDist,
        reid: PosteriorDist,
        height: PosteriorDist | None = None,
    ) -> PosteriorDist:
        """Combine prior, face likelihood, ReID likelihood, and optional height.

        Delegates to :func:`combine_channels` in canonical face -> gallery -> height
        order with multipliers from ``self._config``.
        """
        return combine_channels(
            prior,
            (
                ChannelLikelihood("face", face, self._config.face_weight_multiplier),
                ChannelLikelihood("gallery", reid, 1.0),
                ChannelLikelihood(
                    "height",
                    height if height is not None else PosteriorDist({}),
                    1.0,
                ),
            ),
        )

    # ------------------------------------------------------------------
    # Shadow comparisons
    # ------------------------------------------------------------------

    def _shadow_mismatch(
        self,
        feature: str,
        live_new_id: str | None,
        shadow_eval: CommitEvaluation,
    ) -> None:
        """Increment the shadow-mismatch metric when a shadow evaluation disagrees."""
        if live_new_id != shadow_eval.new_id:
            metrics.metrics.identity_shadow_mismatch_total.labels(feature=feature).inc()

    async def _shadow_gallery_compare(
        self,
        entity: IdentityResolvableEntity,
        prior: PosteriorDist,
        face_likelihood: PosteriorDist,
        height_likelihood: PosteriorDist,
        reid_likelihood: PosteriorDist,
        live_eval: CommitEvaluation,
        captured_at: datetime,
        entity_quality: float,
        evidence_identity_ids: frozenset[str],
        *,
        feature: str,
        vote_window: VoteWindow,
        enable_coherence_boost: bool | None = None,
        enable_multiview: bool = False,
    ) -> None:
        """Evaluate an alternative gallery-query configuration purely for measurement.

        Owns the query -> distribution-diff short-circuit -> combine -> evaluate
        -> ``_shadow_mismatch`` sequence shared by the multiview and coherence-boost
        shadows. Callers must gate invocation with an explicit sample rate — this
        method always issues the gallery query it is called for.
        """
        shadow_reid = await self._from_gallery(
            entity,
            vote_window,
            enable_coherence_boost=enable_coherence_boost,
            enable_multiview=enable_multiview,
        )
        if shadow_reid.distribution == reid_likelihood.distribution:
            return
        shadow_channels = (
            ChannelLikelihood("face", face_likelihood, self._config.face_weight_multiplier),
            self._gallery_channel.wrap(shadow_reid, captured_at=captured_at),
            ChannelLikelihood("height", height_likelihood, self._config.height_weight_multiplier),
        )
        shadow_eval = self._evaluate_channels_counterfactual(
            entity,
            prior,
            shadow_channels,
            face_likelihood=face_likelihood,
            reid_likelihood=shadow_reid,
            best_face_similarity=None,
            captured_at=captured_at,
            entity_quality=entity_quality,
            evidence_identity_ids=evidence_identity_ids,
            check_contradiction=False,
        )
        self._shadow_mismatch(feature, live_eval.new_id, shadow_eval)

    def _shadow_wearable(
        self,
        *,
        entity: IdentityResolvableEntity,
        wearable_evidence: WearableEvidence,
        prior: PosteriorDist,
        base_channels: Sequence[ChannelLikelihood],
        face_likelihood: PosteriorDist,
        reid_likelihood: PosteriorDist,
        best_face_conf: float | None,
        captured_at: datetime,
        entity_quality: float,
        commit_evidence_ids: frozenset[str],
        arcface_authority: str | None,
        live_identity_id: str | None,
    ) -> None:
        """Evaluate wearable counterfactual in shadow mode without changing live decisions."""
        if (
            arcface_authority is not None
            and arcface_authority != "CONFLICT"
            and wearable_evidence.identity_id != arcface_authority
        ):
            metrics.metrics.cts_wearable_shadow_outcomes_total.labels(
                outcome=str(WearableShadowOutcome.WOULD_CONFLICT)
            ).inc()
            return

        wearable_ch, _ = self._wearable_channel.evaluate_wearable(entity, wearable_evidence)
        shadow_posterior = combine_channels(prior, (*base_channels, wearable_ch))
        shadow_evidence_ids = commit_evidence_ids | frozenset({wearable_evidence.identity_id})
        if arcface_authority == "CONFLICT":
            shadow_id: str | None = None
        elif arcface_authority is not None:
            shadow_id = arcface_authority
        else:
            contradicted = self._contradicted(
                entity, shadow_posterior, face_likelihood, best_face_conf
            )
            shadow_eval = self._evaluate_commit(
                entity,
                shadow_posterior,
                face_likelihood,
                reid_likelihood,
                captured_at,
                entity_quality,
                contradicted=contradicted,
                enable_sticky_maintenance=self._config.enable_sticky_maintenance,
                enforce_quality_gate=self._config.enable_quality_gate,
                enforce_flip_debounce=self._config.enable_flip_debounce,
                evidence_identity_ids=shadow_evidence_ids,
            )
            shadow_id = shadow_eval.new_id

        if shadow_id == live_identity_id:
            outcome = WearableShadowOutcome.AGREES
        elif live_identity_id is None and shadow_id is not None:
            outcome = WearableShadowOutcome.WOULD_NAME_UNKNOWN
        else:
            outcome = WearableShadowOutcome.WOULD_CHANGE_IDENTITY
        metrics.metrics.cts_wearable_shadow_outcomes_total.labels(outcome=str(outcome)).inc()

    # ------------------------------------------------------------------
    # Commit rule
    # ------------------------------------------------------------------

    def _commit(
        self,
        entity: IdentityResolvableEntity,
        posterior: PosteriorDist,
        face_likelihood: PosteriorDist,
        reid_likelihood: PosteriorDist,
        captured_at: datetime,
        best_face_confidence: float | None = None,
        entity_quality: float = 1.0,
        *,
        arcface_authority: str | None = None,
        evidence_identity_ids: frozenset[str] = frozenset(),
        wearable_evidence: WearableEvidence | None = None,
    ) -> IdentityDecision:
        """Apply the commit rule to produce an identity decision.

        ``arcface_authority`` is pre-computed by the caller from the entity's
        direct face anchors:
        - None   → no qualifying authoritative anchor
        - "CONFLICT" → two qualifying anchors for different identities → Unknown
        - <id>   → one qualifying anchor; bypass quality gate and debounce

        When no authoritative anchor exists (default in production until M10
        supplies calibrated_confidence), the normal Bayesian posterior path runs.

        The ArcFace-authority pre-emption below sets ``evidence_backed=True``
        unconditionally — correct by construction, because
        ``_check_arcface_authority`` already requires a calibrated, recognized,
        entity-matched anchor for that exact identity, which *is* identity-matched
        evidence. It does not consult ``evidence_identity_ids``.
        """
        (top_id, top_prob), margin = posterior.top_with_margin()

        # --- ArcFace authority pre-emption  ---
        if arcface_authority == "CONFLICT":
            prev_id = entity.current_identity_id
            logger.info(
                "arcface_authority_conflict",
                entity_id=entity.entity_id,
                prev_id=prev_id,
            )
            metrics.metrics.identity_commits_total.labels(source="arcface_conflict").inc()

            return IdentityDecision(
                ph_id=entity.entity_id,
                identity_id=None,
                posterior=posterior,
                revises_previous=(prev_id is not None),
                previous_identity_id=prev_id,
                reason="arcface_authority_conflict: two qualifying anchors for different identities",  # noqa: E501
                evidence_backed=False,
                decision_id=str(uuid.uuid4()),
                inferred_identity_id="",
                effective_identity_id="",
                authority=str(IdentityAuthority.NONE),
                decision_source="arcface_authority_conflict",
                conflict="AUTHORITY_CONFLICT",
            )
        if (
            arcface_authority is not None
            and wearable_evidence is not None
            and wearable_evidence.identity_id != arcface_authority
        ):
            prev_id = entity.current_identity_id
            logger.info(
                "wearable_face_conflict",
                entity_id=entity.entity_id,
                tag_id=wearable_evidence.tag_id,
            )
            metrics.metrics.cts_wearable_face_conflict_total.inc()
            return IdentityDecision(
                ph_id=entity.entity_id,
                identity_id=None,
                posterior=posterior,
                revises_previous=(prev_id is not None),
                previous_identity_id=prev_id,
                reason=("wearable_face_conflict: calibrated direct face disagreed with wearable"),
                evidence_backed=False,
                decision_id=str(uuid.uuid4()),
                inferred_identity_id="",
                effective_identity_id="",
                authority=str(IdentityAuthority.NONE),
                decision_source="wearable_face_conflict",
                conflict="wearable_face",
            )
        if arcface_authority is not None:
            prev_id = entity.current_identity_id
            revises = arcface_authority != prev_id
            reason = f"arcface_authority: {arcface_authority}" if revises else ""
            logger.debug(
                "arcface_authority_commit",
                entity_id=entity.entity_id,
                identity_id=arcface_authority,
                revises=revises,
            )
            metrics.metrics.identity_commits_total.labels(source="arcface_authority").inc()

            return IdentityDecision(
                ph_id=entity.entity_id,
                identity_id=arcface_authority,
                posterior=posterior,
                revises_previous=revises,
                previous_identity_id=prev_id,
                reason=reason,
                evidence_backed=True,
                decision_id=str(uuid.uuid4()),
                inferred_identity_id=top_id if top_id != "UNKNOWN" else "",
                effective_identity_id=arcface_authority,
                authority=str(IdentityAuthority.DIRECT_FACE),
                decision_source="arcface_authority",
            )

        # --- Normal Bayesian posterior path ---
        prev_id = entity.current_identity_id
        contradicted = self._contradicted(entity, posterior, face_likelihood, best_face_confidence)

        live_eval = self._evaluate_commit(
            entity,
            posterior,
            face_likelihood,
            reid_likelihood,
            captured_at,
            entity_quality,
            contradicted=contradicted,
            enable_sticky_maintenance=self._config.enable_sticky_maintenance,
            enforce_quality_gate=self._config.enable_quality_gate,
            enforce_flip_debounce=self._config.enable_flip_debounce,
            evidence_identity_ids=evidence_identity_ids,
        )

        new_id = live_eval.new_id
        if live_eval.quality_gate_blocked:
            metrics.metrics.identity_quality_gate_blocks_total.inc()

        # Sticky maintenance shadow — compute what sticky would decide.
        if not self._config.enable_sticky_maintenance and prev_id is not None:
            sticky_shadow_eval = self._evaluate_commit(
                entity,
                posterior,
                face_likelihood,
                reid_likelihood,
                captured_at,
                entity_quality,
                contradicted=contradicted,
                enable_sticky_maintenance=True,
                enforce_quality_gate=self._config.enable_quality_gate,
                enforce_flip_debounce=self._config.enable_flip_debounce,
                evidence_identity_ids=evidence_identity_ids,
            )
            self._shadow_mismatch("sticky_maintenance", live_eval.new_id, sticky_shadow_eval)

        if not self._config.enable_flip_debounce:
            debounced_eval = self._evaluate_commit(
                entity,
                posterior,
                face_likelihood,
                reid_likelihood,
                captured_at,
                entity_quality,
                contradicted=contradicted,
                enable_sticky_maintenance=self._config.enable_sticky_maintenance,
                enforce_quality_gate=self._config.enable_quality_gate,
                enforce_flip_debounce=True,
                evidence_identity_ids=evidence_identity_ids,
            )
            self._shadow_mismatch("flip_debounce", live_eval.new_id, debounced_eval)

        if (
            prev_id is not None
            and not live_eval.within_maintenance_window
            and not live_eval.has_evidence
        ):
            metrics.metrics.identity_decays_total.inc()
            evidence_age_s: float
            ev_ts = entity.last_independent_identity_evidence_at
            if ev_ts is not None:
                evidence_age_s = (captured_at - ev_ts).total_seconds()
            else:
                evidence_age_s = (captured_at - entity.last_seen_at).total_seconds()
            logger.info(
                "identity_maintenance_window_expired",
                entity_id=entity.entity_id,
                prev_identity_id=prev_id,
                evidence_age_s=round(evidence_age_s, 1),
                max_age_s=self._config.prior_maintenance_max_age_s,
            )

        revises = new_id != prev_id

        reason = ""
        if revises:
            if prev_id is None:
                reason = f"initial_assignment: {top_id} (p={top_prob:.3f})"
            elif new_id is None:
                reason = f"demoted_to_unknown: {top_id} (p={top_prob:.3f}, margin={margin:.3f})"
            else:
                reason = (
                    f"identity_change: {prev_id} -> {new_id} "
                    f"(p={top_prob:.3f}, margin={margin:.3f})"
                )

        # Label prior-held decisions as temporal_prior, not face/reid. Computed
        # once and shared by the commit metric below and decision_source on the
        # returned IdentityDecision (previously duplicated in both places).
        if not live_eval.has_evidence:
            commit_source = "temporal_prior"
        elif top_id in face_likelihood.distribution:
            commit_source = "face"
        elif wearable_evidence is not None and wearable_evidence.identity_id == top_id:
            commit_source = "wearable"
        else:
            commit_source = "reid"

        metrics.metrics.posterior_entropy.observe(posterior.entropy())
        if prev_id is not None and new_id is not None and new_id != prev_id:
            metrics.metrics.identity_flips_total.inc()
        if new_id is not None and revises:
            metrics.metrics.identity_commits_total.labels(source=commit_source).inc()

        if new_id is None:
            if prev_id is not None:
                metrics.metrics.identity_unknown_after_known_total.inc()
            logger.debug(
                "identity_not_committed",
                entity_id=entity.entity_id,
                top_id=top_id,
                top_prob=round(top_prob, 4),
                margin=round(margin, 4),
                has_evidence=live_eval.has_evidence,
                within_maintenance_window=live_eval.within_maintenance_window,
                prev_id=prev_id,
                known_identity_count=len(self._identities),
                quality_gate_blocked=live_eval.quality_gate_blocked,
                flip_debounce_blocked=live_eval.flip_debounce_blocked,
            )
        elif live_eval.within_maintenance_window:
            logger.debug(
                "identity_maintained_by_prior",
                entity_id=entity.entity_id,
                identity_id=new_id,
                top_prob=round(top_prob, 4),
                age_s=round((captured_at - entity.last_seen_at).total_seconds(), 1),
            )

        ev_ts = entity.last_independent_identity_evidence_at
        ev_ts_ns = int(ev_ts.timestamp() * 1e9) if ev_ts else 0

        # Authority ladder rung for this commit (F9/): distinct from
        # commit_source, which states *which evidence led* the posterior.
        if new_id is None:
            authority = IdentityAuthority.NONE
        elif live_eval.within_maintenance_window and not live_eval.evidence_backed:
            authority = IdentityAuthority.TEMPORAL_PRIOR
        else:
            authority = IdentityAuthority.POSTERIOR

        return IdentityDecision(
            ph_id=entity.entity_id,
            identity_id=new_id,
            posterior=posterior,
            revises_previous=revises,
            previous_identity_id=prev_id,
            reason=reason,
            evidence_backed=live_eval.evidence_backed,
            decision_id=str(uuid.uuid4()),
            inferred_identity_id=top_id if top_id != "UNKNOWN" else "",
            effective_identity_id=new_id or "",
            authority=str(authority),
            decision_source=commit_source,
            last_independent_evidence_at_unix_ns=ev_ts_ns,
        )

    def _to_commit_policy(self) -> _CommitPolicy:
        """Build a ``CommitPolicy`` from this resolver's config."""
        c = self._config
        return _CommitPolicy(
            commit_prob=c.commit_prob,
            commit_margin=c.commit_margin,
            commit_prob_dense=c.commit_prob_dense,
            commit_margin_dense=c.commit_margin_dense,
            prior_maintenance_max_age_s=c.prior_maintenance_max_age_s,
            arcface_authority_calibrated_confidence=c.arcface_authority_calibrated_confidence,
            face_commit_min_similarity=c.face_commit_min_similarity,
            min_quality_to_commit=c.min_quality_to_commit,
            enable_quality_gate=c.enable_quality_gate,
            flip_debounce_window_s=c.flip_debounce_window_s,
            enable_flip_debounce=c.enable_flip_debounce,
            contradiction_face_similarity=c.contradiction_face_similarity,
            contradiction_posterior_prob=c.contradiction_posterior_prob,
            contradiction_posterior_margin=c.contradiction_posterior_margin,
            enable_sticky_maintenance=c.enable_sticky_maintenance,
        )

    def _evaluate_commit(
        self,
        entity: IdentityResolvableEntity,
        posterior: PosteriorDist,
        face_likelihood: PosteriorDist,
        reid_likelihood: PosteriorDist,
        captured_at: datetime,
        entity_quality: float,
        *,
        contradicted: bool = False,
        enable_sticky_maintenance: bool = False,
        enforce_quality_gate: bool,
        enforce_flip_debounce: bool,
        evidence_identity_ids: frozenset[str] = frozenset(),
    ) -> _CommitEvaluation:
        """Evaluate the commit rule — thin delegator to the canonical pure function."""
        return _evaluate_commit_pure(
            entity=entity,
            posterior=posterior,
            face_likelihood=face_likelihood,
            reid_likelihood=reid_likelihood,
            captured_at=captured_at,
            entity_quality=entity_quality,
            config=self._to_commit_policy(),
            contradicted=contradicted,
            enable_sticky_maintenance=enable_sticky_maintenance,
            enforce_quality_gate=enforce_quality_gate,
            enforce_flip_debounce=enforce_flip_debounce,
            evidence_identity_ids=evidence_identity_ids,
        )

    # ------------------------------------------------------------------
    # Retroactive revision
    # ------------------------------------------------------------------

    def _build_revision(
        self,
        entity: IdentityResolvableEntity,
        decision: IdentityDecision,
        captured_at: datetime,
    ) -> IdentityRevision | None:
        """Build an IdentityRevision when identity changes.

        The revision covers all observations in the entity that were
        active within the revision horizon.

        Rate-limited: max_revisions_per_ph_per_minute.
        """
        # Rate limiting.
        now = captured_at
        window_start = now.timestamp() - 60.0
        log_key = entity.entity_id
        recent = [
            ts for ts in self._revision_log.get(log_key, []) if ts.timestamp() >= window_start
        ]
        if len(recent) >= self._config.max_revisions_per_ph_per_minute:
            logger.warning(
                "Revision rate limit exceeded",
                entity_id=log_key,
                recent_count=len(recent),
            )
            return None

        # Prune stale entries to prevent unbounded memory growth.
        self._revision_log[log_key] = [
            ts for ts in self._revision_log[log_key] if ts.timestamp() >= window_start
        ]

        # Collect observation IDs within the revision horizon.
        observation_ids = list(entity.observation_ids)

        if not observation_ids:
            return None

        revision = IdentityRevision(
            revision_id=str(uuid.uuid4()),
            ph_id=entity.entity_id,  # entity_id is a PH id
            previous_identity_id=decision.previous_identity_id,
            new_identity_id=decision.identity_id,
            actor="resolver",
            reason=decision.reason,
            applied_at=now,
            rewritten_rows=len(observation_ids),
            evidence=None,
        )

        self._revision_log[entity.entity_id].append(now)
        return revision

    # ------------------------------------------------------------------
    # Evidence ledger
    # ------------------------------------------------------------------

    def _build_evidence_ledger(
        self,
        entity: IdentityResolvableEntity,
        face_likelihood: PosteriorDist,
        reid_likelihood: PosteriorDist,
        face_evidence: list[FaceEvidence],
        best_face_conf: float | None,
        captured_at: datetime,
        *,
        wearable_evidence: WearableEvidence | None = None,
        wearable_probability: float = 0.0,
    ) -> list[IdentityEvidence]:
        """Build the evidence ledger for one entity in one frame."""
        items: list[IdentityEvidence] = []

        # Face evidence.
        entity_obs_ids_set = set(entity.observation_ids)
        for fe in face_evidence:
            if fe.tracklet_id and (
                fe.tracklet_id in entity_obs_ids_set or fe.tracklet_id == entity.entity_id
            ):
                if fe.source == "direct":
                    items.append(
                        IdentityEvidence.direct_face(
                            identity_id=fe.person_id,
                            confidence=fe.match_score,
                            tracklet_id=fe.tracklet_id,
                            captured_at=fe.captured_at,
                            quality=fe.quality,
                        )
                    )
                elif fe.source == "template":
                    items.append(
                        IdentityEvidence.face_template(
                            identity_id=fe.person_id,
                            confidence=fe.match_score,
                            tracklet_id=fe.tracklet_id,
                            captured_at=fe.captured_at,
                            calibrated_confidence=fe.calibrated_confidence,
                        )
                    )
                elif fe.source == "cc_assertion":
                    # Distinct source  so the replay
                    # evaluator and reid-disagreement metrics can segment
                    # external evidence from native association hints.
                    items.append(
                        IdentityEvidence.cc_assertion(
                            identity_id=fe.person_id,
                            confidence=fe.match_score,
                            tracklet_id=fe.tracklet_id,
                            captured_at=fe.captured_at,
                            quality=fe.quality,
                        )
                    )
                else:
                    items.append(
                        IdentityEvidence.association_hint(
                            identity_id=fe.person_id,
                            confidence=fe.match_score,
                            tracklet_id=fe.tracklet_id,
                            captured_at=fe.captured_at,
                        )
                    )

        # ReID evidence (top match from gallery).
        if reid_likelihood.distribution:
            top_reid, top_score = reid_likelihood.top_identity()
            if top_reid != "UNKNOWN" and top_score > 0.3:
                items.append(
                    IdentityEvidence.reid(
                        identity_id=top_reid,
                        confidence=top_score,
                    )
                )

        # Wearable evidence.
        if wearable_evidence is not None:
            items.append(
                IdentityEvidence.wearable(
                    identity_id=wearable_evidence.identity_id,
                    confidence=wearable_probability,
                    tag_id=wearable_evidence.tag_id,
                    captured_at=wearable_evidence.newest_fix_at,
                    evidence_class=str(wearable_evidence.evidence_class),
                    attestation_grade=str(wearable_evidence.attestation_grade),
                )
            )

        # Temporal prior.
        if entity.current_identity_id:
            items.append(
                IdentityEvidence.temporal_prior(
                    identity_id=entity.current_identity_id,
                    confidence=0.6,
                )
            )

        return items

    # ------------------------------------------------------------------
    # Face evidence augmentation
    # ------------------------------------------------------------------

    def _augment_face_evidence(
        self,
        hypotheses: Sequence[IdentityResolvableEntity],
        face_evidence: list[FaceEvidence],
        augmented_anchors: list[FaceAnchor],
    ) -> list[FaceEvidence]:
        """Build FaceEvidence records for all face anchors.

        Direct evidence is matched from ``face_evidence``.  Anchors without
        a pre-existing record are propagated (synthetic) and receive reduced
        weight in the Bayesian combiner.

        When ``face_evidence`` is empty (backward-compat), returns empty
        so all anchors are treated as direct.
        """
        if not face_evidence:
            return []

        # Template evidence is matched to its template by detection id and kept out of the tracklet
        # map, so it never stands in for a PH's frame evidence.
        ev_by_tracklet: dict[str, FaceEvidence] = {
            fe.tracklet_id: fe for fe in face_evidence if fe.tracklet_id and fe.source != "template"
        }
        template_ev: dict[str, FaceEvidence] = {
            fe.detection_id: fe for fe in face_evidence if fe.source == "template"
        }

        result: list[FaceEvidence] = []
        for fa in augmented_anchors:
            if not grade_of(fa).propagates:
                if fa.detection_id in template_ev:
                    result.append(template_ev[fa.detection_id])
                continue
            if fa.tracklet_id in ev_by_tracklet:
                result.append(ev_by_tracklet[fa.tracklet_id])
            else:
                result.append(
                    FaceEvidence(
                        person_id=fa.person_id,
                        match_score=fa.match_score,
                        tracklet_id=fa.tracklet_id,
                        camera_id=fa.camera_id,
                        frame_index=0,  # propagated anchors have no frame index
                        source="propagated",
                        quality=fa.quality,
                        captured_at=fa.captured_at,
                    )
                )
        return result

    # ------------------------------------------------------------------
    # Cross-GT face propagation
    # ------------------------------------------------------------------

    async def _propagate_face_anchors(
        self,
        hypotheses: Sequence[IdentityResolvableEntity],
        face_anchors: list[FaceAnchor],
    ) -> list[FaceAnchor]:
        """Propagate face anchors from face-evidenced entities to similar adjacent ones.

        When Camera A gets a face match for Alice but Camera B (same room,
        overlapping FOV) has a separate entity for the same person, this
        method creates synthetic FaceAnchors for Camera B's entity so that the
        Bayesian resolver can commit Alice's identity there too.

        Confidence of the synthetic anchor is scaled by the gallery cosine
        similarity between the two entities.  Only entities with no direct
        face evidence this frame are candidates for propagation.
        """
        if not face_anchors or len(hypotheses) < 2:
            return face_anchors

        # Map each observation to its entity for fast lookup.
        entity_by_obs: dict[str, IdentityResolvableEntity] = {}
        for entity in hypotheses:
            for oid in entity.observation_ids:
                entity_by_obs[oid] = entity
        # Secondary lookup by entity_id so PH-mode anchors (tracklet_id=ph_id)
        # are found without adding ph_id to observation_ids.
        entity_by_id: dict[str, IdentityResolvableEntity] = {e.entity_id: e for e in hypotheses}

        # Find which entities have direct face evidence and pick the best anchor per entity.
        # only recognized anchors can propagate — candidate and unrecognized
        # evidence is too weak to justify cross-camera identity transfer.
        evidenced_entity_ids: set[str] = set()
        best_anchor_by_entity: dict[str, FaceAnchor] = {}
        for fa in face_anchors:
            # Grades with propagates=False (such as multi-frame templates) neither
            # propagate nor mark their PH as face-evidenced for propagation.
            if fa.recognition_state != "recognized" or not grade_of(fa).propagates:
                continue
            src_entity = entity_by_obs.get(fa.tracklet_id) or entity_by_id.get(fa.tracklet_id)
            if src_entity is None:
                continue
            evidenced_entity_ids.add(src_entity.entity_id)
            existing = best_anchor_by_entity.get(src_entity.entity_id)
            if existing is None or (
                fa.match_score * fa.quality > existing.match_score * existing.quality
            ):
                best_anchor_by_entity[src_entity.entity_id] = fa

        if not evidenced_entity_ids:
            return face_anchors

        # Candidate entities for propagation: no direct face evidence this frame.
        unevidenced = [e for e in hypotheses if e.entity_id not in evidenced_entity_ids]
        if not unevidenced:
            return face_anchors

        threshold = self._config.cross_gt_face_propagation_threshold
        max_props = self._config.cross_gt_face_propagation_max_gts
        synthetic: list[FaceAnchor] = []
        propagated_count = 0

        for src_entity_id, src_anchor in best_anchor_by_entity.items():
            if propagated_count >= max_props:
                break
            src_entity = next((e for e in hypotheses if e.entity_id == src_entity_id), None)
            if src_entity is None or not src_entity.observation_ids:
                continue

            for dst_entity in unevidenced:
                if propagated_count >= max_props:
                    break
                if not dst_entity.observation_ids:
                    continue

                sim = await self._gallery_similarity(
                    set(src_entity.observation_ids), set(dst_entity.observation_ids)
                )
                if sim < threshold:
                    logger.debug(
                        "face_propagation_skipped_low_similarity",
                        src_entity=src_entity_id,
                        dst_entity=dst_entity.entity_id,
                        similarity=round(sim, 4),
                        threshold=threshold,
                    )
                    continue

                syn_match_score = src_anchor.match_score * sim
                if syn_match_score < self._config.face_commit_min_similarity:
                    continue
                # dataclasses.replace (not hand-copied fields, per
                # engineering-standards F1/F10): a synthetic propagated anchor
                # carries origin unchanged  and every
                # other src_anchor field forward automatically, so a new
                # FaceAnchor field is never silently dropped here. Explicitly
                # reset calibrated_confidence to None: propagated evidence
                # describes a different detection than the source anchor and
                # must never carry the source's calibration into
                # _check_arcface_authority for the destination entity; for the
                # same reason it carries no face detection score.
                syn = replace(
                    src_anchor,
                    match_score=syn_match_score,
                    det_score=0.0,
                    quality=src_anchor.quality * 0.8,
                    tracklet_id=dst_entity.observation_ids[0],
                    detection_id="",
                    camera_id=dst_entity.camera_ids[0] if dst_entity.camera_ids else "",
                    calibrated_confidence=None,
                )
                synthetic.append(syn)
                propagated_count += 1
                metrics.metrics.face_propagations_total.inc()

                logger.info(
                    "face_anchor_propagated",
                    src_entity=src_entity_id,
                    dst_entity=dst_entity.entity_id,
                    person_id=src_anchor.person_id,
                    original_confidence=round(src_anchor.match_score, 3),
                    propagated_confidence=round(syn.match_score, 3),
                    gallery_sim=round(sim, 4),
                )

        if synthetic:
            return list(face_anchors) + synthetic
        return face_anchors

    async def _gallery_similarity(self, tids_a: set[str], tids_b: set[str]) -> float:
        """Return gallery cosine similarity, routed through cache when available."""
        if self._gallery_cache is not None:
            return await self._gallery_cache.gallery_similarity(tids_a, tids_b)
        return await self._gallery_repo.gallery_similarity(tids_a, tids_b)

    def register_identity(self, identity: Identity) -> None:
        """Register a known identity for display name lookup."""
        self._identities[identity.identity_id] = identity


# ---------------------------------------------------------------------------
# Contradiction helper (pure, outside the class)
# ---------------------------------------------------------------------------


# _compute_contradiction has been moved to identity.commit_policy.compute_contradiction.
# The resolver calls compute_contradiction (imported at the top of this module).

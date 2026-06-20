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
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
from structlog import get_logger

if TYPE_CHECKING:
    from ..pipeline.gallery_cache import GalleryCache

from ..domain import (
    FaceAnchor,
    GalleryEmbedding,
    Identity,
    IdentityDecision,
    IdentityResolvableEntity,
    IdentityRevision,
    PosteriorDist,
    ResolveOutcome,
    ViewPrototype,
)
from ..inference.evidence import FaceEvidence
from ..observability import metrics
from ..storage.base import GalleryRepository
from .identity.commit_policy import (
    CommitEvaluation,
    FaceLock,
    compute_contradiction,
)
from .identity.commit_policy import (
    evaluate_commit as _evaluate_commit_pure,
)
from .identity.evidence import IdentityEvidence
from .identity.policy import CommitPolicy as _CommitPolicy
from .identity.posterior import EvidencePosterior, combine_posteriors

logger = get_logger(__name__)


# _FaceLock and _CommitEvaluation are now public types in identity.commit_policy.
# Aliases keep internal references consistent.
_FaceLock = FaceLock
_CommitEvaluation = CommitEvaluation


@dataclass(frozen=True)
class _PendingIdentityDecision:
    """Identity decision plus batch-scoped evidence used by guardrails."""

    entity: IdentityResolvableEntity
    decision: IdentityDecision
    direct_face_confidence: float


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

    # Multiplier applied to height likelihood weights before combining.
    # Height is a weak-but-useful demographic signal.  Default 1.5 means
    # height evidence carries 1.5x the base weight, which is less than
    # face (3.0) but more than neutral (1.0).
    height_weight_multiplier: float = 1.5

    # --- Evidence quality gates ---
    # Minimum PH crop-quality EMA needed for a new identity commit.
    # CropQuality maxes out at 0.60 without pose (0.30 area + 0.30 detector
    # confidence) and 0.80 with pose, so 0.35 blocks clearly poor crops while
    # allowing ordinary no-pose detections to commit.
    min_quality_to_commit: float = 0.35

    # Minimum PH crop-quality EMA needed to set or refresh a face lock.
    # Face locks last longer than ordinary prior maintenance, so they require a
    # cleaner crop than the baseline commit threshold while staying reachable
    # for no-pose frames whose practical maximum is 0.60.
    min_quality_to_face_lock: float = 0.45

    # Shadow flag for quality gating.  When false, low-quality commits and face
    # locks are allowed but counted in cts_identity_quality_gate_blocks_total.
    enable_quality_gate: bool = False

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

    # Batch-level guardrail: prevent two active PHs in the same resolver call
    # from acquiring the same resident identity unless each duplicate has strong
    # direct ArcFace evidence for that identity.  Ships shadow-first.
    enable_duplicate_active_identity_guard: bool = False

    # Minimum direct face confidence required to bypass the duplicate-active
    # identity guard.  This is stricter than face_commit_min_confidence because
    # the invariant is about two simultaneously visible PHs sharing one person.
    duplicate_identity_direct_face_min_confidence: float = 0.90

    # Sticky maintenance — hold a committed identity within the maintenance
    # window unless strongly contradicted, even when the posterior argmax is
    # UNKNOWN or a weakly-supported other identity.  Encodes the product-owner
    # choice to "favor continuity and stability" under weak evidence.
    enable_sticky_maintenance: bool = False

    # A different identity's face anchor at or above this confidence
    # contradicts the held identity during sticky maintenance.
    contradiction_face_confidence: float = 0.70

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

    # Maximum age (seconds) of the most recent evidence-backed commit
    # before the prior alone is no longer sufficient to maintain an
    # identity.  Set longer than the face-id cooldown (default 5 s) so
    # identity survives gaps between face-id calls, but short enough
    # that a person who left and returned hours later still requires
    # fresh evidence.
    prior_maintenance_max_age_s: float = 120.0

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
    # _combine() smoothing collapses to posterior≈0.35 — below
    # commit_prob.  The floor ensures the posterior crosses commit_prob
    # even when only alice's entries appear in the top-k results.
    identified_entry_min_likelihood: float = 0.80

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

    # --- Face lock ---
    # Minimum face anchor confidence to set a face lock on a GlobalTrack.
    # Face locks extend the maintenance window to face_lock_maintenance_max_age_s
    # so that a face-confirmed identity persists across frames where face-id is
    # on cooldown or the person turns away.
    face_commit_min_confidence: float = 0.70

    # How long (seconds) a face-locked identity is maintained without fresh face
    # evidence.  Much longer than prior_maintenance_max_age_s because a face ID
    # commit represents a high-confidence, hardware-verified identification that
    # should persist until a contradicting face match arrives.
    face_lock_maintenance_max_age_s: float = 300

    # --- Cross-GT face propagation ---
    # Minimum gallery cosine similarity for propagating a face-confirmed identity
    # to an adjacent GlobalTrack that has no face anchor of its own. Propagation
    # also requires src_face_confidence * gallery_similarity to clear
    # face_commit_min_confidence, so 0.72 rejects weak ReID similarity while
    # preserving high-confidence doorway handoffs.
    cross_gt_face_propagation_threshold: float = 0.72

    # Maximum number of adjacent GlobalTracks to propagate face identity to per
    # resolve() call.  Caps the gallery query overhead for busy scenes.
    cross_gt_face_propagation_max_gts: int = 4

    # --- Multi-view gallery query ---
    # When enabled, _from_gallery builds per-orientation queries from the PH's
    # view_prototypes and takes the max-over-views logistic similarity per
    # identity.  Ships shadow-first (off by default).
    enable_multiview_gallery: bool = False

    # Minimum orientation confidence for seeding a gallery entry from a
    # face-recognized PH's non-frontal observation.
    seed_orientation_min_confidence: float = 0.5


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
            captured_at=datetime.now(UTC),
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
    ) -> None:
        self._gallery_repo = gallery_repo
        self._config = config or ResolverConfig()
        self._gallery_cache = gallery_cache
        # Known identities for display names
        self._identities: dict[str, Identity] = {
            ident.identity_id: ident for ident in identities or []
        }
        # Timestamps for the TTL-based identity-list cache.
        self._identities_loaded_at: datetime | None = (
            None  # None forces an immediate load on first resolve()
        )
        # Revision rate limiter: global_track_id -> list of revision timestamps
        self._revision_log: dict[str, list[datetime]] = defaultdict(list)
        # Face locks: global_track_id -> _FaceLock tracking the strongest committed face identity.
        self._face_locks: dict[str, _FaceLock] = {}

    async def resolve(
        self,
        hypotheses: Sequence[IdentityResolvableEntity],
        new_face_anchors: list[FaceAnchor],
        captured_at: datetime,
        ph_heights: dict[str, float] | None = None,
        ph_qualities: dict[str, float] | None = None,
        face_evidence: list[FaceEvidence] | None = None,
        open_ph_identities: Mapping[str, str] | None = None,
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
            self._identities_loaded_at = now_dt

        # Propagate face anchors from face-evidenced entities to similar adjacent
        # entities that share the same physical space but weren't merged by the
        # cross-camera associator.
        augmented_anchors = await self._propagate_face_anchors(hypotheses, new_face_anchors)
        augmented_evidence = self._augment_face_evidence(
            hypotheses, face_evidence or [], augmented_anchors
        )

        pending_decisions: list[_PendingIdentityDecision] = []

        for entity in hypotheses:
            prior = self._build_prior(entity, captured_at)
            face_likelihood, best_face_conf = self._from_face_anchors(
                entity, augmented_anchors, augmented_evidence
            )
            reid_likelihood = await self._from_gallery(entity)
            height_likelihood = self._from_height(entity, ph_heights or {})
            posterior = self._combine(prior, face_likelihood, reid_likelihood, height_likelihood)
            entity_quality = (ph_qualities or {}).get(entity.entity_id, 1.0)

            # shadow multiview gallery query.
            if not self._config.enable_multiview_gallery and entity.view_prototypes:
                multiview_reid = await self._from_gallery(entity, enable_multiview=True)
                if multiview_reid.distribution != reid_likelihood.distribution:
                    multiview_posterior = self._combine(
                        prior, face_likelihood, multiview_reid, height_likelihood
                    )
                    live_eval_mv = self._evaluate_commit(
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
                    )
                    mv_eval = self._evaluate_commit(
                        entity,
                        multiview_posterior,
                        face_likelihood,
                        multiview_reid,
                        captured_at,
                        entity_quality,
                        contradicted=False,
                        enable_sticky_maintenance=self._config.enable_sticky_maintenance,
                        enforce_quality_gate=self._config.enable_quality_gate,
                        enforce_flip_debounce=self._config.enable_flip_debounce,
                    )
                    if live_eval_mv.new_id != mv_eval.new_id:
                        metrics.metrics.identity_shadow_mismatch_total.labels(
                            feature="multiview_gallery"
                        ).inc()

            if not self._config.enable_embedding_coherence_boost and (
                # sample only a fraction of frames for the shadow comparison.
                # Default 0.0 means no shadow query; set a non-zero rate to
                # evaluate the coherence boost.  This avoids doubling gallery
                # query load every frame in production.
                self._config.coherence_shadow_sample_rate <= 0.0
                or random.random() < self._config.coherence_shadow_sample_rate
            ):
                boosted_reid = await self._from_gallery(entity, enable_coherence_boost=True)
                if boosted_reid.distribution != reid_likelihood.distribution:
                    boosted_posterior = self._combine(
                        prior,
                        face_likelihood,
                        boosted_reid,
                        height_likelihood,
                    )
                    live_eval = self._evaluate_commit(
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
                    )
                    boosted_eval = self._evaluate_commit(
                        entity,
                        boosted_posterior,
                        face_likelihood,
                        boosted_reid,
                        captured_at,
                        entity_quality,
                        contradicted=False,
                        enable_sticky_maintenance=self._config.enable_sticky_maintenance,
                        enforce_quality_gate=self._config.enable_quality_gate,
                        enforce_flip_debounce=self._config.enable_flip_debounce,
                    )
                    if live_eval.new_id != boosted_eval.new_id:
                        metrics.metrics.identity_shadow_mismatch_total.labels(
                            feature="coherence_boost"
                        ).inc()

            # Build identity evidence ledger for this entity.
            evidence_items = self._build_evidence_ledger(
                entity,
                face_likelihood,
                reid_likelihood,
                augmented_evidence,
                best_face_conf,
                captured_at,
            )

            decision = self._commit(
                entity,
                posterior,
                face_likelihood,
                reid_likelihood,
                captured_at,
                best_face_conf,
                entity_quality=entity_quality,
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
            # Use the *original* anchors/evidence (pre-propagation) so the
            # duplicate-guard bypass only trusts a PH's own direct face — a
            # propagated anchor for another person must never grant the bypass.
            direct_face_confidence = self._direct_face_confidence(
                entity,
                new_face_anchors,
                face_evidence or [],
                identity_id=decision.identity_id,
            )
            decision = IdentityDecision(
                ph_id=decision.ph_id,
                identity_id=decision.identity_id,
                posterior=decision.posterior,
                revises_previous=decision.revises_previous,
                previous_identity_id=decision.previous_identity_id,
                reason=decision.reason,
                evidence_backed=decision.evidence_backed,
                evidence={
                    "sources": ep.evidence_summary,
                    "direct_face_confidence": direct_face_confidence,
                    "posterior_entropy": ep.entropy,
                },
            )
            pending_decisions.append(
                _PendingIdentityDecision(
                    entity=entity,
                    decision=decision,
                    direct_face_confidence=direct_face_confidence,
                )
            )

        outcome = ResolveOutcome()
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
                    entity_by_id[decision.ph_id], decision.identity_id
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

        return [self._block_decision(item, blocked) for item in pending]

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
        threshold = self._config.duplicate_identity_direct_face_min_confidence
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
                # No incumbent: let the strongest new assignment claim it.
                winner = max(
                    candidates,
                    key=lambda item: (
                        item.decision.posterior.top_identity()[1],
                        item.decision.posterior.top_with_margin()[1],
                    ),
                )
                candidates = [c for c in candidates if c is not winner]

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
        )

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
    ) -> tuple[PosteriorDist, float | None]:
        """Build likelihood from face anchors associated with this entity.

        Three recognition states are handled distinctly:
        - recognized: (strong positive via p_face).
        - candidate: weak positive for best_candidate_id, scaled by raw similarity
          and frontality.  Does not apply min_conf gating.
        - unrecognized: a face region was present but not recognized.  Adds a small
          mass to UNKNOWN only; never penalizes a held identity.

        Frontality factor (from yaw) is multiplied into every face anchor's effective
        weight so off-axis matches are down-weighted.

        Returns (PosteriorDist, best_confidence) where best_confidence is the
        strongest *recognized* face anchor's confidence, or None when no recognized
        anchor matched.
        """
        entity_obs_ids = set(entity.observation_ids)
        # Build detection_id → evidence lookup from typed evidence records.
        ev_by_detection: dict[str, FaceEvidence] = {}
        if face_evidence:
            for fe in face_evidence:
                if fe.detection_id:
                    ev_by_detection[fe.detection_id] = fe

        # Find face anchors associated with this entity.
        relevant_anchors = [
            fa
            for fa in face_anchors
            if fa.tracklet_id in entity_obs_ids
            or fa.tracklet_id == entity.entity_id
            or fa.detection_id in entity_obs_ids
        ]

        if not relevant_anchors:
            logger.debug(
                "face_no_match_for_entity",
                entity_id=entity.entity_id,
                obs_count=len(entity_obs_ids),
                face_anchor_count=len(face_anchors),
            )
            return PosteriorDist({}), None

        # Separate anchors by recognition state.
        recognized_anchors = [fa for fa in relevant_anchors if fa.recognition_state == "recognized"]
        candidate_anchors = [fa for fa in relevant_anchors if fa.recognition_state == "candidate"]
        unrecognized_anchors = [
            fa for fa in relevant_anchors if fa.recognition_state == "unrecognized"
        ]

        # Build the likelihood distribution.
        likelihood: dict[str, float] = {}
        best_recognized_conf: float | None = None

        # --- Recognized anchors ---
        if recognized_anchors:
            # Use the strongest recognized anchor.
            best = max(recognized_anchors, key=lambda fa: fa.confidence * fa.quality)
            best_recognized_conf = best.confidence

            # Determine evidence source for weight multiplier.
            ev = ev_by_detection.get(best.detection_id)
            source = ev.source if ev and ev.source else "direct"
            weight_mult = (
                self._config.propagated_face_weight_multiplier if source == "propagated" else 1.0
            )

            frontality = self._frontality_factor(best.yaw_deg)
            p_face = self._p_face(best.confidence, best.quality) * weight_mult * frontality

            if best.person_id:
                likelihood[best.person_id] = p_face

            logger.debug(
                "face_anchor_matched",
                entity_id=entity.entity_id,
                person_id=best.person_id,
                confidence=round(best.confidence, 3),
                source=source,
                weight_multiplier=weight_mult,
                frontality_factor=round(frontality, 3),
                anchor_count=len(relevant_anchors),
            )

        # --- Candidate anchors (weak positive, grey zone) ---
        # A candidate face corroborates the held identity through a turn.
        # It is a weak positive for best_candidate_id, scaled by raw similarity.
        #
        # _combine multiplies every face entry by face_weight_multiplier (default 3.0).
        # For candidate entries we want the effective multiplier to be
        # candidate_face_weight_multiplier (default 1.0).  So we pre-adjust the
        # base weight by the ratio so the net effect is correct.
        candidate_ratio = self._config.candidate_face_weight_multiplier / max(
            self._config.face_weight_multiplier, 0.01
        )
        for fa in candidate_anchors:
            if not fa.person_id or fa.person_id == "unknown":
                continue
            frontality = self._frontality_factor(fa.yaw_deg)
            # Base weight: raw cosine similarity scaled by frontality and the
            # candidate-to-face ratio so _combine applies the intended multiplier.
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

        # --- Unrecognized markers (face present, unknown identity) ---
        # Add a small mass to UNKNOWN only.  This nudges toward UNKNOWN for a
        # genuine stranger but never subtracts from a held identity.
        if unrecognized_anchors:
            unknown_mass = self._config.face_present_unknown_unknown_mass
            likelihood["UNKNOWN"] = likelihood.get("UNKNOWN", 0.0) + unknown_mass
            logger.debug(
                "face_unrecognized_marker",
                entity_id=entity.entity_id,
                unrecognized_count=len(unrecognized_anchors),
                unknown_mass_added=unknown_mass,
            )

        # --- Smooth remainder ---
        # - Recognized anchors: spread remainder across all identities + UNKNOWN
        #
        # - Candidate-only anchors: the candidate evidence is a weak positive for
        #   best_candidate_id only.  Put remainder into UNKNOWN so other identities
        #   don't get spurious support from a grey-zone face.
        # - Unrecognized-only: all remainder → UNKNOWN (no positive evidence).
        if likelihood:
            total_face = sum(likelihood.values())
            remainder = max(0.0, 1.0 - total_face)
            if remainder > 0:
                if recognized_anchors:
                    candidates = list(self._identities.keys())
                    if candidates:
                        per_id = remainder / (len(candidates) + 1)
                        for cid in candidates:
                            if cid not in likelihood:
                                likelihood[cid] = per_id
                        likelihood["UNKNOWN"] = likelihood.get("UNKNOWN", 0.0) + per_id
                    else:
                        likelihood["UNKNOWN"] = likelihood.get("UNKNOWN", 0.0) + remainder
                else:
                    # Candidate-only or unrecognized-only: remainder → UNKNOWN.
                    likelihood["UNKNOWN"] = likelihood.get("UNKNOWN", 0.0) + remainder

        return PosteriorDist(likelihood), best_recognized_conf

    def _direct_face_confidence(
        self,
        entity: IdentityResolvableEntity,
        face_anchors: list[FaceAnchor],
        face_evidence: list[FaceEvidence],
        *,
        identity_id: str | None,
    ) -> float:
        """Return the best direct recognized face confidence for this entity/id."""
        if identity_id is None:
            return 0.0

        entity_obs_ids = set(entity.observation_ids)
        matched_ids = entity_obs_ids | {entity.entity_id}

        if face_evidence:
            return max(
                (
                    fe.confidence
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
                fa.confidence
                for fa in face_anchors
                if fa.recognition_state == "recognized"
                and fa.person_id == identity_id
                and (fa.tracklet_id in matched_ids or fa.detection_id in entity_obs_ids)
            ),
            default=0.0,
        )

    def _p_face(self, confidence: float, quality: float) -> float:
        """Probability that a face anchor is correct.

        Real sigmoid calibrated to typical face-recognition performance.
        At (0.5, 0.5) -> ~0.5; at (0.95, 1.0) -> ~0.96.
        """
        combined = 0.7 * confidence + 0.3 * quality
        return min(0.99, 0.5 + 0.5 * math.tanh(4 * (combined - 0.5)))

    def _frontality_factor(self, yaw_deg: float) -> float:
        """Linear frontality ramp based on absolute yaw.

        1.0 at or below frontality_full_yaw_deg (default 15°),
        linearly decreasing to frontality_min_factor (default 0.3)
        at or above frontality_zero_yaw_deg (default 60°).
        """
        abs_yaw = abs(yaw_deg)
        if abs_yaw <= self._config.frontality_full_yaw_deg:
            return 1.0
        if abs_yaw >= self._config.frontality_zero_yaw_deg:
            return self._config.frontality_min_factor
        # Linear interpolation.
        frac = (abs_yaw - self._config.frontality_full_yaw_deg) / (
            self._config.frontality_zero_yaw_deg - self._config.frontality_full_yaw_deg
        )
        return 1.0 - frac * (1.0 - self._config.frontality_min_factor)

    # ------------------------------------------------------------------
    # ReID likelihood from gallery
    # ------------------------------------------------------------------

    async def _from_gallery(
        self,
        entity: IdentityResolvableEntity,
        *,
        enable_coherence_boost: bool | None = None,
        enable_multiview: bool = False,
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
            return await self._from_gallery_multiview(entity, prototypes)

        # --- Single-query path (original behaviour) ---
        try:
            recent = await self._gallery_repo.list_gallery_entries_for_tracklets(
                tracklet_ids=set(entity.observation_ids),
                limit=20,
            )
        except Exception:  # noqa: BLE001
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
            window = embs[-self._config.embedding_coherence_window :]
            norms = np.linalg.norm(window, axis=1, keepdims=True)
            normed = window / np.maximum(norms, 1e-8)
            consecutive_sims = [float(normed[i] @ normed[i + 1]) for i in range(len(normed) - 1)]
            coherence_active = (
                bool(consecutive_sims)
                and min(consecutive_sims) >= self._config.embedding_coherence_min_sim
            )

        try:
            similar = await self._gallery_repo.search_similar(
                embedding=query,
                limit=20,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "reid_search_failed",
                entity_id=entity.entity_id,
                query_dim=len(query),
                exc_info=True,
            )
            return PosteriorDist({})

        if not similar:
            logger.debug(
                "reid_no_similar_matches",
                entity_id=entity.entity_id,
                query_entries=len(recent),
            )
            return PosteriorDist({})

        return self._score_gallery_hits(similar, coherence_active)

    async def _from_gallery_multiview(
        self,
        entity: IdentityResolvableEntity,
        prototypes: tuple[ViewPrototype, ...],
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
                )
            except Exception:  # noqa: BLE001
                logger.warning(
                    "reid_multiview_search_failed",
                    entity_id=entity.entity_id,
                    orientation=int(p.orientation),
                    exc_info=True,
                )
                continue

            all_hits.extend(similar)

        if not all_hits:
            logger.debug(
                "reid_multiview_no_matches",
                entity_id=entity.entity_id,
                queries_run=queries_run,
                prototype_count=len(prototypes),
            )
            return PosteriorDist({})

        # Map hits to per-identity scores using logistic curve.
        likelihood: dict[str, list[float]] = defaultdict(list)
        boosted = False
        for entry, sim in all_hits:
            logit = self._logistic(sim)
            if (
                entry.identity_id is not None
                and sim >= self._config.identified_entry_boost_min_sim
                and logit < self._config.identified_entry_min_likelihood
            ):
                logit = self._config.identified_entry_min_likelihood
                boosted = True
            key = entry.identity_id if entry.identity_id else "UNKNOWN"
            likelihood[key].append(logit)

        # MAX over views per identity (not mean).
        avg: dict[str, float] = {}
        for key, scores in likelihood.items():
            avg[key] = max(scores)  # max-over-views is the core fix

        if not avg:
            return PosteriorDist({})

        if boosted:
            non_match_floor = (1.0 - self._config.identified_entry_min_likelihood) / max(
                len(self._identities), 1
            )
            for iid in self._identities:
                if iid not in avg:
                    avg[iid] = non_match_floor
            if "UNKNOWN" not in avg:
                avg["UNKNOWN"] = non_match_floor

        # Single-identity normalization guard. Without residual UNKNOWN mass a
        # weak best match still normalizes to the only/nearest enrolled identity,
        # so a stranger whose body does not match anyone commits as that identity
        # (the clinical identity-leak this resolver must never produce). Hold the
        # complement of the best identity match strength on UNKNOWN: weak matches
        # resolve to UNKNOWN while strong matches (logit near 1) still commit.
        identity_logits = [v for k, v in avg.items() if k != "UNKNOWN"]
        if identity_logits:
            avg["UNKNOWN"] = max(avg.get("UNKNOWN", 0.0), 1.0 - max(identity_logits))

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

    def _score_gallery_hits(
        self,
        similar: list[tuple[GalleryEmbedding, float]],
        coherence_active: bool = False,
    ) -> PosteriorDist:
        """Score gallery search hits into a PosteriorDist (shared helper)."""
        likelihood: dict[str, list[float]] = defaultdict(list)
        boosted = False
        for entry, sim in similar:
            logit = self._logistic(sim)
            if (
                entry.identity_id is not None
                and sim >= self._config.identified_entry_boost_min_sim
                and logit < self._config.identified_entry_min_likelihood
            ):
                logit = self._config.identified_entry_min_likelihood
                boosted = True
            key = entry.identity_id if entry.identity_id else "UNKNOWN"
            likelihood[key].append(logit)

        # Average scores per identity.
        avg: dict[str, float] = {}
        for key, scores in likelihood.items():
            avg[key] = sum(scores) / len(scores)

        if not avg:
            return PosteriorDist({})

        if boosted:
            non_match_floor = (1.0 - self._config.identified_entry_min_likelihood) / max(
                len(self._identities), 1
            )
            for iid in self._identities:
                if iid not in avg:
                    avg[iid] = non_match_floor
            if "UNKNOWN" not in avg:
                avg["UNKNOWN"] = non_match_floor

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
            face_entry_boosted=boosted,
            coherence_boosted=coherence_boosted_identity,
        )

        return PosteriorDist(avg)

    async def _has_cross_camera_reid_assist(
        self,
        entity: IdentityResolvableEntity,
        identity_id: str,
    ) -> bool:
        """Return True when the committed identity's top ReID hit is cross-camera."""
        try:
            recent = await self._gallery_repo.list_gallery_entries_for_tracklets(
                tracklet_ids=set(entity.observation_ids),
                limit=20,
            )
        except Exception:  # noqa: BLE001
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
            similar = await self._gallery_repo.search_similar(embedding=query, limit=20)
        except Exception:  # noqa: BLE001
            logger.warning(
                "reid_cross_camera_assist_search_failed",
                entity_id=entity.entity_id,
                exc_info=True,
            )
            return False

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
    ) -> PosteriorDist:
        """Build likelihood from height estimates.

        Compares the mean height of this entity's observations against
        enrolled identity height profiles. Returns an empty distribution
        (uniform, no effect) when:
        - No height data is available for this entity.
        - No enrolled identities have height profiles.
        """
        heights_mm: list[float] = []
        for oid in entity.observation_ids:
            h = ph_heights.get(oid)
            if h is not None:
                heights_mm.append(h)

        if not heights_mm:
            return PosteriorDist({})

        # Collect identities that have height profiles.
        has_profiles = any(ident.height_mm is not None for ident in self._identities.values())
        if not has_profiles:
            return PosteriorDist({})

        mean_h = sum(heights_mm) / len(heights_mm)

        likelihood: dict[str, float] = {}
        for identity_id, ident in self._identities.items():
            ref_mm = ident.height_mm
            if ref_mm is None:
                continue
            sigma_mm = ident.height_sigma_mm or 150.0
            z = (mean_h - ref_mm) / sigma_mm
            p = math.exp(-0.5 * z * z)
            likelihood[identity_id] = max(p, 0.01)

        if not likelihood:
            return PosteriorDist({})

        return PosteriorDist(likelihood)

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

        Delegates to the canonical pure function ``combine_posteriors`` in
        ``identity.posterior``. Weights come from ``self._config``.
        """
        return combine_posteriors(
            prior,
            face,
            reid,
            height,
            face_weight_multiplier=self._config.face_weight_multiplier,
            height_weight_multiplier=self._config.height_weight_multiplier,
        )

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
    ) -> IdentityDecision:
        """Apply the commit rule to produce an identity decision.

        Commits an identity if:
        - top_probability >= commit_prob AND margin >= commit_margin
        - At least one non-prior evidence source (face or ReID) supports
          the top identity.  This prevents the temporal prior from
          committing an assignment on its own — a strong prior should
          maintain an existing assignment but not create one.

        Face locks: when a face anchor's confidence exceeds face_commit_min_confidence,
        a face lock is set on this entity.  The face-locked identity uses the longer
        face_lock_maintenance_max_age_s window so it persists across frames where
        face-id is on cooldown.  A different identity's face anchor at the same
        threshold displaces the lock.

        Otherwise, the decision is UNKNOWN (None).
        """
        (top_id, top_prob), margin = posterior.top_with_margin()

        # --- Face lock management ---
        existing_lock = self._face_locks.get(entity.entity_id)
        is_face_evidence = top_id in face_likelihood.distribution and top_id not in ("UNKNOWN", "")
        quality_gate_counted = False
        face_lock_quality_blocked = (
            is_face_evidence
            and best_face_confidence is not None
            and best_face_confidence >= self._config.face_commit_min_confidence
            and entity_quality < self._config.min_quality_to_face_lock
        )
        # Narrow type: only enter the block when confidence is a float.
        if (
            is_face_evidence
            and best_face_confidence is not None
            and best_face_confidence >= self._config.face_commit_min_confidence
            and (not face_lock_quality_blocked or not self._config.enable_quality_gate)
        ):
            face_conf: float = best_face_confidence  # narrowed
            if existing_lock is None or existing_lock.identity_id == top_id:
                self._face_locks[entity.entity_id] = _FaceLock(
                    identity_id=top_id,
                    confidence=face_conf,
                    locked_at=captured_at,
                )
            else:
                logger.info(
                    "face_lock_displaced",
                    entity_id=entity.entity_id,
                    old_identity=existing_lock.identity_id,
                    new_identity=top_id,
                    old_confidence=round(existing_lock.confidence, 3),
                    new_confidence=round(face_conf, 3),
                )
                self._face_locks[entity.entity_id] = _FaceLock(
                    identity_id=top_id,
                    confidence=face_conf,
                    locked_at=captured_at,
                )
        if face_lock_quality_blocked:
            metrics.metrics.identity_quality_gate_blocks_total.inc()
            quality_gate_counted = True

        # Compute contradiction for sticky maintenance.
        prev_id = entity.current_identity_id
        contradicted = compute_contradiction(
            prev_id=prev_id,
            face_likelihood=face_likelihood,
            best_face_confidence=best_face_confidence,
            top_id=top_id,
            top_prob=top_prob,
            margin=margin,
            config=self._to_commit_policy(),
        )

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
        )

        new_id = live_eval.new_id
        if live_eval.quality_gate_blocked and not quality_gate_counted:
            metrics.metrics.identity_quality_gate_blocks_total.inc()
            quality_gate_counted = True

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
            )
            if live_eval.new_id != sticky_shadow_eval.new_id:
                metrics.metrics.identity_shadow_mismatch_total.labels(
                    feature="sticky_maintenance"
                ).inc()

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
            )
            if live_eval.new_id != debounced_eval.new_id:
                metrics.metrics.identity_shadow_mismatch_total.labels(feature="flip_debounce").inc()

        if (
            prev_id is not None
            and not live_eval.within_maintenance_window
            and not live_eval.has_evidence
        ):
            metrics.metrics.identity_decays_total.inc()
            logger.info(
                "identity_maintenance_window_expired",
                entity_id=entity.entity_id,
                prev_identity_id=prev_id,
                identity_age_s=round(
                    (
                        captured_at - (entity.current_identity_committed_at or entity.last_seen_at)
                    ).total_seconds(),
                    1,
                ),
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

        metrics.metrics.posterior_entropy.observe(posterior.entropy())
        if prev_id is not None and new_id is not None and new_id != prev_id:
            metrics.metrics.identity_flips_total.inc()
        if new_id is not None and revises:
            metrics.metrics.identity_commits_total.labels(
                source="face" if top_id in face_likelihood.distribution else "reid",
            ).inc()

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

        return IdentityDecision(
            ph_id=entity.entity_id,
            identity_id=new_id,
            posterior=posterior,
            revises_previous=revises,
            previous_identity_id=prev_id,
            reason=reason,
            evidence_backed=live_eval.evidence_backed,
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
            face_lock_maintenance_max_age_s=c.face_lock_maintenance_max_age_s,
            face_commit_min_confidence=c.face_commit_min_confidence,
            min_quality_to_face_lock=c.min_quality_to_face_lock,
            min_quality_to_commit=c.min_quality_to_commit,
            enable_quality_gate=c.enable_quality_gate,
            flip_debounce_window_s=c.flip_debounce_window_s,
            enable_flip_debounce=c.enable_flip_debounce,
            contradiction_face_confidence=c.contradiction_face_confidence,
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
    ) -> _CommitEvaluation:
        """Evaluate the commit rule — thin delegator to the canonical pure function."""
        return _evaluate_commit_pure(
            entity=entity,
            posterior=posterior,
            face_likelihood=face_likelihood,
            reid_likelihood=reid_likelihood,
            captured_at=captured_at,
            entity_quality=entity_quality,
            face_locks=self._face_locks,
            config=self._to_commit_policy(),
            contradicted=contradicted,
            enable_sticky_maintenance=enable_sticky_maintenance,
            enforce_quality_gate=enforce_quality_gate,
            enforce_flip_debounce=enforce_flip_debounce,
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
                            confidence=fe.confidence,
                            tracklet_id=fe.tracklet_id,
                            captured_at=fe.captured_at,
                            quality=fe.quality,
                        )
                    )
                else:
                    items.append(
                        IdentityEvidence.association_hint(
                            identity_id=fe.person_id,
                            confidence=fe.confidence,
                            tracklet_id=fe.tracklet_id,
                            captured_at=fe.captured_at,
                        )
                    )

        # ReID evidence (top match from gallery).
        if reid_likelihood.distribution:
            top_reid, top_score = max(reid_likelihood.distribution.items(), key=lambda x: x[1])
            if top_reid != "UNKNOWN" and top_score > 0.3:
                items.append(
                    IdentityEvidence.reid(
                        identity_id=top_reid,
                        confidence=top_score,
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

        ev_by_tracklet: dict[str, FaceEvidence] = {
            fe.tracklet_id: fe for fe in face_evidence if fe.tracklet_id
        }

        result: list[FaceEvidence] = []
        for fa in augmented_anchors:
            if fa.tracklet_id in ev_by_tracklet:
                result.append(ev_by_tracklet[fa.tracklet_id])
            else:
                result.append(
                    FaceEvidence(
                        person_id=fa.person_id,
                        confidence=fa.confidence,
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
            if fa.recognition_state != "recognized":
                continue
            src_entity = entity_by_obs.get(fa.tracklet_id) or entity_by_id.get(fa.tracklet_id)
            if src_entity is None:
                continue
            evidenced_entity_ids.add(src_entity.entity_id)
            existing = best_anchor_by_entity.get(src_entity.entity_id)
            if existing is None or (
                fa.confidence * fa.quality > existing.confidence * existing.quality
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

                syn_confidence = src_anchor.confidence * sim
                if syn_confidence < self._config.face_commit_min_confidence:
                    continue
                syn = FaceAnchor(
                    person_id=src_anchor.person_id,
                    confidence=syn_confidence,
                    quality=src_anchor.quality * 0.8,
                    tracklet_id=dst_entity.observation_ids[0],
                    camera_id=dst_entity.camera_ids[0] if dst_entity.camera_ids else "",
                    captured_at=src_anchor.captured_at,
                )
                synthetic.append(syn)
                propagated_count += 1
                metrics.metrics.face_propagations_total.inc()

                logger.info(
                    "face_anchor_propagated",
                    src_entity=src_entity_id,
                    dst_entity=dst_entity.entity_id,
                    person_id=src_anchor.person_id,
                    original_confidence=round(src_anchor.confidence, 3),
                    propagated_confidence=round(syn.confidence, 3),
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

    def get_face_locked_identity(self, global_track_id: str) -> str | None:
        """Return the face-locked identity for a GT, or None if not locked."""
        lock = self._face_locks.get(global_track_id)
        return lock.identity_id if lock is not None else None

    def register_identity(self, identity: Identity) -> None:
        """Register a known identity for display name lookup."""
        self._identities[identity.identity_id] = identity


# ---------------------------------------------------------------------------
# Contradiction helper (pure, outside the class)
# ---------------------------------------------------------------------------


# _compute_contradiction has been moved to identity.commit_policy.compute_contradiction.
# The resolver calls compute_contradiction() (imported at the top of this module).

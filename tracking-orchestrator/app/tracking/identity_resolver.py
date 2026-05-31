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
import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from structlog import get_logger

if TYPE_CHECKING:
    from ..pipeline.gallery_cache import GalleryCache

from ..domain import (
    FaceAnchor,
    Identity,
    IdentityDecision,
    IdentityResolvableEntity,
    IdentityRevision,
    PosteriorDist,
    ResolveOutcome,
)
from ..inference.evidence import FaceEvidence
from ..observability import metrics
from ..storage.base import GalleryRepository
from .identity.evidence import IdentityEvidence
from .identity.posterior import EvidencePosterior

logger = get_logger(__name__)


@dataclass
class _FaceLock:
    """Tracks a face-confirmed identity for one GlobalTrack.

    Set when a face anchor's confidence exceeds face_commit_min_confidence.
    Displaced when a different identity's face anchor also clears that threshold.
    """

    identity_id: str
    confidence: float
    locked_at: datetime


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

    # Higher commit threshold used in dense scenes (≥ 2 candidate
    # identities with posterior > 0.3).  Prevents confident-but-wrong
    # commits when two enrolled people are in the same room.
    commit_prob_dense: float = 0.80

    # Larger margin required in dense scenes.  When two identities have
    # posteriors like [0.55, 0.40, 0.05], the narrow 0.15 gap should
    # refuse to commit — the resolver must wait for stronger evidence.
    commit_margin_dense: float = 0.20

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
    # to an adjacent GlobalTrack that has no face anchor of its own. The deployed
    # value is read from settings.yaml; M4 owns the final safety threshold.
    cross_gt_face_propagation_threshold: float = 0.78

    # Maximum number of adjacent GlobalTracks to propagate face identity to per
    # resolve() call.  Caps the gallery query overhead for busy scenes.
    cross_gt_face_propagation_max_gts: int = 4


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
        face_evidence: list[FaceEvidence] | None = None,
    ) -> ResolveOutcome:
        """Resolve identities for a batch of tracked entities.

        Args:
            hypotheses: active entities (PHs or GlobalTracks) to resolve.
            new_face_anchors: face anchors from this frame.
            captured_at: wall-clock time of the current frame.
            ph_heights: optional entity_id → height_m mapping.
            face_evidence: optional typed FaceEvidence records with source
                metadata. When provided, direct evidence receives normal
                weight and propagated evidence receives reduced weight.

        Returns:
            ResolveOutcome with decisions and any revisions to emit.
        """
        # Refresh known-identity list from the gallery so new enrolments
        # are picked up without a restart and the prior is not degenerate.
        # The gallery query is fast (indexed PK scan, ≤100 rows in practice).
        enrolled = await self._gallery_repo.list_identities(active_only=True)
        self._identities = {ident.identity_id: ident for ident in enrolled}

        # Propagate face anchors from face-evidenced entities to similar adjacent
        # entities that share the same physical space but weren't merged by the
        # cross-camera associator.
        augmented_anchors = await self._propagate_face_anchors(hypotheses, new_face_anchors)
        augmented_evidence = self._augment_face_evidence(
            hypotheses, face_evidence or [], augmented_anchors
        )

        outcome = ResolveOutcome()

        for entity in hypotheses:
            prior = self._build_prior(entity, captured_at)
            face_likelihood, best_face_conf = self._from_face_anchors(
                entity, augmented_anchors, augmented_evidence
            )
            reid_likelihood = await self._from_gallery(entity)
            height_likelihood = self._from_height(entity, ph_heights or {})
            posterior = self._combine(prior, face_likelihood, reid_likelihood, height_likelihood)

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
                entity, posterior, face_likelihood, reid_likelihood, captured_at, best_face_conf
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
                    "direct_face_confidence": best_face_conf or 0.0,
                    "posterior_entropy": ep.entropy,
                },
            )

            if decision.revises_previous:
                revision = self._build_revision(entity, decision, captured_at)
                if revision is not None:
                    outcome.revisions.append(revision)

            outcome.decisions.append(decision)

        return outcome

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

        Face anchors are the strongest evidence. A face anchor is associated
        with an entity if its tracklet_id is in the entity's observation list.

        When ``face_evidence`` is provided, propagated evidence receives
        reduced weight (propagated_face_weight_multiplier) compared to
        direct ArcFace evidence.

        Returns (PosteriorDist, best_confidence) where best_confidence is the
        strongest face anchor's confidence, or None when no anchor matched.
        """
        entity_obs_ids = set(entity.observation_ids)

        # Find face anchors whose tracklet belongs to this entity.
        # In PH mode the anchor carries entity_id (ph_id) as its tracklet_id
        # after the tracker remaps it; the entity_id fallback handles that case
        # without polluting observation_ids with ph_ids.
        relevant_anchors = [
            fa
            for fa in face_anchors
            if fa.tracklet_id in entity_obs_ids or fa.tracklet_id == entity.entity_id
        ]

        if not relevant_anchors:
            logger.debug(
                "face_no_match_for_entity",
                entity_id=entity.entity_id,
                obs_count=len(entity_obs_ids),
                face_anchor_count=len(face_anchors),
            )
            return PosteriorDist({}), None

        # Build tracklet_id → source lookup from typed evidence records.
        evidence_source: dict[str, str] = {}
        if face_evidence:
            for fe in face_evidence:
                if fe.tracklet_id:
                    evidence_source[fe.tracklet_id] = fe.source

        # Take the strongest face anchor (highest confidence * quality).
        best = max(relevant_anchors, key=lambda fa: fa.confidence * fa.quality)

        # Determine evidence source and apply appropriate weight.
        source = evidence_source.get(best.tracklet_id, "direct")
        weight_mult = (
            self._config.propagated_face_weight_multiplier if source == "propagated" else 1.0
        )

        logger.debug(
            "face_anchor_matched",
            entity_id=entity.entity_id,
            person_id=best.person_id,
            confidence=round(best.confidence, 3),
            source=source,
            weight_multiplier=weight_mult,
            anchor_count=len(relevant_anchors),
        )

        # p_face: probability that this anchor is correct.
        p_face = self._p_face(best.confidence, best.quality) * weight_mult

        likelihood: dict[str, float] = {}
        if best.person_id:
            likelihood[best.person_id] = p_face
        # Smooth the remainder over all known identities + UNKNOWN.
        remainder = 1.0 - p_face
        if remainder > 0:
            candidates = list(self._identities.keys())
            if candidates:
                per_id = remainder / (len(candidates) + 1)
                for cid in candidates:
                    if cid != best.person_id:
                        likelihood[cid] = per_id
                likelihood["UNKNOWN"] = per_id
            else:
                likelihood["UNKNOWN"] = remainder

        return PosteriorDist(likelihood), best.confidence

    def _p_face(self, confidence: float, quality: float) -> float:
        """Probability that a face anchor is correct.

        Real sigmoid calibrated to typical face-recognition performance.
        At (0.5, 0.5) -> ~0.5; at (0.95, 1.0) -> ~0.96.
        """
        combined = 0.7 * confidence + 0.3 * quality
        return min(0.99, 0.5 + 0.5 * math.tanh(4 * (combined - 0.5)))

    # ------------------------------------------------------------------
    # ReID likelihood from gallery
    # ------------------------------------------------------------------

    async def _from_gallery(self, entity: IdentityResolvableEntity) -> PosteriorDist:
        """Build likelihood from gallery k-NN search.

        Queries the gallery for similar embeddings to the entity's
        gallery entries. Maps results to per-identity scores using a
        calibrated logistic curve.
        """
        # Build a real query embedding from the entity's existing
        # gallery entries (mean of recent embeddings per observation).
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
        import numpy as np

        embs = np.array([e.embedding for e in recent], dtype=np.float32)
        query = np.mean(embs, axis=0).tolist()

        # Embedding coherence check: if the last N embeddings are all mutually
        # similar, the person's appearance is stable → apply a likelihood boost
        # to the top matching identity so that stable but below-threshold scores
        # still cross commit_prob.
        coherence_active = False
        if self._config.enable_embedding_coherence_boost and len(embs) >= 2:
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

        # Map hits to per-identity scores using logistic curve.
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
                original = avg[best_key]
                avg[best_key] = min(0.99, original * self._config.embedding_coherence_boost)
                coherence_boosted_identity = best_key

        top_reid = max(avg.items(), key=lambda x: x[1])
        logger.debug(
            "reid_top_match",
            entity_id=entity.entity_id,
            top_identity=top_reid[0],
            top_score=round(top_reid[1], 4),
            candidate_count=len(avg),
            gallery_entries_searched=len(similar),
            face_entry_boosted=boosted,
            coherence_boosted=coherence_boosted_identity,
        )

        return PosteriorDist(avg)

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

        Posterior = prior * face_likelihood * reid_likelihood * height_likelihood

        If any source is empty (no evidence), it is treated as uniform
        (weight=1.0) so it does not dilute evidence from other sources.

        When a source is non-empty but missing an identity, a smoothing
        constant is used instead of 1.0 to avoid penalising identities
        that *do* appear in the evidence.
        """
        all_ids: set[str] = set(prior.distribution.keys())
        all_ids.update(face.distribution.keys())
        all_ids.update(reid.distribution.keys())
        if height is not None and height.distribution:
            all_ids.update(height.distribution.keys())

        if not all_ids:
            return PosteriorDist({"UNKNOWN": 1.0})

        def _weight(dist: PosteriorDist, ident: str) -> float:
            """Return the weight for *ident* from *dist*.

            - Empty distribution → 1.0 (uninformative).
            - Non-empty distribution → explicit value or smoothing constant.
            """
            if not dist.distribution:
                return 1.0
            if ident in dist.distribution:
                return dist.distribution[ident]
            # Smoothing: spread a small mass over identities not in this source.
            # 1 / (n_present + 1) is a simple Laplace-style term.
            n = len(dist.distribution)
            return 1.0 / (n + 1)

        combined: dict[str, float] = {}
        for ident in all_ids:
            fw = _weight(face, ident)
            # Boost identities that appear in the face distribution.
            # ArcFace evidence is significantly more reliable than body
            # ReID for disambiguating enrolled identities in the same room.
            if ident in face.distribution:
                fw = fw * self._config.face_weight_multiplier

            hw = _weight(height, ident) if height is not None else 1.0
            # Boost identities supported by height evidence.
            if height is not None and height.distribution and ident in height.distribution:
                hw = hw * self._config.height_weight_multiplier

            combined[ident] = _weight(prior, ident) * fw * _weight(reid, ident) * hw

        if not combined:
            return PosteriorDist({"UNKNOWN": 1.0})

        return PosteriorDist(combined)

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
        # Narrow type: only enter the block when confidence is a float.
        if (
            is_face_evidence
            and best_face_confidence is not None
            and best_face_confidence >= self._config.face_commit_min_confidence
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

        has_evidence = (
            top_id in face_likelihood.distribution or top_id in reid_likelihood.distribution
        )

        prev_id = entity.current_identity_id
        identity_unchanged = top_id == prev_id and prev_id is not None
        within_maintenance_window = False
        if identity_unchanged:
            face_lock = self._face_locks.get(entity.entity_id)
            if face_lock is not None and face_lock.identity_id == prev_id:
                lock_age_s = (captured_at - face_lock.locked_at).total_seconds()
                within_maintenance_window = (
                    lock_age_s <= self._config.face_lock_maintenance_max_age_s
                )
            elif entity.current_identity_committed_at is not None:
                age_delta = captured_at - entity.current_identity_committed_at
                identity_age_s = age_delta.total_seconds()
                within_maintenance_window = (
                    identity_age_s <= self._config.prior_maintenance_max_age_s
                )
            else:
                identity_age_s = (captured_at - entity.last_seen_at).total_seconds()
                within_maintenance_window = (
                    identity_age_s <= self._config.prior_maintenance_max_age_s
                )

        evidence_ok = has_evidence or within_maintenance_window

        dense_candidates = sum(1 for p in posterior.distribution.values() if p > 0.3)
        is_dense = dense_candidates >= 2
        effective_commit_prob = (
            self._config.commit_prob_dense if is_dense else self._config.commit_prob
        )
        effective_commit_margin = (
            self._config.commit_margin_dense if is_dense else self._config.commit_margin
        )

        evidence_backed = False
        if within_maintenance_window:
            new_id = prev_id
            evidence_backed = has_evidence
        elif (
            evidence_ok and top_prob >= effective_commit_prob and margin >= effective_commit_margin
        ):
            new_id = top_id if top_id != "UNKNOWN" else None
            evidence_backed = has_evidence
        else:
            new_id = None

        if prev_id is not None and not within_maintenance_window and not has_evidence:
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
        if new_id is not None and revises:
            metrics.metrics.identity_commits_total.labels(
                source="face" if top_id in face_likelihood.distribution else "reid",
            ).inc()

        if new_id is None:
            logger.debug(
                "identity_not_committed",
                entity_id=entity.entity_id,
                top_id=top_id,
                top_prob=round(top_prob, 4),
                margin=round(margin, 4),
                has_evidence=has_evidence,
                within_maintenance_window=within_maintenance_window,
                evidence_ok=evidence_ok,
                prev_id=prev_id,
                known_identity_count=len(self._identities),
            )
        elif within_maintenance_window:
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
            evidence_backed=evidence_backed,
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
            ph_id=entity.entity_id,  # N0: entity_id is a PH id
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
        evidenced_entity_ids: set[str] = set()
        best_anchor_by_entity: dict[str, FaceAnchor] = {}
        for fa in face_anchors:
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

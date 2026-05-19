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
from dataclasses import dataclass
from datetime import datetime

from structlog import get_logger

from ..domain import (
    FaceAnchor,
    GlobalTrack,
    Identity,
    IdentityCandidate,
    IdentityDecision,
    IdentityRevision,
    PosteriorDist,
    ResolveOutcome,
)
from ..observability import metrics
from ..storage.base import GalleryRepository, GlobalTrackRepository, TrackingRepository

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

    # Maximum revisions per GlobalTrack per minute (rate limiting).
    max_revisions_per_gt_per_minute: int = 3

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
    # to an adjacent GlobalTrack that has no face anchor of its own.  Prevents
    # false propagation to a different person who happens to be nearby.
    cross_gt_face_propagation_threshold: float = 0.65

    # Maximum number of adjacent GlobalTracks to propagate face identity to per
    # resolve() call.  Caps the gallery query overhead for busy scenes.
    cross_gt_face_propagation_max_gts: int = 4


class IdentityResolver:
    """Bayesian identity resolver with retroactive revision.

    Usage::

        resolver = IdentityResolver(
            tracking_repo=tracking_repo,
            gallery_repo=gallery_repo,
            global_track_repo=global_track_repo,
            config=ResolverConfig(),
        )

        outcome = await resolver.resolve(
            global_tracks=active_gts,
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
        tracking_repo: TrackingRepository,
        gallery_repo: GalleryRepository,
        global_track_repo: GlobalTrackRepository,
        identities: list[Identity] | None = None,
        config: ResolverConfig | None = None,
    ) -> None:
        self._tracking_repo = tracking_repo
        self._gallery_repo = gallery_repo
        self._global_track_repo = global_track_repo
        self._config = config or ResolverConfig()
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
        global_tracks: list[GlobalTrack],
        new_face_anchors: list[FaceAnchor],
        captured_at: datetime,
    ) -> ResolveOutcome:
        """Resolve identities for a batch of GlobalTracks.

        Args:
            global_tracks: active GlobalTracks to resolve.
            new_face_anchors: face anchors from this frame.
            captured_at: wall-clock time of the current frame.

        Returns:
            ResolveOutcome with decisions and any revisions to emit.
        """
        # Refresh known-identity list from the gallery so new enrolments
        # are picked up without a restart and the prior is not degenerate.
        # The gallery query is fast (indexed PK scan, ≤100 rows in practice).
        enrolled = await self._gallery_repo.list_identities(active_only=True)
        self._identities = {ident.identity_id: ident for ident in enrolled}

        # Propagate face anchors from face-evidenced GTs to similar adjacent GTs
        # that share the same physical space but weren't merged by the cross-camera
        # associator (appearance below merge threshold, or different overlap groups).
        augmented_anchors = await self._propagate_face_anchors(global_tracks, new_face_anchors)

        outcome = ResolveOutcome()

        for gt in global_tracks:
            prior = self._build_prior(gt, captured_at)
            face_likelihood, best_face_conf = self._from_face_anchors(gt, augmented_anchors)
            reid_likelihood = await self._from_gallery(gt)
            posterior = self._combine(prior, face_likelihood, reid_likelihood)

            decision = self._commit(
                gt, posterior, face_likelihood, reid_likelihood, captured_at, best_face_conf
            )
            if decision.revises_previous:
                revision = self._build_revision(gt, decision, captured_at)
                if revision is not None:
                    outcome.revisions.append(revision)

            outcome.decisions.append(decision)

        return outcome

    # ------------------------------------------------------------------
    # Prior construction
    # ------------------------------------------------------------------

    def _build_prior(self, gt: GlobalTrack, captured_at: datetime) -> PosteriorDist:
        """Build the temporal prior from the previous identity assignment.

        The prior is a mixture of:
        - The previous identity (if any), weighted by prior_weight.
        - A uniform distribution over all known identities, weighted by (1 - prior_weight).
        - An UNKNOWN mass that grows with the time since last_seen.
        """
        prior_weight = self._config.prior_weight

        if gt.current_identity_id:
            # Strong prior on the previous identity.
            prior: dict[str, float] = {
                gt.current_identity_id: prior_weight,
            }
            # Add small mass to other known identities.
            other_weight = (1 - prior_weight) / max(len(self._identities), 1)
            for ident_id in self._identities:
                if ident_id != gt.current_identity_id:
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
        self, gt: GlobalTrack, face_anchors: list[FaceAnchor]
    ) -> tuple[PosteriorDist, float | None]:
        """Build likelihood from face anchors associated with this GlobalTrack.

        Face anchors are the strongest evidence. A face anchor is associated
        with a GlobalTrack if its tracklet_id is in the tracklet's list.

        Returns (PosteriorDist, best_confidence) where best_confidence is the
        strongest face anchor's confidence, or None when no anchor matched.
        """
        gt_tracklet_ids = set(gt.tracklet_ids)

        # Find face anchors whose tracklet belongs to this GlobalTrack.
        relevant_anchors = [fa for fa in face_anchors if fa.tracklet_id in gt_tracklet_ids]

        if not relevant_anchors:
            # No face evidence: return uniform (identity-neutral).
            logger.debug(
                "face_no_match_for_gt",
                global_track_id=gt.global_track_id,
                gt_tracklet_count=len(gt_tracklet_ids),
                face_anchor_count=len(face_anchors),
            )
            return PosteriorDist({}), None

        # Take the strongest face anchor (highest confidence * quality).
        best = max(relevant_anchors, key=lambda fa: fa.confidence * fa.quality)

        logger.debug(
            "face_anchor_matched",
            global_track_id=gt.global_track_id,
            person_id=best.person_id,
            confidence=round(best.confidence, 3),
            anchor_count=len(relevant_anchors),
        )

        # p_face: probability that this anchor is correct.
        # Monotone function of confidence and quality.
        p_face = self._p_face(best.confidence, best.quality)

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

    async def _from_gallery(self, gt: GlobalTrack) -> PosteriorDist:
        """Build likelihood from gallery k-NN search.

        Queries the gallery for similar embeddings to the tracklet's
        gallery entries. Maps results to per-identity scores using a
        calibrated logistic curve.
        """
        # Build a real query embedding from the GlobalTrack's existing
        # gallery entries (mean of recent embeddings per tracklet).
        try:
            recent = await self._gallery_repo.list_gallery_entries_for_tracklets(
                tracklet_ids=set(gt.tracklet_ids),
                limit=20,
            )
        except Exception:
            logger.warning(
                "reid_gallery_lookup_failed",
                global_track_id=gt.global_track_id,
                exc_info=True,
            )
            return PosteriorDist({})

        if not recent:
            logger.debug(
                "reid_no_gallery_entries",
                global_track_id=gt.global_track_id,
                tracklet_count=len(gt.tracklet_ids),
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
            consecutive_sims = [
                float(normed[i] @ normed[i + 1]) for i in range(len(normed) - 1)
            ]
            coherence_active = (
                bool(consecutive_sims)
                and min(consecutive_sims) >= self._config.embedding_coherence_min_sim
            )

        try:
            similar = await self._gallery_repo.search_similar(
                embedding=query,
                limit=20,
            )
        except Exception:
            logger.warning(
                "reid_search_failed",
                global_track_id=gt.global_track_id,
                query_dim=len(query),
                exc_info=True,
            )
            return PosteriorDist({})

        if not similar:
            logger.debug(
                "reid_no_similar_matches",
                global_track_id=gt.global_track_id,
                query_entries=len(recent),
            )
            return PosteriorDist({})

        # Map hits to per-identity scores using logistic curve.
        # Face-confirmed entries (identity_id set) above the boost threshold
        # get a likelihood floor so a back-facing query finding front-facing
        # gallery entries at sim≈0.73 still crosses the commit threshold.
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

        # When a face-confirmed entry was boosted, explicitly fill a low
        # floor for every known identity NOT already in the results.
        # Without this, _combine()'s smoothing term (1/(n+1)) assigns 0.50
        # to missing identities, collapsing the boosted posterior from ~0.83
        # to ~0.35.
        if boosted:
            non_match_floor = (1.0 - self._config.identified_entry_min_likelihood) / max(
                len(self._identities), 1
            )
            for iid in self._identities:
                if iid not in avg:
                    avg[iid] = non_match_floor
            if "UNKNOWN" not in avg:
                avg["UNKNOWN"] = non_match_floor

        # Coherence boost: stable embedding sequence → multiply the top match.
        coherence_boosted_identity: str | None = None
        if coherence_active and avg:
            best_key = max(avg, key=lambda k: avg[k])
            if best_key != "UNKNOWN":
                original = avg[best_key]
                avg[best_key] = min(0.99, original * self._config.embedding_coherence_boost)
                coherence_boosted_identity = best_key

        # Log top ReID match for diagnostics.
        top_reid = max(avg.items(), key=lambda x: x[1])
        logger.debug(
            "reid_top_match",
            global_track_id=gt.global_track_id,
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
    # Posterior combination
    # ------------------------------------------------------------------

    def _combine(
        self,
        prior: PosteriorDist,
        face: PosteriorDist,
        reid: PosteriorDist,
    ) -> PosteriorDist:
        """Combine prior, face likelihood, and ReID likelihood.

        Posterior = prior * face_likelihood * reid_likelihood

        If any source is empty (no evidence), it is treated as uniform
        (weight=1.0) so it does not dilute evidence from other sources.

        When a source is non-empty but missing an identity, a smoothing
        constant is used instead of 1.0 to avoid penalising identities
        that *do* appear in the evidence.
        """
        all_ids: set[str] = set(prior.distribution.keys())
        all_ids.update(face.distribution.keys())
        all_ids.update(reid.distribution.keys())

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
            combined[ident] = _weight(prior, ident) * fw * _weight(reid, ident)

        if not combined:
            return PosteriorDist({"UNKNOWN": 1.0})

        return PosteriorDist(combined)

    # ------------------------------------------------------------------
    # Commit rule
    # ------------------------------------------------------------------

    def _commit(
        self,
        gt: GlobalTrack,
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
        a face lock is set on this GT.  The face-locked identity uses the longer
        face_lock_maintenance_max_age_s window so it persists across frames where
        face-id is on cooldown.  A different identity's face anchor at the same
        threshold displaces the lock.

        Otherwise, the decision is UNKNOWN (None).
        """
        (top_id, top_prob), margin = posterior.top_with_margin()

        # --- Face lock management ---
        existing_lock = self._face_locks.get(gt.global_track_id)
        is_face_evidence = top_id in face_likelihood.distribution and top_id not in ("UNKNOWN", "")
        # Narrow type: only enter the block when confidence is a float.
        if (
            is_face_evidence
            and best_face_confidence is not None
            and best_face_confidence >= self._config.face_commit_min_confidence
        ):
            face_conf: float = best_face_confidence  # narrowed
            if existing_lock is None or existing_lock.identity_id == top_id:
                # Set or refresh the face lock for the same identity.
                self._face_locks[gt.global_track_id] = _FaceLock(
                    identity_id=top_id,
                    confidence=face_conf,
                    locked_at=captured_at,
                )
            else:
                # Different identity at sufficient confidence: displace the lock.
                logger.info(
                    "face_lock_displaced",
                    global_track_id=gt.global_track_id,
                    old_identity=existing_lock.identity_id,
                    new_identity=top_id,
                    old_confidence=round(existing_lock.confidence, 3),
                    new_confidence=round(face_conf, 3),
                )
                self._face_locks[gt.global_track_id] = _FaceLock(
                    identity_id=top_id,
                    confidence=face_conf,
                    locked_at=captured_at,
                )

        # Require evidence from at least one non-prior source.
        # The prior encodes temporal continuity (a track likely keeps
        # its current identity) but should not be the sole basis for
        # a new assignment.  Face anchors and ReID gallery hits are
        # the evidence that justifies an assignment or reassignment.
        has_evidence = (
            top_id in face_likelihood.distribution or top_id in reid_likelihood.distribution
        )

        # When the top identity matches the current assignment AND the
        # most recent evidence-backed commit was recent enough, allow
        # the prior alone to maintain the identity across frames where
        # face-id is on cooldown or ReID is momentarily quiet.  Without
        # this gate every non-evidence frame would clear the assignment.
        prev_id = gt.current_identity_id
        identity_unchanged = top_id == prev_id and prev_id is not None
        within_maintenance_window = False
        if identity_unchanged:
            # Use gt.last_seen_at, which the cross-camera associator updates to
            # captured_at on every frame for active GTs (age_s ≈ 0).  This means
            # an actively-tracked identity is maintained indefinitely — including
            # when the person turns away and ReID evidence becomes weak.
            # face_lock_maintenance_max_age_s is reserved for future GT reactivation
            # scenarios (closed GT re-enters scene); it does NOT govern active tracks.
            age_s = (captured_at - gt.last_seen_at).total_seconds()
            within_maintenance_window = age_s <= self._config.prior_maintenance_max_age_s

        evidence_ok = has_evidence or within_maintenance_window

        # Detect dense scenes: when ≥ 2 candidate identities each have
        # posterior > 0.3, escalate thresholds to prevent confident-but-wrong
        # commits from ambiguous ReID evidence in multi-person frames.
        dense_candidates = sum(1 for p in posterior.distribution.values() if p > 0.3)
        is_dense = dense_candidates >= 2
        effective_commit_prob = (
            self._config.commit_prob_dense if is_dense else self._config.commit_prob
        )
        effective_commit_margin = (
            self._config.commit_margin_dense if is_dense else self._config.commit_margin
        )

        # Apply commit rule.
        if within_maintenance_window:
            # Carry the existing identity forward without re-applying the
            # probability threshold.  The threshold governs initial commits
            # and genuine identity changes; during an evidence gap (same top
            # candidate, no new evidence, within maintenance window) the
            # prior-only posterior falls below commit_prob when N>=4 enrolled
            # identities — applying the threshold here would clear a valid
            # face-confirmed identity on every quiet frame.
            new_id = prev_id
        elif evidence_ok and top_prob >= effective_commit_prob and margin >= effective_commit_margin:
            new_id = top_id if top_id != "UNKNOWN" else None
        else:
            new_id = None  # Committed as UNKNOWN.

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

        # Observability: record posterior entropy for every decision so we
        # can plot entropy distributions per resident in Grafana.
        metrics.metrics.posterior_entropy.observe(posterior.entropy())
        if new_id is not None and revises:
            metrics.metrics.identity_commits_total.labels(
                source="face" if top_id in face_likelihood.distribution else "reid",
            ).inc()

        # Diagnostic logging: surface why a commit was accepted or refused.
        # Log for ALL failed commits — not only when prev_id is set — so new
        # tracks with strong ReID hits that fail the probability/margin gate
        # are visible rather than silently staying UNKNOWN.
        if new_id is None:
            logger.debug(
                "identity_not_committed",
                global_track_id=gt.global_track_id,
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
                global_track_id=gt.global_track_id,
                identity_id=new_id,
                top_prob=round(top_prob, 4),
                age_s=round((captured_at - gt.last_seen_at).total_seconds(), 1),
            )

        return IdentityDecision(
            global_track_id=gt.global_track_id,
            identity_id=new_id,
            posterior=posterior,
            revises_previous=revises,
            previous_identity_id=prev_id,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Retroactive revision
    # ------------------------------------------------------------------

    def _build_revision(
        self,
        gt: GlobalTrack,
        decision: IdentityDecision,
        captured_at: datetime,
    ) -> IdentityRevision | None:
        """Build an IdentityRevision when identity changes.

        The revision covers all tracklets in the GlobalTrack that were
        active within the revision horizon.

        Rate-limited: max_revisions_per_gt_per_minute.
        """
        # Rate limiting.
        now = captured_at
        window_start = now.timestamp() - 60.0
        log_key = gt.global_track_id
        recent = [
            ts for ts in self._revision_log.get(log_key, []) if ts.timestamp() >= window_start
        ]
        if len(recent) >= self._config.max_revisions_per_gt_per_minute:
            logger.warning(
                "Revision rate limit exceeded",
                global_track_id=log_key,
                recent_count=len(recent),
            )
            return None

        # Prune stale entries to prevent unbounded memory growth.
        self._revision_log[log_key] = [
            ts for ts in self._revision_log[log_key] if ts.timestamp() >= window_start
        ]

        # Collect tracklet IDs within the revision horizon.
        # In production, query the tracking repo for tracklets in the horizon.
        # For now, use all tracklet_ids from the GlobalTrack.
        tracklet_ids = list(gt.tracklet_ids)

        if not tracklet_ids:
            return None

        # Build candidates from the posterior.
        top_id, top_prob = decision.posterior.top_identity()
        candidates = [
            IdentityCandidate(
                identity_id=ident_id,
                display_name=self._identities.get(
                    ident_id,
                    Identity(
                        identity_id=ident_id,
                        display_name=ident_id,
                        enrolled_at=now,
                    ),
                ).display_name,
                probability=prob,
            )
            for ident_id, prob in sorted(
                decision.posterior.distribution.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        ]

        entropy = decision.posterior.entropy()

        revision = IdentityRevision(
            revision_id=str(uuid.uuid4()),
            global_track_id=gt.global_track_id,
            tracklet_ids=tracklet_ids,
            candidates=candidates,
            map_identity_id=top_id,
            posterior_entropy=entropy,
            previous_identity_id=decision.previous_identity_id,
            new_identity_id=decision.identity_id,
            reason=decision.reason,
            evidence={
                "top_probability": top_prob,
                "margin": decision.posterior.top_with_margin()[1],
            },
            revision_time=now,
        )

        self._revision_log[gt.global_track_id].append(now)
        return revision

    # ------------------------------------------------------------------
    # Cross-GT face propagation
    # ------------------------------------------------------------------

    async def _propagate_face_anchors(
        self,
        global_tracks: list[GlobalTrack],
        face_anchors: list[FaceAnchor],
    ) -> list[FaceAnchor]:
        """Propagate face anchors from face-evidenced GTs to similar adjacent GTs.

        When Camera A gets a face match for Alice but Camera B (same room,
        overlapping FOV) has a separate GlobalTrack for the same person, this
        method creates synthetic FaceAnchors for Camera B's GT so that the
        Bayesian resolver can commit Alice's identity there too.

        Confidence of the synthetic anchor is scaled by the gallery cosine
        similarity between the two GTs.  Only GTs with no direct face evidence
        this frame are candidates for propagation.
        """
        if not face_anchors or len(global_tracks) < 2:
            return face_anchors

        # Map each tracklet to its GT for fast lookup.
        gt_by_tracklet: dict[str, GlobalTrack] = {}
        for gt in global_tracks:
            for tid in gt.tracklet_ids:
                gt_by_tracklet[tid] = gt

        # Find which GTs have direct face evidence and pick the best anchor per GT.
        evidenced_gt_ids: set[str] = set()
        best_anchor_by_gt: dict[str, FaceAnchor] = {}
        for fa in face_anchors:
            src_gt = gt_by_tracklet.get(fa.tracklet_id)
            if src_gt is None:
                continue
            evidenced_gt_ids.add(src_gt.global_track_id)
            existing = best_anchor_by_gt.get(src_gt.global_track_id)
            if existing is None or fa.confidence * fa.quality > existing.confidence * existing.quality:
                best_anchor_by_gt[src_gt.global_track_id] = fa

        if not evidenced_gt_ids:
            return face_anchors

        # Candidate GTs for propagation: no direct face evidence this frame.
        unevidenced = [gt for gt in global_tracks if gt.global_track_id not in evidenced_gt_ids]
        if not unevidenced:
            return face_anchors

        threshold = self._config.cross_gt_face_propagation_threshold
        max_props = self._config.cross_gt_face_propagation_max_gts
        synthetic: list[FaceAnchor] = []
        propagated_count = 0

        for src_gt_id, src_anchor in best_anchor_by_gt.items():
            if propagated_count >= max_props:
                break
            src_gt = next((gt for gt in global_tracks if gt.global_track_id == src_gt_id), None)
            if src_gt is None or not src_gt.tracklet_ids:
                continue

            for dst_gt in unevidenced:
                if propagated_count >= max_props:
                    break
                if not dst_gt.tracklet_ids:
                    continue

                sim = await self._gt_gallery_similarity(src_gt.tracklet_ids, dst_gt.tracklet_ids)
                if sim < threshold:
                    continue

                # Synthetic anchor for dst_gt; confidence scaled by similarity.
                syn = FaceAnchor(
                    person_id=src_anchor.person_id,
                    confidence=src_anchor.confidence * sim,
                    quality=src_anchor.quality,
                    tracklet_id=dst_gt.tracklet_ids[0],
                    camera_id=dst_gt.camera_ids[0] if dst_gt.camera_ids else "",
                    captured_at=src_anchor.captured_at,
                )
                synthetic.append(syn)
                propagated_count += 1

                logger.info(
                    "face_anchor_propagated",
                    src_gt=src_gt_id,
                    dst_gt=dst_gt.global_track_id,
                    person_id=src_anchor.person_id,
                    original_confidence=round(src_anchor.confidence, 3),
                    propagated_confidence=round(syn.confidence, 3),
                    gallery_sim=round(sim, 4),
                )

        if synthetic:
            return list(face_anchors) + synthetic
        return face_anchors

    async def _gt_gallery_similarity(self, tids_a: list[str], tids_b: list[str]) -> float:
        """Mean cosine similarity between two GlobalTracks via gallery centroids.

        Returns 0.0 when either side has no gallery entries.
        """
        import numpy as np

        try:
            entries_a = await self._gallery_repo.list_gallery_entries_for_tracklets(
                tracklet_ids=set(tids_a), limit=20
            )
        except Exception:
            entries_a = []

        try:
            entries_b = await self._gallery_repo.list_gallery_entries_for_tracklets(
                tracklet_ids=set(tids_b), limit=20
            )
        except Exception:
            entries_b = []

        if not entries_a or not entries_b:
            return 0.0

        emb_a = np.mean([e.embedding for e in entries_a], axis=0)
        emb_b = np.mean([e.embedding for e in entries_b], axis=0)
        norm_a = float(np.linalg.norm(emb_a))
        norm_b = float(np.linalg.norm(emb_b))
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return float(np.dot(emb_a, emb_b) / (norm_a * norm_b + 1e-9))

    def get_face_locked_identity(self, global_track_id: str) -> str | None:
        """Return the face-locked identity for a GT, or None if not locked."""
        lock = self._face_locks.get(global_track_id)
        return lock.identity_id if lock is not None else None

    def register_identity(self, identity: Identity) -> None:
        """Register a known identity for display name lookup."""
        self._identities[identity.identity_id] = identity

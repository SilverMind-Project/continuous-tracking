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
from ..storage.base import GalleryRepository, GlobalTrackRepository, TrackingRepository

logger = get_logger(__name__)


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
    commit_margin: float = 0.25

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
        outcome = ResolveOutcome()

        for gt in global_tracks:
            prior = self._build_prior(gt, captured_at)
            face_likelihood = self._from_face_anchors(gt, new_face_anchors)
            reid_likelihood = await self._from_gallery(gt)
            posterior = self._combine(prior, face_likelihood, reid_likelihood)

            decision = self._commit(gt, posterior, face_likelihood, reid_likelihood, captured_at)
            if decision.revises_previous:
                revision = self._build_revision(
                    gt, decision, captured_at
                )
                if revision is not None:
                    outcome.revisions.append(revision)

            outcome.decisions.append(decision)

        return outcome

    # ------------------------------------------------------------------
    # Prior construction
    # ------------------------------------------------------------------

    def _build_prior(
        self, gt: GlobalTrack, captured_at: datetime
    ) -> PosteriorDist:
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
    ) -> PosteriorDist:
        """Build likelihood from face anchors associated with this GlobalTrack.

        Face anchors are the strongest evidence. A face anchor is associated
        with a GlobalTrack if its tracklet_id is in the tracklet's list.

        Returns a PosteriorDist with a point mass at the face-confirmed
        identity, scaled by p_face(confidence, quality).
        """
        gt_tracklet_ids = set(gt.tracklet_ids)

        # Find face anchors whose tracklet belongs to this GlobalTrack.
        relevant_anchors = [
            fa for fa in face_anchors if fa.tracklet_id in gt_tracklet_ids
        ]

        if not relevant_anchors:
            # No face evidence: return uniform (identity-neutral).
            return PosteriorDist({})

        # Take the strongest face anchor (highest confidence * quality).
        best = max(relevant_anchors, key=lambda fa: fa.confidence * fa.quality)

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
                    likelihood[cid] = per_id
                likelihood["UNKNOWN"] = per_id
            else:
                likelihood["UNKNOWN"] = remainder

        return PosteriorDist(likelihood)

    def _p_face(self, confidence: float, quality: float) -> float:
        """Probability that a face anchor is correct.

        Monotone function of confidence and quality, calibrated to
        typical face recognition performance.
        """
        combined = 0.7 * confidence + 0.3 * quality
        # Sigmoid-like: at combined=0.9 -> ~0.95, at combined=0.5 -> ~0.5
        return min(0.99, 0.5 + 0.5 * (2 * combined - 1))

    # ------------------------------------------------------------------
    # ReID likelihood from gallery
    # ------------------------------------------------------------------

    async def _from_gallery(self, gt: GlobalTrack) -> PosteriorDist:
        """Build likelihood from gallery k-NN search.

        Queries the gallery for similar embeddings to the tracklet's
        gallery entries. Maps results to per-identity scores using a
        calibrated logistic curve.
        """
        # In production, collect gallery embeddings for this GlobalTrack's
        # tracklets and use them for k-NN search. For now, query the full
        # gallery with a placeholder embedding.

        # Query the gallery for similar embeddings (limit to top-k per tracklet).
        # We use a simplified approach: search the full gallery and aggregate.
        try:
            similar = await self._gallery_repo.search_similar(
                embedding=[0.0] * 768,  # Placeholder — in production, use actual embeddings.
                limit=20,
            )
        except Exception:
            # Gallery unavailable: return uniform.
            return PosteriorDist({})

        if not similar:
            return PosteriorDist({})

        # Map hits to per-identity scores using logistic curve.
        likelihood: dict[str, list[float]] = defaultdict(list)
        for entry in similar:
            sim = _cosine_similarity_from_distance(entry.quality)
            # Logistic curve: midpoint at reid_decision_sim.
            logit = self._logistic(sim)
            key = entry.identity_id if entry.identity_id else "UNKNOWN"
            likelihood[key].append(logit)

        # Average scores per identity.
        avg: dict[str, float] = {}
        for key, scores in likelihood.items():
            avg[key] = sum(scores) / len(scores)

        if not avg:
            return PosteriorDist({})

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

        Posterior ∝ prior × face_likelihood × reid_likelihood

        If any source is empty (no evidence), it is treated as a
        near-zero weight so it does not overpower the prior.
        """
        all_ids: set[str] = set(prior.distribution.keys())
        all_ids.update(face.distribution.keys())
        all_ids.update(reid.distribution.keys())

        if not all_ids:
            return PosteriorDist({"UNKNOWN": 1.0})

        # Weight for empty distributions: "no evidence" should be neutral
        # (weight=1.0) so it does not dilute evidence from other sources.
        # This ensures that a strong prior or face anchor is not weakened
        # by empty likelihoods from other sources.
        default_weight = 1.0

        combined: dict[str, float] = {}
        for ident in all_ids:
            p = prior.distribution.get(ident, default_weight)
            f = face.distribution.get(ident, default_weight)
            r = reid.distribution.get(ident, default_weight)
            combined[ident] = p * f * r

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
    ) -> IdentityDecision:
        """Apply the commit rule to produce an identity decision.

        Commits an identity if:
        - top_probability >= commit_prob AND margin >= commit_margin
        - At least one non-prior evidence source (face or ReID) supports
          the top identity.  This prevents the temporal prior from
          committing an assignment on its own — a strong prior should
          maintain an existing assignment but not create one.

        Otherwise, the decision is UNKNOWN (None).
        """
        (top_id, top_prob), margin = posterior.top_with_margin()

        # Require evidence from at least one non-prior source.
        # The prior encodes temporal continuity (a track likely keeps
        # its current identity) but should not be the sole basis for
        # a new assignment.  Face anchors and ReID gallery hits are
        # the evidence that justifies an assignment or reassignment.
        has_evidence = (
            top_id in face_likelihood.distribution
            or top_id in reid_likelihood.distribution
        )

        # Apply commit rule.
        if (
            has_evidence
            and top_prob >= self._config.commit_prob
            and margin >= self._config.commit_margin
        ):
            new_id = top_id if top_id != "UNKNOWN" else None
        else:
            new_id = None  # Committed as UNKNOWN.

        prev_id = gt.current_identity_id
        revises = new_id != prev_id

        reason = ""
        if revises:
            if prev_id is None:
                reason = f"initial_assignment: {top_id} (p={top_prob:.3f})"
            elif new_id is None:
                reason = f"demoted_to_unknown: {top_id} (p={top_prob:.3f}, margin={margin:.3f})"
            else:
                reason = f"identity_change: {prev_id} -> {new_id} (p={top_prob:.3f}, margin={margin:.3f})"

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
        recent = [
            ts
            for ts in self._revision_log.get(gt.global_track_id, [])
            if ts.timestamp() >= window_start
        ]
        if len(recent) >= self._config.max_revisions_per_gt_per_minute:
            logger.warning(
                "Revision rate limit exceeded",
                global_track_id=gt.global_track_id,
                recent_count=len(recent),
            )
            return None

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
                display_name=self._identities.get(ident_id, Identity(
                    identity_id=ident_id,
                    display_name=ident_id,
                    enrolled_at=now,
                )).display_name,
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

    def register_identity(self, identity: Identity) -> None:
        """Register a known identity for display name lookup."""
        self._identities[identity.identity_id] = identity


def _cosine_similarity_from_distance(quality: float) -> float:
    """Convert a gallery entry's quality score to an approximate cosine similarity.

    In production, this would be the actual cosine similarity from the
    pgvector search. Here we use quality as a proxy.
    """
    return quality

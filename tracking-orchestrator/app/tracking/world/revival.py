"""PH revival: select a recently-closed PH to revive instead of spawning new.

Pure function, no I/O, no DB.  Tested independently like ``dedup.py``.
"""

from __future__ import annotations

from datetime import datetime

from ...domain import CameraTopologyEdge, PersonHypothesis, WorldObservation
from .config import WorldTrackerConfig
from .helpers import cosine_similarity
from .topology import plausible_transit


def select_revival_candidate(
    obs: WorldObservation,
    recent_closed: list[PersonHypothesis],
    now: datetime,
    cfg: WorldTrackerConfig,
    *,
    enable_cross_camera: bool = False,
    topology_edges: list[CameraTopologyEdge] | None = None,
) -> PersonHypothesis | None:
    """Pick the best recently-closed PH to revive for *obs*, or None.

    Selection rules (in order):

    Same-camera (unchanged):
    1. Same camera only.
    2. ``closed.closed_at`` within ``revive_max_age_s`` of *now*.
    3. Euclidean distance within ``revive_max_distance_m``.
    4. Appearance gate: cosine similarity >= ``revive_appearance_min_sim``.
       If either embedding is missing, fall back to space and time only.
    5. Identity-conflict hard gate: if *obs* has a face anchor naming a
       different identity at >= ``face_conflict_threshold``, reject.
    6. Among passing candidates, pick the highest appearance similarity,
       breaking ties by smallest distance.

    Cross-camera (gated by *enable_cross_camera*):
    1. Any camera (different camera allowed).
    2. ``closed.closed_at`` within ``revive_max_age_s`` (unchanged).
    3. Topology gate: ``plausible_transit(closed_camera, obs_camera, elapsed_s)``
       >= ``cross_camera_min_plausibility``.  Unseen edges get the floor.
    4. Appearance gate: max cosine similarity over the closed PH's
       ``view_prototypes`` vs ``obs.embedding`` >=
       ``cross_camera_revive_appearance_min_sim``.  Falls back to
       ``gallery_mean`` if view_prototypes is empty.
    5. Identity-conflict hard gate (unchanged from same-camera).
    6. Tiebreaking: highest appearance similarity, then smallest transit
       elapsed time.

    Returns None when no candidate passes all gates.
    """
    if not recent_closed:
        return None

    edges = topology_edges or []
    candidates: list[tuple[float, float, PersonHypothesis]] = []  # (sim, tiebreaker, ph)

    for closed in recent_closed:
        # 2. Age gate (shared by both paths).
        if closed.closed_at is None:
            continue
        age_s = (now - closed.closed_at).total_seconds()
        if age_s <= 0 or age_s > cfg.revive_max_age_s:
            continue

        is_cross_camera = closed.last_seen_camera != obs.camera_id

        if not is_cross_camera:
            # ---- Same-camera path ----

            # 3. Distance gate.
            cx, cy = closed.state_mean[0], closed.state_mean[1]
            ox = obs.floor_point.x_mm / 1000.0
            oy = obs.floor_point.y_mm / 1000.0
            dist = ((ox - cx) ** 2 + (oy - cy) ** 2) ** 0.5
            if dist > cfg.revive_max_distance_m:
                continue

            # 4. Appearance gate.
            sim = 0.0
            if closed.gallery_mean is not None and obs.embedding is not None:
                sim = cosine_similarity(closed.gallery_mean, obs.embedding)
                if sim < cfg.revive_appearance_min_sim:
                    continue
            # Missing embedding → fall back to space and time only (favor continuity).

            # 5. Identity-conflict hard gate.
            if _has_face_conflict(obs, closed, cfg):
                continue

            # 6. Passed all gates. Tie: similarity, then negative distance.
            candidates.append((sim, -dist, closed))

        elif enable_cross_camera:
            # ---- Cross-camera path ----

            elapsed_s = (now - closed.closed_at).total_seconds()

            # 3. Topology gate.
            transit_plaus = plausible_transit(
                closed.last_seen_camera, obs.camera_id, elapsed_s, edges
            )
            if transit_plaus < cfg.cross_camera_min_plausibility:
                continue

            # 4. Multi-view appearance gate.
            sim = _max_view_cosine(closed, obs.embedding)
            if sim < cfg.cross_camera_revive_appearance_min_sim:
                continue

            # 5. Identity-conflict hard gate (same as same-camera).
            if _has_face_conflict(obs, closed, cfg):
                continue

            # 6. Passed all gates. Tie: similarity, then negative elapsed.
            candidates.append((sim, -elapsed_s, closed))

        # else: cross-camera not enabled and camera differs → skip

    if not candidates:
        return None

    # Pick the highest similarity, breaking ties by secondary key.
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def _has_face_conflict(
    obs: WorldObservation,
    closed: PersonHypothesis,
    cfg: WorldTrackerConfig,
) -> bool:
    """Return True if *obs* has a recognized face anchor that conflicts with *closed*.

    Only recognized anchors (direct ArcFace matches) assert identity strongly
    enough to block revival.  Candidate and unrecognized anchors are weak
    positive evidence and must not trigger the conflict gate.
    """
    return bool(
        obs.face_anchor is not None
        and obs.face_anchor.recognition_state == "recognized"
        and obs.face_anchor.person_id
        and closed.current_identity_id
        and obs.face_anchor.person_id != closed.current_identity_id
        and obs.face_anchor.confidence >= cfg.face_conflict_threshold
    )


def _max_view_cosine(
    ph: PersonHypothesis,
    obs_embedding: list[float] | None,
) -> float:
    """Return the max cosine similarity between any of *ph*'s view prototypes
    and *obs_embedding*.

    Falls back to ``gallery_mean`` when ``view_prototypes`` is empty or
    *obs_embedding* is None.
    """
    if obs_embedding is None:
        return 0.0

    if ph.view_prototypes:
        best = 0.0
        for vp in ph.view_prototypes:
            sim = cosine_similarity(list(vp.embedding), obs_embedding)
            if sim > best:
                best = sim
        return best

    # Fall back to single gallery_mean.
    if ph.gallery_mean is not None:
        return cosine_similarity(ph.gallery_mean, obs_embedding)

    return 0.0

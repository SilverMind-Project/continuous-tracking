"""Cross-camera observation dedup pass.

Pre-association floor-point dedup: before the Hungarian assignment runs, collapse
observations from different cameras that are within ``cfg.dedup_max_distance_m``
of each other on the floor plane (and not in identity conflict) into one
representative observation. This prevents a single person observed simultaneously
by two overlapping cameras from spawning two Person Hypotheses.

The function is pure (no I/O, no DB). It is called by WorldTracker.step() between
building the observation lists and calling associate().
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from structlog import get_logger

if TYPE_CHECKING:
    from ...domain import FaceAnchor, WorldObservation
    from .config import WorldTrackerConfig

logger = get_logger(__name__)


@dataclass(frozen=True)
class DedupCluster:
    """One cluster produced by the dedup pass.

    ``representative`` is the single observation passed to ``associate()``.
    ``sources`` lists all detection IDs in the cluster (including the representative).
    The representative is always in ``sources``.
    """

    representative: WorldObservation
    sources: tuple[str, ...]  # detection_ids of all cluster members


def dedup_observations(
    observations: list[WorldObservation],
    cfg: WorldTrackerConfig,
) -> tuple[list[WorldObservation], dict[str, tuple[str, ...]]]:
    """Collapse same-floor, cross-camera observations into one representative each.

    Args:
        observations: raw observations for this frame (all cameras).
        cfg: WorldTrackerConfig; uses ``dedup_enabled``, ``dedup_max_distance_m``,
             ``dedup_require_no_face_conflict``.

    Returns:
        (deduped_observations, cluster_map) where:
        - ``deduped_observations`` replaces the original list (same or shorter length);
        - ``cluster_map`` maps each representative's ``detection_id`` to a tuple of ALL
          source ``detection_id``s in its cluster (including itself). Singleton clusters
          have a one-element tuple containing their own detection_id.
    """
    if not cfg.dedup_enabled or len(observations) < 2:
        singleton_map: dict[str, tuple[str, ...]] = {
            obs.detection_id: (obs.detection_id,) for obs in observations if obs.detection_id
        }
        return list(observations), singleton_map

    n = len(observations)
    # Union-Find to identify connected components.
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        obs_i = observations[i]
        if not obs_i.floor_point.calibrated:
            continue
        for j in range(i + 1, n):
            obs_j = observations[j]
            if not obs_j.floor_point.calibrated:
                continue
            # Rule 1: different cameras only.
            if obs_i.camera_id == obs_j.camera_id:
                continue
            # Rule 2: within residual-aware geometric gate.
            xi = obs_i.floor_point.x_mm / 1000.0
            yi = obs_i.floor_point.y_mm / 1000.0
            xj = obs_j.floor_point.x_mm / 1000.0
            yj = obs_j.floor_point.y_mm / 1000.0
            dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            effective_gate = _effective_distance_gate_m(obs_i, obs_j, cfg)
            if dist > effective_gate:
                continue
            # Rule 3: no identity conflict if both have committed face ids.
            if cfg.dedup_require_no_face_conflict:
                fa_i: FaceAnchor | None = obs_i.face_anchor
                fa_j: FaceAnchor | None = obs_j.face_anchor
                if (
                    fa_i is not None
                    and fa_j is not None
                    and fa_i.person_id is not None
                    and fa_j.person_id is not None
                    and fa_i.person_id != fa_j.person_id
                ):
                    continue
            _union(i, j)

    # Build clusters from the union-find roots.
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        root = _find(i)
        clusters.setdefault(root, []).append(i)

    deduped: list[WorldObservation] = []
    cluster_map: dict[str, tuple[str, ...]] = {}

    for _root, members in clusters.items():
        if len(members) == 1:
            obs = observations[members[0]]
            deduped.append(obs)
            single: tuple[str, ...] = (obs.detection_id,)
            cluster_map[obs.detection_id] = single
        else:
            # Multi-observation cluster: collapse to one representative.
            cluster_obs = [observations[m] for m in members]
            rep = _select_representative(cluster_obs)
            rep = _build_representative(rep, cluster_obs)
            deduped.append(rep)
            src_ids = tuple(obs.detection_id for obs in cluster_obs)
            cluster_map[rep.detection_id] = src_ids

            n_collapsed = len(members) - 1
            logger.debug(
                "dedup_cluster_formed",
                representative_camera=rep.camera_id,
                cluster_size=len(members),
                cameras=[o.camera_id for o in cluster_obs],
            )
            # Metrics are incremented by the caller (WorldTracker) so it can
            # access the prometheus registry without importing from here.
            _ = n_collapsed  # surfaced to caller via cluster_map length

    return deduped, cluster_map


def _select_representative(cluster: list[WorldObservation]) -> WorldObservation:
    """Return the highest-quality observation; break ties by (camera_id, detection_id) ascending."""
    return min(
        cluster,
        key=lambda obs: (-obs.quality, obs.camera_id, obs.detection_id),
    )


def _effective_distance_gate_m(
    obs_i: WorldObservation,
    obs_j: WorldObservation,
    cfg: WorldTrackerConfig,
) -> float:
    """Return the pairwise dedup gate widened by calibration residuals."""
    residual_i = obs_i.floor_residual_m or 0.0
    residual_j = obs_j.floor_residual_m or 0.0
    widened = cfg.dedup_max_distance_m + cfg.dedup_residual_coeff_k * (residual_i + residual_j)
    return min(widened, cfg.dedup_max_distance_ceiling_m)


def _build_representative(
    best: WorldObservation,
    cluster: list[WorldObservation],
) -> WorldObservation:
    """Build the dedup representative from the best observation in a cluster.

    Floor position: quality-weighted mean of calibrated floor points.
    Embedding: taken from the best (highest-quality) observation unchanged.
    Face anchor: highest-confidence face anchor in the cluster.
    active_cameras: the representative itself carries camera_id; the tracker
                    adds all cameras via cluster_map during PH update.
    """
    import dataclasses

    from ...domain import FloorPoint

    # Quality-weighted mean floor point.
    total_weight = 0.0
    wx = 0.0
    wy = 0.0
    missing_fp = 0
    for obs in cluster:
        if not obs.floor_point.calibrated:
            missing_fp += 1
            continue
        w = obs.quality if obs.quality > 0.0 else 1e-6
        wx += obs.floor_point.x_mm / 1000.0 * w
        wy += obs.floor_point.y_mm / 1000.0 * w
        total_weight += w

    if missing_fp > 0:
        # Caller (WorldTracker) increments the metric; we just note the count.
        pass

    if total_weight > 0.0:
        mean_x_m = wx / total_weight
        mean_y_m = wy / total_weight
        mean_fp = FloorPoint(
            x_mm=round(mean_x_m * 1000.0),
            y_mm=round(mean_y_m * 1000.0),
            calibrated=True,
        )
    else:
        mean_fp = best.floor_point

    # Best face anchor in the cluster.
    best_face = best.face_anchor
    for obs in cluster:
        if obs.face_anchor is not None and (
            best_face is None or obs.face_anchor.confidence > best_face.confidence
        ):
            best_face = obs.face_anchor

    return dataclasses.replace(
        best,
        floor_point=mean_fp,
        face_anchor=best_face,
    )

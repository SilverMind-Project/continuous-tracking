"""Camera adjacency topology model: co-occurrence statistics and plausibility scoring.

Pure functions with no I/O.  The caller fetches topology edges from the
repository and passes them in.  This keeps the module testable without mocks
and matches the existing patterns in ``revival.py`` and ``dedup.py``.
"""

from __future__ import annotations

import math
from datetime import datetime

from ...domain import CameraTopologyEdge

# Minimum observations before a learned distribution may reject a transit.
MIN_OBSERVATIONS_FOR_LEARNED = 3

# Lower bound on the learned spread. A handful of handoffs at nearly the same speed would
# otherwise give a spread of a few tenths of a second and reject an ordinary brisk walk.
_MIN_TRANSIT_STD_S = 1.0


def plausible_transit(
    from_camera: str,
    to_camera: str,
    elapsed_s: float,
    edges: list[CameraTopologyEdge],
) -> float:
    """Return how plausible one person crossing *from_camera* to *to_camera* in *elapsed_s* is.

    The score is in [0, 1] and is compared against ``cross_camera_min_plausibility``.

    - An unseen camera pair, or one with fewer than ``MIN_OBSERVATIONS_FOR_LEARNED`` handoffs,
      returns 1.0: there is no evidence against the transit, so the first handoffs always pass
      and the topology can be learned.
    - A transit at or slower than the learned mean returns 1.0. People stop on the way between
      rooms, so a slow transit is not evidence of a different person; the revival age limit
      bounds how slow it can be.
    - A transit faster than the learned mean decays as a one-sided Gaussian, so a crossing that
      is physically too fast for the learned walk scores near 0 and is rejected.
    """
    edge = _find_edge(from_camera, to_camera, edges)
    if edge is None or edge.observation_count < MIN_OBSERVATIONS_FOR_LEARNED:
        return 1.0
    if elapsed_s >= edge.mean_transit_s:
        return 1.0

    std = max(math.sqrt(max(edge.variance_transit_s2, 0.0)), _MIN_TRANSIT_STD_S)
    z = (edge.mean_transit_s - elapsed_s) / std
    return math.exp(-0.5 * z * z)


def record_handoff(
    from_camera: str,
    to_camera: str,
    elapsed_s: float,
    edges: list[CameraTopologyEdge],
    now: datetime,
) -> CameraTopologyEdge:
    """Record one observed handoff and return the updated ``CameraTopologyEdge``.

    Uses the Welford online algorithm for numerically stable single-pass mean
    and variance updates.  The caller should upsert the returned edge into the
    repository.
    """
    existing = _find_edge(from_camera, to_camera, edges)
    if existing is None:
        return CameraTopologyEdge(
            from_camera=from_camera,
            to_camera=to_camera,
            observation_count=1,
            mean_transit_s=elapsed_s,
            variance_transit_s2=0.0,
            last_updated_at=now,
        )

    n = existing.observation_count
    old_mean = existing.mean_transit_s
    old_var = existing.variance_transit_s2

    # Welford online update.
    new_n = n + 1
    delta = elapsed_s - old_mean
    new_mean = old_mean + delta / new_n
    delta2 = elapsed_s - new_mean
    new_var = (old_var * n + delta * delta2) / new_n if new_n > 1 else 0.0

    return CameraTopologyEdge(
        from_camera=from_camera,
        to_camera=to_camera,
        observation_count=new_n,
        mean_transit_s=new_mean,
        variance_transit_s2=new_var,
        last_updated_at=now,
    )


def _find_edge(
    from_camera: str, to_camera: str, edges: list[CameraTopologyEdge]
) -> CameraTopologyEdge | None:
    """Linear scan for the directed edge (list size is small -- O(n_camera_pairs))."""
    for edge in edges:
        if edge.from_camera == from_camera and edge.to_camera == to_camera:
            return edge
    return None

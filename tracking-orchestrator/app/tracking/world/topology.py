"""Camera adjacency topology model: co-occurrence statistics and plausibility scoring.

Pure functions with no I/O.  The caller fetches topology edges from the
repository and passes them in.  This keeps the module testable without mocks
and matches the existing patterns in ``revival.py`` and ``dedup.py``.
"""

from __future__ import annotations

import math
from datetime import datetime

from ...domain import CameraTopologyEdge

# Default floor for unseen edges: small enough to allow first handoff,
# large enough that implausible long jumps are filtered once data accumulates.
_DEFAULT_PLAUSIBILITY_FLOOR = 0.05

# Number of standard deviations at which the Gaussian score decays to 1/e
# (roughly ~0.37; below the floor so effectively floor-clipped).
_PLAUSIBILITY_SIGMA = 3.0

# Minimum observations before the learned distribution is trusted over the floor.
_MIN_OBSERVATIONS_FOR_LEARNED = 3


def plausible_transit(
    from_camera: str,
    to_camera: str,
    elapsed_s: float,
    edges: list[CameraTopologyEdge],
    *,
    floor: float = _DEFAULT_PLAUSIBILITY_FLOOR,
) -> float:
    """Return transit plausibility [0, 1] for a camera pair given *elapsed_s*.

    For unseen edges or edges with fewer than ``_MIN_OBSERVATIONS_FOR_LEARNED``
    samples, returns *floor* (default 0.05) so the first few handoffs always
    pass.  For well-observed edges, returns a Gaussian-shaped score centred on
    the learned mean transit time, clipped to *floor*.

    The caller is expected to have loaded the edge list from the repository
    before calling this function.
    """
    edge = _find_edge(from_camera, to_camera, edges)
    if edge is None or edge.observation_count < _MIN_OBSERVATIONS_FOR_LEARNED:
        return floor

    std = math.sqrt(max(edge.variance_transit_s2, 1e-6))
    z = (elapsed_s - edge.mean_transit_s) / std
    gaussian = math.exp(-0.5 * z * z)
    return max(floor, gaussian)


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

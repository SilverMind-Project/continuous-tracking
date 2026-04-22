"""Camera adjacency graph for cross-camera association.

Maintains a directed graph of camera reachability with optional time-based
constraints. Used by the CrossCameraAssociator to filter impossible
tracklet pairings (e.g., a person cannot teleport from camera A to camera C
without passing through camera B first).

The adjacency graph supports:
- Static reachability (A -> B means camera B is reachable from A).
- Time-bounded reachability (A -> B is only valid within N seconds of B's
  last_seen_at, since a person might have left the house).
- Overlap detection (A and B share a field of view).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..domain import CameraId


@dataclass(frozen=True)
class AdjacencyEdge:
    """One directed edge in the camera adjacency graph."""

    from_camera: CameraId
    to_camera: CameraId
    max_transition_seconds: float = 300.0  # 5 minutes default
    overlap: bool = False


class CameraAdjacency:
    """Directed graph of camera reachability.

    Usage::

        adj = CameraAdjacency()
        adj.add_edge("cam_kitchen", "cam_hallway", max_transition_seconds=120)
        adj.add_edge("cam_hallway", "cam_bedroom", max_transition_seconds=60)

        # Check reachability
        assert adj.reachable("cam_kitchen", "cam_hallway")
        assert adj.reachable("cam_kitchen", "cam_bedroom")  # transitive
        assert not adj.reachable("cam_bedroom", "cam_kitchen")  # not directed

    The graph is built once at startup from configuration and never mutated
    during normal operation. Calibration reloads replace the entire graph.
    """

    def __init__(self) -> None:
        # Adjacency list: from_camera -> list of (to_camera, edge)
        self._edges: dict[CameraId, list[AdjacencyEdge]] = {}
        # Overlap set: frozenset({a, b}) -> True
        self._overlaps: set[frozenset[str]] = set()

    def add_edge(self, edge: AdjacencyEdge) -> None:
        """Add a directed edge to the adjacency graph."""
        self._edges.setdefault(edge.from_camera, []).append(edge)
        if edge.overlap:
            self._overlaps.add(frozenset({edge.from_camera, edge.to_camera}))

    def remove_camera(self, camera_id: CameraId) -> None:
        """Remove all edges involving a camera."""
        self._edges.pop(camera_id, None)
        self._overlaps = {s for s in self._overlaps if camera_id not in s}
        # Remove edges pointing to this camera
        for camera_id_key in list(self._edges):
            self._edges[camera_id_key] = [
                e for e in self._edges[camera_id_key] if e.to_camera != camera_id
            ]

    def reachable(
        self,
        from_camera: CameraId,
        to_camera: CameraId,
        within_s: float | None = None,
    ) -> bool:
        """Check if to_camera is reachable from from_camera.

        Uses BFS for transitive reachability. If within_s is provided,
        only edges whose max_transition_seconds fits within the budget are
        considered (i.e., the transition must be fast enough).

        Args:
            from_camera: source camera ID.
            to_camera: destination camera ID.
            within_s: optional time budget in seconds. If set, only edges
                with max_transition_seconds <= within_s pass the filter.

        Returns:
            True if to_camera is reachable from from_camera.
        """
        if from_camera == to_camera:
            return True

        visited: set[CameraId] = set()
        queue: list[CameraId] = [from_camera]

        while queue:
            current = queue.pop(0)
            if current == to_camera:
                return True
            if current in visited:
                continue
            visited.add(current)

            for edge in self._edges.get(current, []):
                if edge.to_camera in visited:
                    continue
                if within_s is not None and edge.max_transition_seconds > within_s:
                    continue
                queue.append(edge.to_camera)

        return False

    def overlap(self, camera_a: CameraId, camera_b: CameraId) -> bool:
        """Check if two cameras share a field of view."""
        return frozenset({camera_a, camera_b}) in self._overlaps

    def get_neighbors(self, camera_id: CameraId) -> list[CameraId]:
        """Get all cameras reachable from the given camera (one hop)."""
        return [edge.to_camera for edge in self._edges.get(camera_id, [])]

    def get_max_transition(self, from_camera: CameraId, to_camera: CameraId) -> float | None:
        """Get the maximum transition time between two adjacent cameras.

        Returns None if the cameras are not directly connected.
        """
        for edge in self._edges.get(from_camera, []):
            if edge.to_camera == to_camera:
                return edge.max_transition_seconds
        return None

    def get_all_edges(self) -> list[AdjacencyEdge]:
        """Return all edges in the graph."""
        all_edges: list[AdjacencyEdge] = []
        for edges in self._edges.values():
            all_edges.extend(edges)
        return all_edges

    def has_camera(self, camera_id: CameraId) -> bool:
        """Check if the graph knows about a camera."""
        return camera_id in self._edges or any(
            edge.to_camera == camera_id for edges in self._edges.values() for edge in edges
        )

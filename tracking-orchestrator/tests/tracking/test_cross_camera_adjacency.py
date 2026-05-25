"""Unit tests for bidirectional adjacency checks in cross-camera association."""

from __future__ import annotations

from app.tracking.camera_adjacency import AdjacencyEdge, CameraAdjacency


def _adj_with_edges(*edges: AdjacencyEdge) -> CameraAdjacency:
    adj = CameraAdjacency()
    for edge in edges:
        adj.add_edge(edge)
    return adj


class TestReachableOneDirection:
    def test_forward_edge_is_reachable(self) -> None:
        adj = _adj_with_edges(
            AdjacencyEdge(from_camera="cam-a", to_camera="cam-b", max_transition_seconds=30),
        )
        assert adj.reachable("cam-a", "cam-b", within_s=60)

    def test_reverse_direction_not_reachable_without_edge(self) -> None:
        adj = _adj_with_edges(
            AdjacencyEdge(from_camera="cam-a", to_camera="cam-b", max_transition_seconds=30),
        )
        assert not adj.reachable("cam-b", "cam-a", within_s=60)


class TestReachableBidirectional:
    def test_both_directions_when_both_edges_configured(self) -> None:
        adj = _adj_with_edges(
            AdjacencyEdge(from_camera="cam-a", to_camera="cam-b", max_transition_seconds=30),
            AdjacencyEdge(from_camera="cam-b", to_camera="cam-a", max_transition_seconds=30),
        )
        assert adj.reachable("cam-a", "cam-b", within_s=60)
        assert adj.reachable("cam-b", "cam-a", within_s=60)


class TestReachableTimeBudget:
    def test_reachable_when_path_cost_within_budget(self) -> None:
        adj = _adj_with_edges(
            AdjacencyEdge(from_camera="cam-a", to_camera="cam-b", max_transition_seconds=10),
            AdjacencyEdge(from_camera="cam-b", to_camera="cam-c", max_transition_seconds=20),
        )
        assert adj.reachable("cam-a", "cam-c", within_s=30)

    def test_not_reachable_when_path_cost_exceeds_budget(self) -> None:
        adj = _adj_with_edges(
            AdjacencyEdge(from_camera="cam-a", to_camera="cam-b", max_transition_seconds=30),
            AdjacencyEdge(from_camera="cam-b", to_camera="cam-c", max_transition_seconds=30),
        )
        assert not adj.reachable("cam-a", "cam-c", within_s=40)


class TestReachableSameCamera:
    def test_same_camera_always_reachable(self) -> None:
        adj = CameraAdjacency()
        assert adj.reachable("cam-a", "cam-a")
        assert adj.reachable("cam-a", "cam-a", within_s=10)


class TestReachableNoEdges:
    def test_not_reachable_without_any_edges(self) -> None:
        adj = CameraAdjacency()
        assert not adj.reachable("cam-a", "cam-b")

    def test_not_reachable_without_any_edges_even_with_budget(self) -> None:
        adj = CameraAdjacency()
        assert not adj.reachable("cam-a", "cam-b", within_s=999)

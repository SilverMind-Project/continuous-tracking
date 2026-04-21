"""Tests for CameraAdjacency."""

from __future__ import annotations

from app.tracking.camera_adjacency import AdjacencyEdge, CameraAdjacency


class TestCameraAdjacency:
    """Test the CameraAdjacency graph."""

    def test_add_edge(self) -> None:
        adj = CameraAdjacency()
        adj.add_edge(AdjacencyEdge("cam_a", "cam_b", max_transition_seconds=120))
        assert adj.reachable("cam_a", "cam_b")

    def test_reachable_reflexive(self) -> None:
        adj = CameraAdjacency()
        assert adj.reachable("cam_a", "cam_a")

    def test_reachable_directed(self) -> None:
        adj = CameraAdjacency()
        adj.add_edge(AdjacencyEdge("cam_a", "cam_b"))
        assert adj.reachable("cam_a", "cam_b")
        assert not adj.reachable("cam_b", "cam_a")

    def test_reachable_transitive(self) -> None:
        adj = CameraAdjacency()
        adj.add_edge(AdjacencyEdge("cam_a", "cam_b"))
        adj.add_edge(AdjacencyEdge("cam_b", "cam_c"))
        assert adj.reachable("cam_a", "cam_c")
        assert not adj.reachable("cam_c", "cam_a")

    def test_reachable_time_bounded(self) -> None:
        adj = CameraAdjacency()
        adj.add_edge(AdjacencyEdge("cam_a", "cam_b", max_transition_seconds=60))
        # Within budget: 30s < 60s
        assert adj.reachable("cam_a", "cam_b", within_s=30)
        # Exceeds budget: 120s > 60s
        assert not adj.reachable("cam_a", "cam_b", within_s=120)

    def test_overlap(self) -> None:
        adj = CameraAdjacency()
        adj.add_edge(AdjacencyEdge("cam_a", "cam_b", overlap=True))
        assert adj.overlap("cam_a", "cam_b")
        assert adj.overlap("cam_b", "cam_a")
        assert not adj.overlap("cam_a", "cam_c")

    def test_get_neighbors(self) -> None:
        adj = CameraAdjacency()
        adj.add_edge(AdjacencyEdge("cam_a", "cam_b"))
        adj.add_edge(AdjacencyEdge("cam_a", "cam_c"))
        neighbors = adj.get_neighbors("cam_a")
        assert set(neighbors) == {"cam_b", "cam_c"}

    def test_get_max_transition(self) -> None:
        adj = CameraAdjacency()
        adj.add_edge(AdjacencyEdge("cam_a", "cam_b", max_transition_seconds=120))
        assert adj.get_max_transition("cam_a", "cam_b") == 120
        assert adj.get_max_transition("cam_a", "cam_c") is None

    def test_remove_camera(self) -> None:
        adj = CameraAdjacency()
        adj.add_edge(AdjacencyEdge("cam_a", "cam_b"))
        adj.add_edge(AdjacencyEdge("cam_b", "cam_c"))
        adj.remove_camera("cam_b")
        assert not adj.reachable("cam_a", "cam_b")
        assert not adj.reachable("cam_b", "cam_c")

    def test_get_all_edges(self) -> None:
        adj = CameraAdjacency()
        adj.add_edge(AdjacencyEdge("cam_a", "cam_b"))
        adj.add_edge(AdjacencyEdge("cam_b", "cam_c"))
        edges = adj.get_all_edges()
        assert len(edges) == 2

    def test_has_camera(self) -> None:
        adj = CameraAdjacency()
        adj.add_edge(AdjacencyEdge("cam_a", "cam_b"))
        assert adj.has_camera("cam_a")
        assert adj.has_camera("cam_b")
        assert not adj.has_camera("cam_c")

    def test_multiple_edges_same_pair(self) -> None:
        adj = CameraAdjacency()
        adj.add_edge(AdjacencyEdge("cam_a", "cam_b"))
        adj.add_edge(AdjacencyEdge("cam_a", "cam_b"))
        edges = adj.get_all_edges()
        assert len(edges) == 2

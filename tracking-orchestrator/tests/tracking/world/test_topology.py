"""Camera adjacency topology model: unit tests for pure functions."""

from __future__ import annotations

from datetime import UTC, datetime

from app.domain import CameraTopologyEdge
from app.tracking.world.topology import plausible_transit, record_handoff

_NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)


def test_unseen_edge_returns_floor() -> None:
    """plausible_transit returns the configured floor for an unseen edge."""
    p = plausible_transit("cam-a", "cam-b", 5.0, [])
    assert abs(p - 0.05) < 1e-6


def test_unseen_edge_returns_floor_with_data() -> None:
    """plausible_transit returns floor for an unseen edge even when other edges exist."""
    edge = CameraTopologyEdge(
        from_camera="cam-a",
        to_camera="cam-b",
        observation_count=10,
        mean_transit_s=3.0,
        variance_transit_s2=1.0,
    )
    # Different camera pair -> unseen.
    p = plausible_transit("cam-x", "cam-y", 5.0, [edge])
    assert abs(p - 0.05) < 1e-6


def test_few_samples_edge_returns_floor() -> None:
    """An edge with fewer than 3 observations returns the floor."""
    edge = CameraTopologyEdge(
        from_camera="cam-a",
        to_camera="cam-b",
        observation_count=2,
        mean_transit_s=3.0,
        variance_transit_s2=0.5,
    )
    p = plausible_transit("cam-a", "cam-b", 5.0, [edge])
    assert abs(p - 0.05) < 1e-6


def test_well_observed_edge_peaks_near_mean() -> None:
    """plausible_transit returns near 1.0 at the learned mean and decays away."""
    edge = CameraTopologyEdge(
        from_camera="cam-a",
        to_camera="cam-b",
        observation_count=10,
        mean_transit_s=5.0,
        variance_transit_s2=1.0,
    )
    p_at_mean = plausible_transit("cam-a", "cam-b", 5.0, [edge])
    p_off = plausible_transit("cam-a", "cam-b", 20.0, [edge])
    assert p_at_mean > 0.9, "should peak near 1.0 at the mean"
    assert p_off <= 0.05, "should decay to floor far from mean"


def test_custom_floor_respected() -> None:
    """The floor parameter controls the minimum returned value."""
    p = plausible_transit("cam-a", "cam-b", 5.0, [], floor=0.10)
    assert abs(p - 0.10) < 1e-6


def test_record_handoff_first_observation() -> None:
    """record_handoff creates a new edge with count=1 and mean=elapsed."""
    edge = record_handoff("cam-a", "cam-b", 5.0, [], _NOW)
    assert edge.observation_count == 1
    assert edge.mean_transit_s == 5.0
    assert edge.variance_transit_s2 == 0.0
    assert edge.last_updated_at == _NOW


def test_record_handoff_welford_update() -> None:
    """record_handoff updates count, mean, and variance correctly."""
    now = datetime.now(UTC)
    # First observation: 5.0
    edge = record_handoff("cam-a", "cam-b", 5.0, [], now)
    assert edge.observation_count == 1
    assert edge.mean_transit_s == 5.0

    # Second observation: 7.0 -> new mean = 6.0
    edge = record_handoff("cam-a", "cam-b", 7.0, [edge], now)
    assert edge.observation_count == 2
    assert abs(edge.mean_transit_s - 6.0) < 1e-6
    # variance after 2 obs: ((1*0 + (5-6)*(7-6))/2) = 1.0
    assert edge.variance_transit_s2 > 0.0

    # Third observation: 3.0 -> new mean = 5.0
    edge = record_handoff("cam-a", "cam-b", 3.0, [edge], now)
    assert edge.observation_count == 3
    assert abs(edge.mean_transit_s - 5.0) < 1e-6


def test_record_handoff_creates_separate_edge_for_reverse_direction() -> None:
    """Cam-A→Cam-B and Cam-B→Cam-A are distinct edges."""
    edge_ab = record_handoff("cam-a", "cam-b", 5.0, [], _NOW)
    edge_ba = record_handoff("cam-b", "cam-a", 3.0, [edge_ab], _NOW)
    assert edge_ba.observation_count == 1
    assert edge_ba.mean_transit_s == 3.0
    assert edge_ab.observation_count == 1  # unchanged


def test_zero_variance_does_not_divide_by_zero() -> None:
    """plausible_transit handles zero variance without raising."""
    edge = CameraTopologyEdge(
        from_camera="cam-a",
        to_camera="cam-b",
        observation_count=10,
        mean_transit_s=5.0,
        variance_transit_s2=0.0,
    )
    # Should not raise.
    p = plausible_transit("cam-a", "cam-b", 5.0, [edge])
    assert p > 0.9  # at the exact mean, Gaussian = 1.0

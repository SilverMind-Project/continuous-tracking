"""Tests for device-free Chi-square dedup and camera/sensor non-merging rules."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.domain import BoundingBox, FloorPoint, ObservationSource, WorldObservation
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.dedup import dedup_observations

_T0 = datetime(2026, 11, 25, 10, 0, 0, tzinfo=UTC)


def _cam_obs(
    camera_id: str,
    det_id: str,
    x_m: float,
    y_m: float,
    *,
    cov: tuple[float, float, float, float] = (0.04, 0.0, 0.0, 0.04),
    residual_m: float = 0.05,
) -> WorldObservation:
    return WorldObservation(
        camera_id=camera_id,
        frame_index=1,
        captured_at=_T0,
        floor_point=FloorPoint(x_mm=round(x_m * 1000.0), y_mm=round(y_m * 1000.0), calibrated=True),
        bbox=BoundingBox(x_min=10.0, y_min=10.0, x_max=50.0, y_max=100.0),
        embedding=[0.1] * 8,
        detection_confidence=0.95,
        detection_id=det_id,
        quality=0.8,
        floor_residual_m=residual_m,
        floor_cov_random=cov,
        source=ObservationSource.CAMERA,
    )


def _sensor_obs(
    sensor_id: str,
    det_id: str,
    x_m: float,
    y_m: float,
    *,
    cov: tuple[float, float, float, float] | None = (0.04, 0.0, 0.0, 0.04),
    residual_m: float = 0.10,
) -> WorldObservation:
    return WorldObservation(
        camera_id=f"sensor:{sensor_id}",
        frame_index=1,
        captured_at=_T0,
        floor_point=FloorPoint(x_mm=round(x_m * 1000.0), y_mm=round(y_m * 1000.0), calibrated=True),
        bbox=None,
        embedding=[],
        detection_confidence=1.0,
        detection_id=det_id,
        quality=0.0,
        floor_residual_m=residual_m,
        floor_cov_random=cov,
        source=ObservationSource.DEVICE_FREE,
    )


def test_two_device_free_sensors_merge_under_chi2_and_propagate_worst_residual() -> None:
    cfg = WorldTrackerConfig(dedup_max_distance_m=0.4, dedup_device_free_chi2=9.21, k_cal=1.0)
    # 0.85m apart (exceeds Euclidean distance gate, inside 2-DoF chi2=9.21 under var=0.25 each).
    s1 = _sensor_obs("radar-a", "d1", 2.0, 3.0, cov=(0.25, 0.0, 0.0, 0.25), residual_m=0.08)
    s2 = _sensor_obs("lidar-b", "d2", 2.85, 3.0, cov=(0.25, 0.0, 0.0, 0.25), residual_m=0.20)

    deduped, cluster_map = dedup_observations([s1, s2], cfg)
    assert len(deduped) == 1
    rep = deduped[0]
    assert set(cluster_map[rep.detection_id]) == {"d1", "d2"}
    assert rep.floor_residual_m == pytest.approx(0.20)
    assert rep.floor_cov_random is not None
    # Fused total variance < standalone total (0.29) and >= bias floor (0.04).
    assert 0.04 <= rep.floor_cov_random[0] < 0.20


def test_same_sensor_and_camera_sensor_never_merge_and_camera_pair_unchanged() -> None:
    cfg = WorldTrackerConfig(dedup_max_distance_m=0.6, dedup_device_free_chi2=9.21)

    same_sensor = [
        _sensor_obs("radar-a", "d1", 2.0, 3.0),
        _sensor_obs("radar-a", "d2", 2.0, 3.0),
    ]
    out_same, _ = dedup_observations(same_sensor, cfg)
    assert len(out_same) == 2

    mixed = [
        _cam_obs("cam-1", "c1", 2.0, 3.0),
        _sensor_obs("radar-a", "s1", 2.0, 3.0),
    ]
    out_mixed, _ = dedup_observations(mixed, cfg)
    assert len(out_mixed) == 2

    cams = [
        _cam_obs("cam-1", "c1", 2.0, 3.0),
        _cam_obs("cam-2", "c2", 2.2, 3.0),
    ]
    out_cams, cluster_map = dedup_observations(cams, cfg)
    assert len(out_cams) == 1
    assert set(cluster_map[out_cams[0].detection_id]) == {"c1", "c2"}


def test_device_free_missing_or_singular_covariance_fails_closed() -> None:
    cfg = WorldTrackerConfig()
    no_cov = _sensor_obs("radar-a", "d1", 2.0, 3.0, cov=None)
    good = _sensor_obs("lidar-b", "d2", 2.0, 3.0, cov=(0.04, 0.0, 0.0, 0.04))
    singular = _sensor_obs("lidar-c", "d3", 2.0, 3.0, cov=(0.0, 0.0, 0.0, 0.0), residual_m=0.0)

    out, _ = dedup_observations([no_cov, good, singular], cfg)
    assert len(out) == 3

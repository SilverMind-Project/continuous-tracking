"""Unit tests for PH running height inverse-variance fusion, pose cache, and pipeline stage."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from app.calibration.state import CalibrationState
from app.domain import BoundingBox, CameraProfile, Detection, FloorPoint
from app.observability import metrics as _metrics
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.height_estimation import (
    CameraPoseCache,
    HeightEstimationConfig,
    HeightEstimationStage,
)
from app.services.camera_profile_map import CameraProfileMap
from app.tracking.camera_pose import CameraPoseUnreliableReason, update_ph_height_estimate
from app.transport.redis_streams import FrameReady
from tests.unit.tracking.test_camera_pose import _synthetic_camera
from tests.unit.tracking.test_standing_height import _make_pose_result, _project_point


def test_update_ph_height_estimate_inverse_variance_fusion() -> None:
    # First observation initializes PH state.
    h1, s1, ok1 = update_ph_height_estimate(None, None, 1.62, 0.08)
    assert ok1 is True
    assert h1 == pytest.approx(1.62)
    assert s1 == pytest.approx(0.08)

    # Second observation with smaller sigma pulls estimate closer to it and shrinks sigma.
    h2, s2, ok2 = update_ph_height_estimate(h1, s1, 1.66, 0.04)
    assert ok2 is True
    expected_prec = 1.0 / (0.08**2) + 1.0 / (0.04**2)
    expected_h = (1.62 / (0.08**2) + 1.66 / (0.04**2)) / expected_prec
    expected_s = math.sqrt(1.0 / expected_prec)
    assert h2 == pytest.approx(expected_h)
    assert s2 == pytest.approx(expected_s)
    assert s2 is not None and s2 < 0.04


def test_update_ph_height_estimate_innovation_gate_rejects_sitting_outlier() -> None:
    """A 0.95 m sitting-like outlier is > 3 sigma away from 1.65 ± 0.04 m and must be rejected."""
    h, s, accepted = update_ph_height_estimate(1.65, 0.04, 0.95, 0.06)
    assert accepted is False
    assert h == pytest.approx(1.65)
    assert s == pytest.approx(0.04)


def test_update_ph_height_estimate_floors_sigma() -> None:
    h, s, accepted = update_ph_height_estimate(1.70, 0.025, 1.70, 0.025, min_sigma_m=0.02)
    assert accepted is True
    assert h == pytest.approx(1.70)
    assert s == pytest.approx(0.02)


@pytest.mark.asyncio
async def test_camera_pose_cache_counts_once_per_epoch_and_revalidates_on_update() -> None:
    cal = CalibrationState()
    h_i2f, _, _, _, _ = _synthetic_camera(centre_m=(0.0, 0.0, 2.4), tilt_deg=-65.0, fov_deg=75.0)
    await cal.set_homography("cam-a", h_i2f.tolist())

    profiles = CameraProfileMap()
    await profiles.set_all(
        [
            CameraProfile(
                camera_id="cam-a",
                horizontal_fov_deg=95.0,  # wrong FOV -> height out of tolerance
                mounting_height_m=2.4,
                tilt_deg=-65.0,
            )
        ]
    )

    cache = CameraPoseCache(calibration_state=cal, camera_profiles=profiles)
    counter = _metrics.metrics.camera_height_pose_unreliable_total.labels(
        reason=CameraPoseUnreliableReason.HEIGHT_OUT_OF_TOLERANCE.value
    )
    before = counter._value.get()

    v1 = cache.get("cam-a", width_px=1920, height_px=1080)
    v2 = cache.get("cam-a", width_px=1920, height_px=1080)
    assert v1.reliable is False
    assert v2 is v1  # served from cache
    assert counter._value.get() == before + 1.0  # incremented once, not twice

    # Operator fixes horizontal_fov_deg -> profile version bumps -> cache re-validates.
    await profiles.set_all(
        [
            CameraProfile(
                camera_id="cam-a",
                horizontal_fov_deg=75.0,
                mounting_height_m=2.4,
                tilt_deg=-65.0,
            )
        ]
    )
    v3 = cache.get("cam-a", width_px=1920, height_px=1080)
    assert v3.reliable is True
    assert v3.pose is not None
    assert counter._value.get() == before + 1.0


@pytest.mark.asyncio
async def test_height_estimation_stage_populates_standing_and_skips_non_standing() -> None:
    width_px, height_px = 1920, 1080
    h_i2f, rot, centre, f_px, principal = _synthetic_camera(
        centre_m=(0.0, 0.0, 2.5),
        tilt_deg=-30.0,
        yaw_deg=0.0,
        fov_deg=75.0,
        width_px=width_px,
        height_px=height_px,
    )
    cal = CalibrationState()
    await cal.set_homography("cam-a", h_i2f.tolist())
    profiles = CameraProfileMap(
        [
            CameraProfile(
                camera_id="cam-a",
                horizontal_fov_deg=75.0,
                mounting_height_m=2.5,
                tilt_deg=-30.0,
            )
        ]
    )

    stage = HeightEstimationStage(
        calibration_state=cal,
        camera_profiles=profiles,
        config=HeightEstimationConfig(),
    )

    foot_xy = (3.2, 0.1)
    true_h = 1.64
    foot_u, foot_v = _project_point((foot_xy[0], foot_xy[1], 0.0), rot, centre, f_px, principal)
    _, head_v = _project_point((foot_xy[0], foot_xy[1], true_h), rot, centre, f_px, principal)

    half_w = int(0.18 * (foot_v - head_v))
    standing_box = BoundingBox(
        x_min=round(foot_u - half_w),
        y_min=round(head_v),
        x_max=round(foot_u + half_w),
        y_max=round(foot_v),
    )
    ts = datetime(2026, 11, 21, 12, 0, 0, tzinfo=UTC)
    det_standing = Detection(
        detection_id="det-stand",
        camera_id="cam-a",
        bbox=standing_box,
        embedding=[],
        capture_time=ts,
        event_time=ts,
        confidence=0.95,
        floor_point=FloorPoint(
            x_mm=round(foot_xy[0] * 1000),
            y_mm=round(foot_xy[1] * 1000),
            calibrated=True,
        ),
    )
    det_sitting = Detection(
        detection_id="det-sit",
        camera_id="cam-a",
        bbox=BoundingBox(x_min=800, y_min=400, x_max=1050, y_max=600),
        embedding=[],
        capture_time=ts,
        event_time=ts,
        confidence=0.95,
        floor_point=det_standing.floor_point,
    )

    frame = FrameReady(
        camera_id="cam-a",
        frame_index=1,
        minio_key="k",
        width=width_px,
        height=height_px,
    )
    ctx = FrameContext(
        frame=frame,
        event_time=ts,
        capture_time=ts,
        effective_width=width_px,
        effective_height=height_px,
    )
    ctx.domain_detections = [det_standing, det_sitting]
    ctx.det_pose_result = {
        "det-stand": _make_pose_result(
            shoulder_x=foot_u,
            hip_x=foot_u,
            shoulder_y=head_v + 40.0,
            hip_y=(head_v + foot_v) / 2.0,
            ankle_y=foot_v - 10.0,
        ),
        "det-sit": _make_pose_result(ankle_score=0.05),
    }

    await stage.run(ctx)

    assert "det-stand" in ctx.height_by_detection
    assert "det-sit" not in ctx.height_by_detection
    est_h, est_sigma = ctx.height_by_detection["det-stand"]
    assert est_h == pytest.approx(true_h, abs=0.03)
    assert 0.04 <= est_sigma <= 0.25

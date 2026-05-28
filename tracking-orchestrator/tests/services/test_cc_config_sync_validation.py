"""WTR5: CC config sync validation tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.calibration.validator import validate_homography


def test_error_homography_is_rejected():
    """A degenerate matrix (zero determinant) must return severity=error."""
    # Near-zero determinant (degenerate).
    degenerate = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1e-7],
    ]
    result = validate_homography(
        matrix=degenerate,
        residuals=[0.6],  # also high residual
    )
    assert result.severity == "error"
    assert any("degenerate" in i for i in result.issues)


def test_warning_homography_is_accepted_with_warning():
    """A matrix with moderate residual gets severity=warning."""
    matrix = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    result = validate_homography(
        matrix=matrix,
        residuals=[0.3],  # moderate residual (0.25 < 0.3 <= 0.5)
    )
    assert result.severity == "warning"
    assert any("warning" in i for i in result.issues)


def test_ok_homography_passes():
    """A valid matrix with low residuals gets severity=ok."""
    matrix = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    result = validate_homography(
        matrix=matrix,
        residuals=[0.1],
        image_width=1920,
        image_height=1080,
    )
    assert result.severity == "ok"
    assert result.issues == []


@pytest.mark.asyncio
async def test_rejected_homography_does_not_reach_calibration_state():
    """Error-severity homography must not call set_homography."""
    from app.calibration.state import CalibrationState
    from app.services.cc_config_sync import CCConfigSyncService

    cal_state = CalibrationState()
    cal_state.set_homography = AsyncMock()
    camera_room_map = MagicMock()
    camera_room_map.set_all = AsyncMock()

    client = MagicMock()
    client.get = AsyncMock(
        return_value=[
            {
                "id": "cam-1",
                "enabled": True,
                "room_id": 1,
                "room_name": "living_room",
                "homography_matrix": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1e-7]],
                "homography_residual_m": 0.6,
            },
        ]
    )

    svc = CCConfigSyncService(
        client=client,
        calibration_state=cal_state,
        camera_room_map=camera_room_map,
        poll_interval_s=999,
    )

    await svc._poll()

    # set_homography must NOT be called for an error-severity matrix.
    cal_state.set_homography.assert_not_called()
    # Camera→room binding should still be set even if homography was rejected.
    camera_room_map.set_all.assert_called_once()

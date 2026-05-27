"""Polls cognitive-companion for camera + room + homography state (M2).

Replaces the static ``settings.yaml.camera_room_map`` and makes CC the
single source of truth for calibration data.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime

from structlog import get_logger

from ..calibration.state import CalibrationState
from ..services.camera_room_map import CameraRoomBinding, CameraRoomMap

logger = get_logger(__name__)


class CCConfigSyncService:
    """Background task that polls CC every *poll_interval_s* and applies
    camera→room bindings and homography matrices atomically."""

    def __init__(
        self,
        client: object,  # CognitiveCompanionClient (protocol)
        calibration_state: CalibrationState,
        camera_room_map: CameraRoomMap,
        poll_interval_s: float = 60.0,
    ) -> None:
        self._client = client
        self._calibration_state = calibration_state
        self._camera_room_map = camera_room_map
        self._poll_interval_s = poll_interval_s
        self._stop_event = asyncio.Event()

    async def run(self) -> None:
        """Run the poll loop until stopped."""
        logger.info(
            "cc_config_sync_started",
            poll_interval_s=self._poll_interval_s,
        )
        while not self._stop_event.is_set():
            try:
                await self._poll()
            except Exception:
                logger.exception("cc_config_sync_poll_failed")
            with suppress(TimeoutError):
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._poll_interval_s)

    async def stop(self) -> None:
        self._stop_event.set()

    async def _poll(self) -> None:
        """Fetch camera configs from CC and apply to local state."""
        try:
            cameras = await self._client.get(  # type: ignore[attr-defined]
                "/api/v1/cts/cameras?include_calibration=true"
            )
        except Exception:
            logger.warning("cc_config_sync_fetch_failed")
            return

        if not isinstance(cameras, list):
            return

        bindings: list[CameraRoomBinding] = []
        now = datetime.now(UTC)

        for cam in cameras:
            if not isinstance(cam, dict):
                continue
            camera_id = cam.get("id", "")
            if not camera_id or not cam.get("enabled", True):
                continue

            # Camera→room binding.
            room_id = cam.get("room_id")
            room_name = cam.get("room_name", "")
            if room_id is not None and room_name:
                bindings.append(
                    CameraRoomBinding(
                        camera_id=camera_id,
                        room_id=str(room_id),
                        room_name=room_name,
                        bound_at=now,
                    )
                )

            # Homography matrix — validate before applying.
            matrix = cam.get("homography_matrix")
            if matrix and isinstance(matrix, list) and len(matrix) == 3:
                residual_m = cam.get("homography_residual_m")
                residuals = [float(residual_m)] if residual_m is not None else None

                # Validate: reject matrices that fail the server-side sanity
                # checks (degenerate determinant, residual > 0.5 m, etc.).
                from ..calibration.validator import validate_homography
                from ..observability import metrics as _m

                validation = validate_homography(
                    matrix=matrix,
                    residuals=residuals,
                    image_width=cam.get("frame_natural_width", 0) or 0,
                    image_height=cam.get("frame_natural_height", 0) or 0,
                )
                if not validation.ok:
                    reason = (
                        "high_residual"
                        if any("residual" in i for i in validation.issues)
                        else "degenerate"
                        if any("degenerate" in i for i in validation.issues)
                        else "validation_failed"
                    )
                    logger.warning(
                        "cc_config_sync_homography_rejected",
                        camera_id=camera_id,
                        severity=validation.severity,
                        code=reason,
                        issues=validation.issues,
                    )
                    _m.metrics.homography_rejected_total.labels(
                        reason=reason, camera_id=camera_id
                    ).inc()
                    continue  # skip this camera's homography

                if validation.severity == "warning":
                    warn_reason = "high_residual"
                    _m.metrics.homography_warning_total.labels(
                        reason=warn_reason, camera_id=camera_id
                    ).inc()
                    logger.info(
                        "cc_config_sync_homography_warning",
                        camera_id=camera_id,
                        code=warn_reason,
                        issues=validation.issues,
                    )

                try:
                    await self._calibration_state.set_homography(
                        camera_id=camera_id,
                        matrix=matrix,
                        max_residual_m=float(residual_m) if residual_m is not None else 0.0,
                    )
                except Exception:
                    logger.warning(
                        "cc_config_sync_homography_set_failed",
                        camera_id=camera_id,
                    )

        if bindings:
            await self._camera_room_map.set_all(bindings)
            logger.debug(
                "cc_config_sync_applied",
                camera_count=len(bindings),
            )

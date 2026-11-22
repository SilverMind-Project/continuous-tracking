"""Thread-safe calibration state store for the orchestrator.

Homography matrices and privacy zones arrive from the CC config sync (and from the
/internal/calibration/* endpoints CC calls when an operator saves). The pipeline reads from
this store on every frame so updates take effect without restart.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.domain import CalibrationQuality, CameraCalibration


@dataclass
class PrivacyZoneConfig:
    zone_id: str
    polygon: list[list[float]]
    policy: str  # drop_detection | blur_region | mask_region (canonical)
    name: str = ""
    enabled: bool = True


@dataclass
class CalibrationState:
    """In-memory calibration store.  All mutations are protected by an asyncio lock."""

    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    # camera_id -> 3x3 homography matrix (row-major)
    homographies: dict[str, list[list[float]]] = field(default_factory=dict)

    # camera_id -> CameraCalibration (typed, richer than bare matrix)
    calibrations: dict[str, CameraCalibration] = field(default_factory=dict)

    # camera_id -> list of privacy zones
    privacy_zones: dict[str, list[PrivacyZoneConfig]] = field(default_factory=dict)

    # ISO-8601 wall-clock of the last successful reload
    last_reload_at: str | None = None

    # arbitrary metadata per camera (e.g. reprojection error)
    camera_meta: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Observed native frame dimensions (width_px, height_px) reported by FetchStage.
    observed_stream_sizes: dict[str, tuple[int, int]] = field(default_factory=dict)

    # Monotonic version incremented on every calibration mutation.
    version: int = 0

    async def set_homography(
        self,
        camera_id: str,
        matrix: list[list[float]],
        *,
        meta: dict[str, Any] | None = None,
        points: list[list[float]] | None = None,
        floor_plan_id: str = "",
        image_width: int = 0,
        image_height: int = 0,
        max_residual_m: float = 0.0,
        mean_residual_m: float = 0.0,
        quality_status: str = "ok",
        quality_point_count: int = 0,
    ) -> None:
        async with self._lock:
            self.homographies[camera_id] = matrix
            cm = self.camera_meta.setdefault(camera_id, {})
            if meta:
                cm.update(meta)
            if points is not None:
                cm["points"] = points

            # Populate the typed calibrations dict when enough metadata exists.
            if floor_plan_id or image_width or image_height:
                quality = CalibrationQuality(
                    max_residual_m=max_residual_m,
                    mean_residual_m=mean_residual_m,
                    status=quality_status,  # type: ignore[arg-type]
                    point_count=quality_point_count,
                )
                self.calibrations[camera_id] = CameraCalibration(
                    camera_id=camera_id,
                    floor_plan_id=floor_plan_id,
                    matrix=[row[:] for row in matrix],
                    image_width=image_width,
                    image_height=image_height,
                    quality=quality,
                    calibrated_at=datetime.now(UTC),
                )

            self.version += 1

    async def remove_homographies(self, camera_ids: set[str]) -> list[str]:
        """Drop the homography of each listed camera; return the ids that had one, sorted.

        The camera's detections fall back to the virtual room on the next frame. Its fit
        metadata goes with the matrix; privacy zones are unaffected.
        """
        async with self._lock:
            removed = sorted(cid for cid in camera_ids if cid in self.homographies)
            for camera_id in removed:
                del self.homographies[camera_id]
                self.calibrations.pop(camera_id, None)
                self.camera_meta.pop(camera_id, None)
            if removed:
                self.version += 1
            return removed

    async def set_privacy_zones(
        self,
        camera_id: str,
        zones: list[PrivacyZoneConfig],
    ) -> None:
        async with self._lock:
            self.privacy_zones[camera_id] = zones
            self.version += 1

    async def reload(self) -> None:
        """Mark the last reload time.  Callers may extend this for filesystem re-reads."""
        async with self._lock:
            self.last_reload_at = datetime.now(UTC).isoformat()
            self.version += 1

    def calibrated_camera_ids(self) -> frozenset[str]:
        """Cameras whose floor homography is in use right now."""
        return frozenset(self.homographies)

    def record_observed_stream_size(self, camera_id: str, width_px: int, height_px: int) -> bool:
        """Record observed frame pixel dimensions for *camera_id*.

        Returns ``True`` when the recorded size is new or changed.
        """
        if width_px <= 0 or height_px <= 0:
            return False
        size = (width_px, height_px)
        if self.observed_stream_sizes.get(camera_id) == size:
            return False
        self.observed_stream_sizes[camera_id] = size
        return True

    def snapshot(self) -> dict[str, Any]:
        return {
            "homographies": dict(self.homographies),
            "calibrations": {
                cam_id: {
                    "floor_plan_id": cal.floor_plan_id,
                    "image_width": cal.image_width,
                    "image_height": cal.image_height,
                    "quality_status": cal.quality.status,
                    "max_residual_m": cal.quality.max_residual_m,
                    "mean_residual_m": cal.quality.mean_residual_m,
                    "point_count": cal.quality.point_count,
                    "calibrated_at": cal.calibrated_at.isoformat(),
                }
                for cam_id, cal in self.calibrations.items()
            },
            "privacy_zones": {
                k: [{"zone_id": z.zone_id, "policy": z.policy, "enabled": z.enabled} for z in v]
                for k, v in self.privacy_zones.items()
            },
            "last_reload_at": self.last_reload_at,
            "version": self.version,
        }


# Module-level singleton shared by the FastAPI app.
calibration_state = CalibrationState()

"""Polls cognitive-companion for camera, room, and homography state.

Replaces the static ``settings.yaml.camera_room_map`` and makes CC the
single source of truth for calibration data.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from structlog import get_logger
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from ..calibration.state import CalibrationState
from ..calibration.validator import validate_homography
from ..observability import metrics as _m
from ..services.camera_room_map import (
    CameraRoomBinding,
    CameraRoomMap,
    RoomPolygonBinding,
    RoomPolygonMap,
)

logger = get_logger(__name__)


def _log_fetch_retry(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome is not None else None
    logger.warning(
        "cc_config_sync_fetch_retry",
        attempt=retry_state.attempt_number,
        error=str(exc) if exc is not None else "",
    )


class CCConfigSyncContractError(RuntimeError):
    """Raised when CC returns malformed config data."""


def _require_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CCConfigSyncContractError(f"{context} must be an object")
    return value


def _require_list(value: Any, *, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise CCConfigSyncContractError(f"{context} must be a list")
    return value


def _require_non_empty_str(data: dict[str, Any], key: str, *, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise CCConfigSyncContractError(f"{context}.{key} must be a non-empty string")
    return value


def _require_bool(data: dict[str, Any], key: str, *, context: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise CCConfigSyncContractError(f"{context}.{key} must be a boolean")
    return value


def _require_number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CCConfigSyncContractError(f"{context} must be a number")
    return float(value)


def _optional_int(data: dict[str, Any], key: str, *, context: str, default: int = 0) -> int:
    value = data.get(key)
    if value is None:
        return default
    if not isinstance(value, int):
        raise CCConfigSyncContractError(f"{context}.{key} must be an integer")
    return value


def _optional_float(data: dict[str, Any], key: str, *, context: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    return _require_number(value, context=f"{context}.{key}")


def _optional_float_list(data: dict[str, Any], key: str, *, context: str) -> list[float] | None:
    value = data.get(key)
    if value is None:
        return None
    raw_items = _require_list(value, context=f"{context}.{key}")
    return [_require_number(item, context=f"{context}.{key}[]") for item in raw_items]


def _optional_matrix3(data: dict[str, Any], key: str, *, context: str) -> list[list[float]] | None:
    value = data.get(key)
    if value is None:
        return None
    rows = _require_list(value, context=f"{context}.{key}")
    if len(rows) != 3:
        raise CCConfigSyncContractError(f"{context}.{key} must have exactly 3 rows")
    matrix: list[list[float]] = []
    for row_index, row in enumerate(rows):
        row_items = _require_list(row, context=f"{context}.{key}[{row_index}]")
        if len(row_items) != 3:
            raise CCConfigSyncContractError(
                f"{context}.{key}[{row_index}] must have exactly 3 columns"
            )
        matrix.append(
            [
                _require_number(item, context=f"{context}.{key}[{row_index}][]")
                for item in row_items
            ]
        )
    return matrix


class CCConfigSyncService:
    """Background task that polls CC every *poll_interval_s* and applies
    camera→room bindings and homography matrices atomically."""

    def __init__(
        self,
        client: object,  # CognitiveCompanionClient (protocol)
        calibration_state: CalibrationState,
        camera_room_map: CameraRoomMap,
        room_polygon_map: RoomPolygonMap,
        poll_interval_s: float = 60.0,
        fetch_retry_attempts: int = 3,
        fetch_retry_initial_s: float = 3.0,
        fetch_retry_max_s: float = 15.0,
    ) -> None:
        self._client = client
        self._calibration_state = calibration_state
        self._camera_room_map = camera_room_map
        self._room_polygon_map = room_polygon_map
        self._poll_interval_s = poll_interval_s
        self._fetch_retry_attempts = fetch_retry_attempts
        self._fetch_retry_initial_s = fetch_retry_initial_s
        self._fetch_retry_max_s = fetch_retry_max_s
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
        now = datetime.now(UTC)
        try:
            await self._poll_validated(now)
        except CCConfigSyncContractError:
            await self._camera_room_map.set_all([])
            await self._room_polygon_map.set_all([])
            raise

    async def _poll_validated(self, now: datetime) -> None:
        """Apply one CC sync poll after clearing stale maps on contract failures."""
        cameras = _require_list(await self._fetch_cameras(), context="/api/v1/cts/cameras")

        bindings: list[CameraRoomBinding] = []

        for index, cam_raw in enumerate(cameras):
            context = f"/api/v1/cts/cameras[{index}]"
            cam = _require_mapping(cam_raw, context=context)
            camera_id = _require_non_empty_str(cam, "id", context=context)
            if not _require_bool(cam, "enabled", context=context):
                continue

            # Camera→room binding.
            room_id = cam.get("room_id")
            if room_id is None:
                raise CCConfigSyncContractError(
                    f"{context}.room_id is required for enabled cameras"
                )
            room_name = _require_non_empty_str(cam, "room_name", context=context)
            bindings.append(
                CameraRoomBinding(
                    camera_id=camera_id,
                    room_id=str(room_id),
                    room_name=room_name,
                    bound_at=now,
                )
            )

            # Homography matrix — validate before applying.
            matrix = _optional_matrix3(cam, "homography_matrix", context=context)
            if matrix is not None:
                residual_m = _optional_float(cam, "homography_residual_m", context=context)
                residuals = _optional_float_list(cam, "homography_residuals", context=context)
                if residuals is None and residual_m is not None:
                    residuals = [residual_m]
                floor_plan_id = _require_non_empty_str(
                    cam, "homography_floor_plan_id", context=context
                )

                # Validate: reject matrices that fail the server-side sanity
                # checks (degenerate determinant, residual > 0.5 m, etc.).
                validation = validate_homography(
                    matrix=matrix,
                    residuals=residuals,
                    image_width=_optional_int(cam, "frame_natural_width", context=context),
                    image_height=_optional_int(cam, "frame_natural_height", context=context),
                )
                if validation.severity == "error":
                    logger.warning(
                        "cc_config_sync_homography_rejected",
                        camera_id=camera_id,
                        severity=validation.severity,
                        code=validation.code,
                        issues=validation.issues,
                    )
                    _m.metrics.homography_rejected_total.labels(
                        reason=validation.code, camera_id=camera_id
                    ).inc()
                    continue

                if validation.severity == "warning":
                    _m.metrics.homography_warning_total.labels(
                        reason=validation.code, camera_id=camera_id
                    ).inc()
                    logger.info(
                        "cc_config_sync_homography_warning",
                        camera_id=camera_id,
                        code=validation.code,
                        issues=validation.issues,
                    )

                mean_residual_m = sum(residuals) / len(residuals) if residuals else 0.0
                await self._calibration_state.set_homography(
                    camera_id=camera_id,
                    matrix=matrix,
                    max_residual_m=residual_m if residual_m is not None else 0.0,
                    mean_residual_m=mean_residual_m,
                    floor_plan_id=floor_plan_id,
                    image_width=_optional_int(cam, "frame_natural_width", context=context),
                    image_height=_optional_int(cam, "frame_natural_height", context=context),
                    quality_status=validation.severity,
                    quality_point_count=len(residuals) if residuals else 0,
                )

        await self._camera_room_map.set_all(bindings)
        logger.debug(
            "cc_config_sync_applied",
            camera_count=len(bindings),
        )

        await self._sync_room_polygons(now)

    async def _sync_room_polygons(self, now: datetime) -> None:
        """Fetch CC room polygons and convert normalized floor-plan coords to metres."""
        floor_plan = _require_mapping(
            await self._fetch_floor_plan(), context="/api/v1/household/floor-plan"
        )
        width_px = floor_plan.get("floor_plan_width")
        height_px = floor_plan.get("floor_plan_height")
        meters_per_pixel = floor_plan.get("floor_meters_per_pixel")
        if not width_px or not height_px or not meters_per_pixel:
            await self._room_polygon_map.set_all([])
            logger.warning(
                "cc_config_sync_room_polygons_disabled",
                reason="floor_plan_scale_missing",
            )
            return

        mpp = _require_number(
            meters_per_pixel,
            context="/api/v1/household/floor-plan.floor_meters_per_pixel",
        )
        width_m = (
            _require_number(
                width_px,
                context="/api/v1/household/floor-plan.floor_plan_width",
            )
            * mpp
        )
        height_m = (
            _require_number(
                height_px,
                context="/api/v1/household/floor-plan.floor_plan_height",
            )
            * mpp
        )
        rooms = _require_list(await self._fetch_rooms(), context="/api/v1/rooms")
        bindings: list[RoomPolygonBinding] = []
        for index, room_raw in enumerate(rooms):
            context = f"/api/v1/rooms[{index}]"
            room = _require_mapping(room_raw, context=context)
            room_id = room.get("id")
            if room_id is None:
                raise CCConfigSyncContractError(f"{context}.id is required")
            room_name = _require_non_empty_str(room, "name", context=context)
            polygon_raw = room.get("floor_polygon")
            if polygon_raw is None:
                continue
            polygon_points = _require_list(polygon_raw, context=f"{context}.floor_polygon")
            polygon_m: list[tuple[float, float]] = []
            for point_index, point_raw in enumerate(polygon_points):
                point = _require_list(
                    point_raw, context=f"{context}.floor_polygon[{point_index}]"
                )
                if len(point) < 2:
                    raise CCConfigSyncContractError(
                        f"{context}.floor_polygon[{point_index}] must have at least 2 values"
                    )
                x = _require_number(point[0], context=f"{context}.floor_polygon[{point_index}][0]")
                y = _require_number(point[1], context=f"{context}.floor_polygon[{point_index}][1]")
                polygon_m.append((x * width_m, y * height_m))
            if len(polygon_m) < 3:
                raise CCConfigSyncContractError(
                    f"{context}.floor_polygon must have at least 3 points"
                )
            bindings.append(
                RoomPolygonBinding(
                    room_id=str(room_id),
                    room_name=str(room_name),
                    polygon_m=polygon_m,
                    bound_at=now,
                )
            )

        await self._room_polygon_map.set_all(bindings)
        logger.debug("cc_config_sync_room_polygons_applied", room_count=len(bindings))

    async def _fetch_cameras(self) -> Any:
        """Fetch CC camera config with bounded retry for transient upstream failures."""
        return await self._fetch_with_retry("/api/v1/cts/cameras")

    async def _fetch_rooms(self) -> Any:
        """Fetch CC room config with bounded retry for transient upstream failures."""
        return await self._fetch_with_retry("/api/v1/rooms")

    async def _fetch_floor_plan(self) -> Any:
        """Fetch CC floor-plan scale with bounded retry for transient upstream failures."""
        return await self._fetch_with_retry("/api/v1/household/floor-plan")

    async def _fetch_with_retry(self, path: str) -> Any:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(self._fetch_retry_attempts),
            wait=wait_exponential_jitter(
                initial=self._fetch_retry_initial_s,
                max=self._fetch_retry_max_s,
            ),
            before_sleep=_log_fetch_retry,
            reraise=True,
        ):
            with attempt:
                return await self._client.get(path)  # type: ignore[attr-defined]

        raise RuntimeError("unreachable tenacity retry state")

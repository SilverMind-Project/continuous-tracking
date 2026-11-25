"""Polls cognitive-companion for camera, room, homography, and privacy-zone state.

Replaces the static ``settings.yaml.camera_room_map`` and makes CC the
single source of truth for calibration data. Privacy zones live only in
:class:`CalibrationState` memory, so they are restored from CC before the
pipeline starts (:func:`restore_privacy_zones`) and re-applied on every poll.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final, NoReturn, get_args

from cts_contracts import CameraRole, DeviceKind, RangingTechnology, RfNodeRole
from structlog import get_logger
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.calibration.state import CalibrationState, PrivacyZoneConfig
from app.calibration.validator import validate_homography
from app.domain import (
    CameraProfile,
    OverlapGroup,
    PrivacyPolicy,
    RfNode,
    TransitZone,
    WearableBinding,
    WearableDevice,
    WearableSiteConfig,
)
from app.observability import metrics as _m
from app.observability.metrics import CameraProfileRejectReason, CCSyncResource
from app.services.camera_profile_map import CameraProfileMap
from app.services.camera_room_map import (
    CameraRoomBinding,
    CameraRoomMap,
    RoomPolygonBinding,
    RoomPolygonMap,
)
from app.services.overlap_group_sync import OVERLAP_GROUPS_PATH, parse_overlap_groups
from app.services.transit_zone_map import TransitZoneBinding, TransitZoneMap
from app.services.wearable_site_config_map import WearableSiteConfigMap

if TYPE_CHECKING:
    from app.services.cc_client import CognitiveCompanionClient
    from app.storage.height_profiles import IdentityHeightProfileRepository

logger = get_logger(__name__)

_PRIVACY_POLICIES: Final[frozenset[str]] = frozenset(get_args(PrivacyPolicy))
_CAMERAS_PATH: Final = "/api/v1/cts/cameras"
WEARABLES_SITE_CONFIG_PATH: Final = "/api/v1/wearables/site-config"
PERSON_HEIGHTS_PATH: Final = "/api/v1/persons/heights"
_OPERATOR_HEIGHT_SIGMA_M: Final[float] = 0.05


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


def _optional_bool(data: dict[str, Any], key: str, *, context: str) -> bool | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise CCConfigSyncContractError(f"{context}.{key} must be a boolean")
    return value


def _optional_positive_int(data: dict[str, Any], key: str, *, context: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CCConfigSyncContractError(f"{context}.{key} must be a positive integer")
    return value


def _require_finite_number(value: Any, *, context: str) -> float:
    num = _require_number(value, context=context)
    if not math.isfinite(num):
        raise CCConfigSyncContractError(f"{context} must be a finite number")
    return num


def _parse_iso_utc(value: Any, *, context: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise CCConfigSyncContractError(f"{context} must be an ISO-8601 timestamp string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise CCConfigSyncContractError(f"{context} is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CCConfigSyncContractError(f"{context} must be timezone-aware")
    return parsed.astimezone(UTC)


def parse_wearable_site_config(payload: Any) -> WearableSiteConfig:
    """Decode CC's ``GET /api/v1/wearables/site-config`` payload into frozen domain objects.

    Converts RF node coordinates from metres to integer millimetres at the
    boundary (matching ``FloorPoint``) and validates every closed vocabulary.
    Raises :class:`CCConfigSyncContractError` on any malformed field so the
    caller keeps its previous live configuration.
    """
    ctx = WEARABLES_SITE_CONFIG_PATH
    root = _require_mapping(payload, context=ctx)
    version = _require_non_empty_str(root, "version", context=ctx)

    floor_plan_raw = root.get("floor_plan_id")
    if floor_plan_raw is not None and (not isinstance(floor_plan_raw, str) or not floor_plan_raw):
        raise CCConfigSyncContractError(f"{ctx}.floor_plan_id must be a non-empty string or null")
    floor_plan_id: str | None = floor_plan_raw

    raw_nodes = _require_list(root.get("nodes"), context=f"{ctx}.nodes")
    nodes: list[RfNode] = []
    for index, node_raw in enumerate(raw_nodes):
        nctx = f"{ctx}.nodes[{index}]"
        node = _require_mapping(node_raw, context=nctx)
        anchor_id = _require_non_empty_str(node, "anchor_id", context=nctx)
        adapter_id = _require_non_empty_str(node, "adapter_id", context=nctx)

        raw_role = _require_non_empty_str(node, "node_role", context=nctx)
        node_role = RfNodeRole.from_wire(raw_role)
        if node_role is None:
            raise CCConfigSyncContractError(f"{nctx}.node_role {raw_role!r} is unknown")

        raw_tech = _require_non_empty_str(node, "technology", context=nctx)
        technology = RangingTechnology.from_wire(raw_tech)
        if technology is None:
            raise CCConfigSyncContractError(f"{nctx}.technology {raw_tech!r} is unknown")

        x_m = _require_finite_number(node.get("x_m"), context=f"{nctx}.x_m")
        y_m = _require_finite_number(node.get("y_m"), context=f"{nctx}.y_m")
        z_m = _require_finite_number(node.get("z_m"), context=f"{nctx}.z_m")

        room_id_raw = node.get("room_id")
        if room_id_raw is None:
            room_id: str | None = None
        elif isinstance(room_id_raw, str) and room_id_raw:
            room_id = room_id_raw
        elif isinstance(room_id_raw, int) and not isinstance(room_id_raw, bool):
            room_id = str(room_id_raw)
        else:
            raise CCConfigSyncContractError(f"{nctx}.room_id must be a string, int, or null")

        uwb_addr_raw = node.get("uwb_address_hex")
        if uwb_addr_raw is not None and (not isinstance(uwb_addr_raw, str) or not uwb_addr_raw):
            raise CCConfigSyncContractError(f"{nctx}.uwb_address_hex must be a string or null")
        uwb_channel = _optional_positive_int(node, "uwb_channel", context=nctx)
        uwb_preamble = _optional_positive_int(node, "uwb_preamble_index", context=nctx)

        nodes.append(
            RfNode(
                anchor_id=anchor_id,
                adapter_id=adapter_id,
                node_role=node_role,
                technology=technology,
                x_mm=round(x_m * 1000.0),
                y_mm=round(y_m * 1000.0),
                z_mm=round(z_m * 1000.0),
                room_id=room_id,
                uwb_address_hex=uwb_addr_raw,
                uwb_channel=uwb_channel,
                uwb_preamble_index=uwb_preamble,
            )
        )

    raw_bindings = _require_list(root.get("bindings"), context=f"{ctx}.bindings")
    bindings: list[WearableBinding] = []
    for index, binding_raw in enumerate(raw_bindings):
        bctx = f"{ctx}.bindings[{index}]"
        binding = _require_mapping(binding_raw, context=bctx)
        tag_id = _require_non_empty_str(binding, "tag_id", context=bctx)
        member_id = _require_non_empty_str(binding, "member_id", context=bctx)
        valid_from = _parse_iso_utc(binding.get("valid_from"), context=f"{bctx}.valid_from")
        valid_to_raw = binding.get("valid_to")
        valid_to = (
            _parse_iso_utc(valid_to_raw, context=f"{bctx}.valid_to")
            if valid_to_raw is not None
            else None
        )
        if valid_to is not None and valid_to <= valid_from:
            raise CCConfigSyncContractError(f"{bctx}.valid_to must be strictly after valid_from")
        bindings.append(
            WearableBinding(
                tag_id=tag_id,
                member_id=member_id,
                valid_from=valid_from,
                valid_to=valid_to,
            )
        )

    raw_devices = _require_list(root.get("devices"), context=f"{ctx}.devices")
    devices: list[WearableDevice] = []
    for index, device_raw in enumerate(raw_devices):
        dctx = f"{ctx}.devices[{index}]"
        device = _require_mapping(device_raw, context=dctx)
        tag_id = _require_non_empty_str(device, "tag_id", context=dctx)
        adapter_id = _require_non_empty_str(device, "adapter_id", context=dctx)
        hardware_key = _require_non_empty_str(device, "hardware_key", context=dctx)
        raw_kind = _require_non_empty_str(device, "device_kind", context=dctx)
        device_kind = DeviceKind.from_wire(raw_kind)
        if device_kind is None:
            raise CCConfigSyncContractError(f"{dctx}.device_kind {raw_kind!r} is unknown")
        is_active = _require_bool(device, "is_active", context=dctx)
        # Optional: CC builds that predate the field omit it, and single-tag devices send null.
        physical_raw = device.get("physical_device_id")
        if physical_raw is not None and (not isinstance(physical_raw, str) or not physical_raw):
            raise CCConfigSyncContractError(
                f"{dctx}.physical_device_id must be a non-empty string or null"
            )
        devices.append(
            WearableDevice(
                tag_id=tag_id,
                adapter_id=adapter_id,
                hardware_key=hardware_key,
                device_kind=device_kind,
                is_active=is_active,
                physical_device_id=physical_raw,
            )
        )

    return WearableSiteConfig(
        floor_plan_id=floor_plan_id,
        nodes=tuple(nodes),
        bindings=tuple(bindings),
        devices=tuple(devices),
        version=version,
    )


def parse_person_heights(raw: Any) -> dict[str, float | None]:
    """Decode CC ``GET /api/v1/persons/heights`` into ``{person_id: height_cm | None}``."""
    items = _require_list(raw, context=PERSON_HEIGHTS_PATH)
    heights: dict[str, float | None] = {}
    for index, entry in enumerate(items):
        context = f"{PERSON_HEIGHTS_PATH}[{index}]"
        row = _require_mapping(entry, context=context)
        person_id = _require_non_empty_str(row, "person_id", context=context)
        height_cm = _optional_float(row, "height_cm", context=context)
        if height_cm is not None and not (90.0 <= height_cm <= 220.0):
            raise CCConfigSyncContractError(
                f"{context}.height_cm must be in [90.0, 220.0], got {height_cm!r}"
            )
        heights[person_id] = height_cm
    return heights


def parse_camera_profile(
    cam: dict[str, Any], *, context: str
) -> tuple[CameraProfile | None, CameraProfileRejectReason | None]:
    """Decode one CC camera dict into a frozen :class:`CameraProfile`.

    Missing optional fields decode to ``None``; a missing ``role`` defaults to
    ``CameraRole.SURVEILLANCE`` and a missing ``face_id_enabled`` defaults to ``True``.
    An explicit unknown ``role`` is rejected loudly with
    ``CameraProfileRejectReason.UNKNOWN_ROLE`` and never silently defaults.
    """
    camera_id = _require_non_empty_str(cam, "id", context=context)

    raw_role = cam.get("role")
    if raw_role is None:
        role = CameraRole.SURVEILLANCE
    elif not isinstance(raw_role, str):
        return None, CameraProfileRejectReason.UNKNOWN_ROLE
    else:
        parsed_role = CameraRole.from_wire(raw_role)
        if parsed_role is None:
            return None, CameraProfileRejectReason.UNKNOWN_ROLE
        role = parsed_role

    try:
        face_id_enabled_opt = _optional_bool(cam, "face_id_enabled", context=context)
        face_id_enabled = True if face_id_enabled_opt is None else face_id_enabled_opt
        min_raw_similarity = _optional_float(cam, "face_id_min_raw_similarity", context=context)
        horizontal_fov_deg = _optional_float(cam, "horizontal_fov_deg", context=context)
        mounting_height_m = _optional_float(cam, "mounting_height_m", context=context)
        tilt_deg = _optional_float(cam, "tilt_deg", context=context)
        ir_capable = _optional_bool(cam, "ir_capable", context=context)
        frame_natural_width_px = _optional_positive_int(cam, "frame_natural_width", context=context)
        frame_natural_height_px = _optional_positive_int(
            cam, "frame_natural_height", context=context
        )
    except CCConfigSyncContractError:
        return None, CameraProfileRejectReason.MALFORMED_FIELD

    return (
        CameraProfile(
            camera_id=camera_id,
            role=role,
            face_id_enabled=face_id_enabled,
            face_id_min_raw_similarity=min_raw_similarity,
            horizontal_fov_deg=horizontal_fov_deg,
            mounting_height_m=mounting_height_m,
            tilt_deg=tilt_deg,
            ir_capable=ir_capable,
            frame_natural_width_px=frame_natural_width_px,
            frame_natural_height_px=frame_natural_height_px,
        ),
        None,
    )


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
            [_require_number(item, context=f"{context}.{key}[{row_index}][]") for item in row_items]
        )
    return matrix


def _privacy_polygon(value: Any, *, context: str) -> list[list[float]]:
    points = _require_list(value, context=context)
    if len(points) < 3:
        raise CCConfigSyncContractError(f"{context} needs at least 3 points")
    polygon: list[list[float]] = []
    for index, point_raw in enumerate(points):
        point = _require_list(point_raw, context=f"{context}[{index}]")
        if len(point) != 2:
            raise CCConfigSyncContractError(f"{context}[{index}] must be [x, y]")
        x = _require_number(point[0], context=f"{context}[{index}][0]")
        y = _require_number(point[1], context=f"{context}[{index}][1]")
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise CCConfigSyncContractError(f"{context}[{index}] must be normalized to [0, 1]")
        polygon.append([x, y])
    return polygon


def _privacy_zones(cam: dict[str, Any], *, context: str) -> list[PrivacyZoneConfig] | None:
    """Parse one camera's ``privacy_zones``; ``None`` when CC predates the field."""
    if "privacy_zones" not in cam:
        return None
    zones_context = f"{context}.privacy_zones"
    zones: list[PrivacyZoneConfig] = []
    for index, zone_raw in enumerate(_require_list(cam["privacy_zones"], context=zones_context)):
        zone_context = f"{zones_context}[{index}]"
        zone = _require_mapping(zone_raw, context=zone_context)
        policy = _require_non_empty_str(zone, "policy", context=zone_context)
        if policy not in _PRIVACY_POLICIES:
            raise CCConfigSyncContractError(f"{zone_context}.policy {policy!r} is not a policy")
        name = zone.get("name", "")
        if not isinstance(name, str):
            raise CCConfigSyncContractError(f"{zone_context}.name must be a string")
        zones.append(
            PrivacyZoneConfig(
                zone_id=_require_non_empty_str(zone, "zone_id", context=zone_context),
                polygon=_privacy_polygon(zone.get("polygon"), context=f"{zone_context}.polygon"),
                policy=policy,
                name=name,
                enabled=_require_bool(zone, "enabled", context=zone_context),
            )
        )
    return zones


async def apply_privacy_zones(cameras: list[Any], calibration_state: CalibrationState) -> bool:
    """Make *calibration_state* hold exactly the privacy zones CC lists for each camera.

    Every camera's zones are validated before any is applied, so a malformed zone raises
    :class:`CCConfigSyncContractError` and leaves the zones in force untouched rather than
    dropping a privacy mask. A camera no longer listed loses its zones. Returns ``False``, and
    changes nothing, when CC predates the ``privacy_zones`` field.
    """
    parsed: dict[str, list[PrivacyZoneConfig]] = {}
    for index, cam_raw in enumerate(cameras):
        context = f"{_CAMERAS_PATH}[{index}]"
        cam = _require_mapping(cam_raw, context=context)
        camera_id = _require_non_empty_str(cam, "id", context=context)
        zones = _privacy_zones(cam, context=context)
        if zones is None:
            return False
        parsed[camera_id] = zones

    for camera_id in set(calibration_state.privacy_zones) - set(parsed):
        parsed[camera_id] = []
    for camera_id, zones in parsed.items():
        if calibration_state.privacy_zones.get(camera_id, []) == zones:
            continue
        await calibration_state.set_privacy_zones(camera_id, zones)
        logger.info(
            "cc_config_sync_privacy_zones_applied",
            camera_id=camera_id,
            zone_count=len(zones),
        )
    return True


async def restore_privacy_zones(
    client: CognitiveCompanionClient, calibration_state: CalibrationState
) -> int:
    """Load every camera's privacy zones from CC before the pipeline processes a frame.

    Zones live only in memory, so without this a restart would track people inside zones the
    operator excluded until the first config poll. Raises ``RuntimeError`` when CC cannot be read
    or returns malformed zones, so the orchestrator does not start without them. Returns the
    number of zones restored.
    """
    try:
        cameras = await client.get(_CAMERAS_PATH)
    except Exception as exc:  # any transport failure means startup must stop
        logger.exception("privacy_zones_restore_fetch_failed")
        raise RuntimeError("Failed to fetch privacy zones from CC") from exc
    try:
        applied = await apply_privacy_zones(
            _require_list(cameras, context=_CAMERAS_PATH), calibration_state
        )
    except CCConfigSyncContractError as exc:
        raise RuntimeError(f"CC returned malformed privacy zones: {exc}") from exc
    if not applied:
        logger.warning("privacy_zones_restore_unavailable", reason="cc_predates_privacy_zones")
        return 0
    return sum(len(zones) for zones in calibration_state.privacy_zones.values())


class CCConfigSyncService:
    """Background task that polls CC every *poll_interval_s* and applies
    camera→room bindings, homography matrices, privacy zones, room polygons, transit zones,
    and camera overlap groups.

    CC is authoritative for homographies: a camera whose matrix CC no longer holds (cleared,
    rejected by validation, camera disabled or deleted) loses its runtime homography at the
    next poll and falls back to its virtual room.
    """

    def __init__(
        self,
        client: object,  # CognitiveCompanionClient (protocol)
        calibration_state: CalibrationState,
        camera_room_map: CameraRoomMap,
        room_polygon_map: RoomPolygonMap,
        transit_zone_map: TransitZoneMap,
        poll_interval_s: float = 60.0,
        fetch_retry_attempts: int = 3,
        fetch_retry_initial_s: float = 3.0,
        fetch_retry_max_s: float = 15.0,
        overlap_group_sink: Callable[[Sequence[OverlapGroup]], None] | None = None,
        camera_profile_map: CameraProfileMap | None = None,
        wearable_site_config_map: WearableSiteConfigMap | None = None,
        height_profile_repo: IdentityHeightProfileRepository | None = None,
        on_height_profiles_updated: Callable[[], None] | None = None,
    ) -> None:
        self._client = client
        self._calibration_state = calibration_state
        self._camera_room_map = camera_room_map
        self._room_polygon_map = room_polygon_map
        self._transit_zone_map = transit_zone_map
        self._camera_profile_map = (
            camera_profile_map if camera_profile_map is not None else CameraProfileMap()
        )
        self._wearable_site_config_map = wearable_site_config_map
        self._height_profile_repo = height_profile_repo
        self._on_height_profiles_updated = on_height_profiles_updated
        self._poll_interval_s = poll_interval_s
        self._fetch_retry_attempts = fetch_retry_attempts
        self._fetch_retry_initial_s = fetch_retry_initial_s
        self._fetch_retry_max_s = fetch_retry_max_s
        self._stop_event = asyncio.Event()
        self._warned_privacy_zones_unavailable = False
        self._warned_uncalibrated_id_cameras: set[str] = set()
        self._overlap_group_sink = overlap_group_sink
        self._applied_overlap_groups: tuple[OverlapGroup, ...] | None = None
        if overlap_group_sink is None:
            logger.warning("cc_config_sync_overlap_groups_not_wired")

    @property
    def camera_profile_map(self) -> CameraProfileMap:
        """Live map of synced camera profiles."""
        return self._camera_profile_map

    @property
    def wearable_site_config_map(self) -> WearableSiteConfigMap | None:
        """Live holder of synced wearable site configuration."""
        return self._wearable_site_config_map

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
            await self._transit_zone_map.set_all([])
            _m.metrics.transit_zones_loaded.set(0)
            raise

    async def _poll_validated(self, now: datetime) -> None:
        """Apply one CC sync poll after clearing stale maps on contract failures."""
        cameras = _require_list(await self._fetch_cameras(), context=_CAMERAS_PATH)

        # Privacy first: a later homography rejection skips the rest of a camera's loop body.
        if (
            not await apply_privacy_zones(cameras, self._calibration_state)
            and not self._warned_privacy_zones_unavailable
        ):
            self._warned_privacy_zones_unavailable = True
            logger.warning("cc_config_sync_privacy_zones_unavailable")

        bindings: list[CameraRoomBinding] = []
        profiles: list[CameraProfile] = []
        calibrated_camera_ids: set[str] = set()

        for index, cam_raw in enumerate(cameras):
            context = f"/api/v1/cts/cameras[{index}]"
            cam = _require_mapping(cam_raw, context=context)
            camera_id = _require_non_empty_str(cam, "id", context=context)
            if not _require_bool(cam, "enabled", context=context):
                continue

            profile, reject_reason = parse_camera_profile(cam, context=context)
            if reject_reason is not None or profile is None:
                reason_value = (
                    reject_reason.value
                    if reject_reason is not None
                    else CameraProfileRejectReason.MALFORMED_FIELD.value
                )
                logger.warning(
                    "cc_config_sync_camera_profile_rejected",
                    camera_id=camera_id,
                    reason=reason_value,
                    raw_role=cam.get("role"),
                )
                _m.metrics.camera_profile_rejected_total.labels(
                    camera_id=camera_id,
                    reason=reason_value,
                ).inc()
                continue
            profiles.append(profile)

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
                calibrated_camera_ids.add(camera_id)

        removed = await self._calibration_state.remove_homographies(
            set(self._calibration_state.homographies) - calibrated_camera_ids
        )
        if removed:
            logger.info("cc_config_sync_homography_removed", camera_ids=removed)

        active_profile_ids = {p.camera_id for p in profiles}
        self._warned_uncalibrated_id_cameras &= active_profile_ids
        for profile in profiles:
            if (
                profile.role is CameraRole.FACE_CAPABLE
                and profile.camera_id not in calibrated_camera_ids
            ):
                if profile.camera_id not in self._warned_uncalibrated_id_cameras:
                    logger.warning(
                        "identification_camera_uncalibrated",
                        camera_id=profile.camera_id,
                        role=profile.role.value,
                    )
                    self._warned_uncalibrated_id_cameras.add(profile.camera_id)
            else:
                self._warned_uncalibrated_id_cameras.discard(profile.camera_id)

        await self._camera_profile_map.set_all(profiles)
        await self._camera_room_map.set_all(bindings)
        logger.debug(
            "cc_config_sync_applied",
            camera_count=len(bindings),
        )

        await self._sync_overlap_groups()
        await self._sync_wearables()
        await self._sync_person_heights(now)

        scale_m = await self._floor_plan_scale_m()
        if scale_m is None:
            await self._room_polygon_map.set_all([])
            await self._transit_zone_map.set_all([])
            _m.metrics.transit_zones_loaded.set(0)
            logger.warning(
                "cc_config_sync_room_polygons_disabled",
                reason="floor_plan_scale_missing",
            )
            logger.warning(
                "cc_config_sync_transit_zones_disabled",
                reason="floor_plan_scale_missing",
            )
            return

        width_m, height_m = scale_m
        await self._sync_room_polygons(now, width_m=width_m, height_m=height_m)
        await self._sync_transit_zones(now, width_m=width_m, height_m=height_m)

    async def _sync_wearables(self) -> None:
        """Fetch CC wearable site-config and publish to the live holder.

        A transport failure or malformed payload keeps the previous snapshot
        intact and increments ``cts_cc_sync_failures_total{resource="wearables"}``.
        """
        if self._wearable_site_config_map is None:
            return
        try:
            raw = await self._fetch_with_retry(WEARABLES_SITE_CONFIG_PATH)
            parsed = parse_wearable_site_config(raw)
        except Exception:  # keep previous valid wearable config; retry next poll
            _m.metrics.cc_sync_failures_total.labels(resource=CCSyncResource.WEARABLES.value).inc()
            logger.warning("cc_config_sync_wearables_failed", exc_info=True)
            return
        current = self._wearable_site_config_map.get_sync()
        if parsed == current:
            return
        await self._wearable_site_config_map.set_config(parsed)
        logger.info(
            "cc_config_sync_wearables_applied",
            version=parsed.version,
            node_count=len(parsed.nodes),
            binding_count=len(parsed.bindings),
            device_count=len(parsed.devices),
        )

    async def _sync_person_heights(self, now: datetime) -> None:
        """Fetch operator-entered member heights from CC and upsert operator profiles.

        Never overwrites a ``learned`` profile; removes stale ``operator`` profiles when
        cleared in CC. Keeps previous profiles intact on any transport or contract failure.
        """
        if self._height_profile_repo is None:
            return
        try:
            raw = await self._fetch_with_retry(PERSON_HEIGHTS_PATH)
            entries = parse_person_heights(raw)
        except Exception:
            _m.metrics.cc_sync_failures_total.labels(
                resource=CCSyncResource.PERSON_HEIGHTS.value
            ).inc()
            logger.warning("cc_config_sync_person_heights_failed", exc_info=True)
            return

        changed = False
        for person_id, height_cm in entries.items():
            if height_cm is None:
                if await self._height_profile_repo.delete_operator_profile(person_id):
                    changed = True
                continue
            height_m = height_cm / 100.0
            if await self._height_profile_repo.upsert_operator_if_not_learned(
                person_id,
                height_m=height_m,
                height_sigma_m=_OPERATOR_HEIGHT_SIGMA_M,
                updated_at=now,
            ):
                changed = True

        if changed and self._on_height_profiles_updated is not None:
            self._on_height_profiles_updated()

    async def _sync_overlap_groups(self) -> None:
        """Apply CC's overlap groups to the tracker; keep the last applied set on failure.

        A transient CC failure or a malformed response must not dissolve the groups that are
        in use, so this logs and returns instead of raising.
        """
        if self._overlap_group_sink is None:
            return
        try:
            groups = tuple(parse_overlap_groups(await self._fetch_with_retry(OVERLAP_GROUPS_PATH)))
        except Exception:  # keep the applied groups; the next poll retries
            logger.exception("cc_config_sync_overlap_groups_failed")
            return
        if groups == self._applied_overlap_groups:
            return
        self._overlap_group_sink(groups)
        self._applied_overlap_groups = groups
        _m.metrics.overlap_groups_loaded.set(len(groups))
        logger.info(
            "cc_config_sync_overlap_groups_applied",
            group_count=len(groups),
            camera_ids=sorted({cid for g in groups for cid in g.camera_ids}),
        )

    async def _floor_plan_scale_m(self) -> tuple[float, float] | None:
        """Return floor-plan dimensions in metres, or None when scale is unset."""
        floor_plan = _require_mapping(
            await self._fetch_floor_plan(), context="/api/v1/household/floor-plan"
        )
        width_px = floor_plan.get("floor_plan_width")
        height_px = floor_plan.get("floor_plan_height")
        meters_per_pixel = floor_plan.get("floor_meters_per_pixel")
        if not width_px or not height_px or not meters_per_pixel:
            return None

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
        return width_m, height_m

    async def _sync_room_polygons(self, now: datetime, *, width_m: float, height_m: float) -> None:
        """Fetch CC room polygons and convert normalized floor-plan coords to metres."""
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
                point = _require_list(point_raw, context=f"{context}.floor_polygon[{point_index}]")
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

    async def _sync_transit_zones(self, now: datetime, *, width_m: float, height_m: float) -> None:
        """Fetch CC transit zones and convert normalized floor-plan coords to metres."""
        zones = _require_list(
            await self._fetch_transit_zones(), context="/api/v1/cts/transit-zones"
        )
        bindings: list[TransitZoneBinding] = []
        for index, zone_raw in enumerate(zones):
            context = f"/api/v1/cts/transit-zones[{index}]"
            zone = _require_mapping(zone_raw, context=context)
            zone_id = zone.get("id")
            if zone_id is None:
                self._reject_transit_zone("missing_id", context=context)
            name = _require_non_empty_str(zone, "name", context=context)
            kind = _require_non_empty_str(zone, "kind", context=context)
            inside_room_id = zone.get("inside_room_id")
            if inside_room_id is None:
                self._reject_transit_zone("missing_inside_room_id", context=context)
            outside_room_id = zone.get("outside_room_id")
            if outside_room_id is None:
                self._reject_transit_zone("missing_outside_room_id", context=context)

            polygon_raw = zone.get("polygon")
            if polygon_raw is None:
                self._reject_transit_zone("missing_polygon", context=context)
            polygon_points = _require_list(polygon_raw, context=f"{context}.polygon")
            polygon_m: list[tuple[float, float]] = []
            for point_index, point_raw in enumerate(polygon_points):
                point = _require_list(point_raw, context=f"{context}.polygon[{point_index}]")
                if len(point) < 2:
                    self._reject_transit_zone("malformed_polygon_point", context=context)
                x = _require_number(point[0], context=f"{context}.polygon[{point_index}][0]")
                y = _require_number(point[1], context=f"{context}.polygon[{point_index}][1]")
                polygon_m.append((x * width_m, y * height_m))
            if len(polygon_m) < 3:
                self._reject_transit_zone("polygon_too_short", context=context)

            direction_raw = zone.get("direction_vec")
            if direction_raw is None:
                self._reject_transit_zone("missing_direction_vec", context=context)
            direction_items = _require_list(direction_raw, context=f"{context}.direction_vec")
            if len(direction_items) != 2:
                self._reject_transit_zone("malformed_direction_vec", context=context)
            direction_vec = (
                _require_number(direction_items[0], context=f"{context}.direction_vec[0]")
                * width_m,
                _require_number(direction_items[1], context=f"{context}.direction_vec[1]")
                * height_m,
            )

            bindings.append(
                TransitZoneBinding(
                    zone=TransitZone(
                        zone_id=str(zone_id),
                        name=name,
                        kind=kind,
                        polygon=polygon_m,
                        inside_room_id=str(inside_room_id),
                        outside_room_id=str(outside_room_id),
                        direction_vec=direction_vec,
                    ),
                    bound_at=now,
                )
            )

        await self._transit_zone_map.set_all(bindings)
        _m.metrics.transit_zones_loaded.set(len(bindings))
        logger.debug("cc_config_sync_transit_zones_applied", zone_count=len(bindings))

    def _reject_transit_zone(self, reason: str, *, context: str) -> NoReturn:
        _m.metrics.transit_zone_rejected_total.labels(reason=reason).inc()
        raise CCConfigSyncContractError(f"{context} rejected: {reason}")

    async def _fetch_cameras(self) -> Any:
        """Fetch CC camera config with bounded retry for transient upstream failures."""
        return await self._fetch_with_retry("/api/v1/cts/cameras")

    async def _fetch_rooms(self) -> Any:
        """Fetch CC room config with bounded retry for transient upstream failures."""
        return await self._fetch_with_retry("/api/v1/rooms")

    async def _fetch_floor_plan(self) -> Any:
        """Fetch CC floor-plan scale with bounded retry for transient upstream failures."""
        return await self._fetch_with_retry("/api/v1/household/floor-plan")

    async def _fetch_transit_zones(self) -> Any:
        """Fetch CC transit zones with bounded retry for transient upstream failures."""
        return await self._fetch_with_retry("/api/v1/cts/transit-zones")

    async def _fetch_with_retry(self, path: str) -> Any:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(self._fetch_retry_attempts),
            wait=wait_exponential_jitter(
                multiplier=self._fetch_retry_initial_s,
                max=self._fetch_retry_max_s,
            ),
            before_sleep=_log_fetch_retry,
            reraise=True,
        ):
            with attempt:
                return await self._client.get(path)  # type: ignore[attr-defined]

        raise RuntimeError("unreachable tenacity retry state")

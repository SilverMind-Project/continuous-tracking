"""Site configuration client polling Cognitive Companion's wearable site-config endpoint.

Polls ``GET /api/v1/wearables/site-config`` with ``X-API-Key`` authentication,
validating the JSON payload with Pydantic v2 at the HTTP boundary and converting
it to a frozen :class:`rf_ingress.domain.SiteConfigSnapshot`. On transient
failure, the last valid snapshot is retained.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

import httpx
import structlog
from cts_contracts.wearables import DeviceKind, RangingTechnology, RfNodeRole
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rf_ingress.domain import (
    PositionSensorConfig,
    RfNodeGeometry,
    SiteBinding,
    SiteConfigSnapshot,
    SiteDevice,
    ensure_utc,
)
from rf_ingress.observability import SITE_CONFIG_POLL_FAILURES_TOTAL

logger = structlog.get_logger(__name__)


class _SiteConfigNodeSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    anchor_id: str
    adapter_id: str
    node_role: RfNodeRole
    technology: RangingTechnology
    x_m: float
    y_m: float
    z_m: float
    room_id: str | None = None
    uwb_address_hex: str | None = None
    uwb_channel: int | None = None
    uwb_preamble_index: int | None = None


class _SiteConfigBindingSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tag_id: str
    member_id: str
    valid_from: datetime
    valid_to: datetime | None = None


class _SiteConfigDeviceSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tag_id: str
    adapter_id: str
    hardware_key: str
    device_kind: DeviceKind
    is_active: bool


class _SiteConfigPositionSensorSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sensor_id: str
    room_id: str | None = None
    x_m: float
    y_m: float
    yaw_deg: float | None = None
    calibration_residual_m: float


class _WearableSiteConfigSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    floor_plan_id: str | None = None
    nodes: list[_SiteConfigNodeSchema]
    bindings: list[_SiteConfigBindingSchema]
    devices: list[_SiteConfigDeviceSchema]
    position_sensors: list[_SiteConfigPositionSensorSchema] = Field(default_factory=list)
    version: str


def parse_site_config_payload(payload: object) -> SiteConfigSnapshot:
    """Validate a raw JSON-decoded site-config payload into a frozen snapshot."""
    parsed = _WearableSiteConfigSchema.model_validate(payload)
    nodes = tuple(
        RfNodeGeometry(
            anchor_id=n.anchor_id,
            x_m=float(n.x_m),
            y_m=float(n.y_m),
            z_m=float(n.z_m),
            adapter_id=n.adapter_id,
            node_role=n.node_role,
            technology=n.technology,
            room_id=n.room_id,
            uwb_address_hex=n.uwb_address_hex,
            uwb_channel=n.uwb_channel,
            uwb_preamble_index=n.uwb_preamble_index,
        )
        for n in parsed.nodes
    )
    bindings = tuple(
        SiteBinding(
            tag_id=b.tag_id,
            member_id=b.member_id,
            valid_from=ensure_utc(b.valid_from),
            valid_to=ensure_utc(b.valid_to) if b.valid_to is not None else None,
        )
        for b in parsed.bindings
    )
    devices = tuple(
        SiteDevice(
            tag_id=d.tag_id,
            adapter_id=d.adapter_id,
            hardware_key=d.hardware_key,
            device_kind=d.device_kind,
            is_active=d.is_active,
        )
        for d in parsed.devices
    )
    position_sensors = tuple(
        PositionSensorConfig(
            sensor_id=s.sensor_id,
            room_id=s.room_id,
            x_m=float(s.x_m),
            y_m=float(s.y_m),
            yaw_deg=float(s.yaw_deg) if s.yaw_deg is not None else None,
            calibration_residual_m=float(s.calibration_residual_m),
        )
        for s in parsed.position_sensors
    )
    return SiteConfigSnapshot(
        floor_plan_id=parsed.floor_plan_id,
        nodes=nodes,
        bindings=bindings,
        devices=devices,
        position_sensors=position_sensors,
        version=parsed.version,
    )


class SiteConfigProvider(Protocol):
    """Protocol consumed by adapters, runtime, and health reporting."""

    def snapshot(self) -> SiteConfigSnapshot | None: ...

    def last_synced_at(self) -> datetime | None: ...

    async def refresh(self) -> SiteConfigSnapshot | None: ...

    async def close(self) -> None: ...


class InMemorySiteConfigProvider:
    """In-memory :class:`SiteConfigProvider` for unit tests and offline replay."""

    def __init__(
        self,
        initial: SiteConfigSnapshot | None = None,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._snapshot: SiteConfigSnapshot | None = initial
        self._clock: Callable[[], datetime] = clock
        self._last_synced: datetime | None = ensure_utc(clock()) if initial is not None else None

    def set_snapshot(self, snapshot: SiteConfigSnapshot | None) -> None:
        """Replace the active snapshot immediately."""
        self._snapshot = snapshot
        if snapshot is not None:
            self._last_synced = ensure_utc(self._clock())

    def snapshot(self) -> SiteConfigSnapshot | None:
        return self._snapshot

    def last_synced_at(self) -> datetime | None:
        return self._last_synced

    async def refresh(self) -> SiteConfigSnapshot | None:
        return self._snapshot

    async def close(self) -> None:
        return None


class HttpSiteConfigClient:
    """HTTP polling client for CC ``GET /api/v1/wearables/site-config``."""

    def __init__(
        self,
        *,
        cc_api_url: str,
        api_key: str,
        timeout_s: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._base_url: str = cc_api_url.rstrip("/")
        self._api_key: str = api_key
        self._clock: Callable[[], datetime] = clock
        self._client: httpx.AsyncClient = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout_s,
            transport=transport,
            headers={"X-API-Key": self._api_key},
        )
        self._snapshot: SiteConfigSnapshot | None = None
        self._last_synced: datetime | None = None
        self._last_error: str | None = None

    def snapshot(self) -> SiteConfigSnapshot | None:
        return self._snapshot

    def last_synced_at(self) -> datetime | None:
        return self._last_synced

    @property
    def last_error(self) -> str | None:
        return self._last_error

    async def refresh(self) -> SiteConfigSnapshot | None:
        """Fetch and validate the latest site config; retain last good snapshot on error."""
        try:
            response = await self._client.get("/api/v1/wearables/site-config")
            response.raise_for_status()
            payload = response.json()
            snapshot = parse_site_config_payload(payload)
        except (httpx.HTTPError, ValueError, ValidationError) as exc:
            SITE_CONFIG_POLL_FAILURES_TOTAL.inc()
            self._last_error = type(exc).__name__
            logger.warning(
                "rf_site_config_poll_failed",
                error_type=type(exc).__name__,
                has_cached_snapshot=self._snapshot is not None,
            )
            return self._snapshot

        previous_version = self._snapshot.version if self._snapshot is not None else ""
        self._snapshot = snapshot
        self._last_synced = ensure_utc(self._clock())
        self._last_error = None
        if snapshot.version != previous_version:
            logger.info(
                "rf_site_config_updated",
                version=snapshot.version,
                floor_plan_id=snapshot.floor_plan_id,
                node_count=len(snapshot.nodes),
                device_count=len(snapshot.devices),
                binding_count=len(snapshot.bindings),
            )
        return self._snapshot

    async def close(self) -> None:
        await self._client.aclose()

"""Device-side ranging MQTT adapter (``device_ranging.v1`` payload profile).

Consumes micro-batched range epochs published by a wearable controller (such as
the Wear OS companion app in UWB burst mode) on ``rf/devices/{tag_id}/ranges``.
Vendor/wire payload schemas live strictly inside this module; emitted
:class:`rf_ingress.domain.RangeMeasurement` values carry technology, variance,
``sent_at`` (for burst clock-offset decoupling), and ``device_local`` azimuth
telemetry (ignored by the solver on 3+ anchors).
"""

from __future__ import annotations

import json
import math
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Literal

import structlog
from cts_contracts.wearables import (
    AdapterKind,
    FixKind,
    MotionState,
    RangingTechnology,
    WearState,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rf_ingress.adapters._mqtt_transport import (
    AiomqttMessageStream,
    InMemoryMqttMessageStream,
    MqttMessageStream,
)
from rf_ingress.adapters.base import AdapterDeps, AdapterHealth
from rf_ingress.config import AdapterConfig
from rf_ingress.domain import (
    AdapterCapabilityDescriptor,
    AzimuthReferenceFrame,
    RangeMeasurement,
    SiteConfigSnapshot,
)
from rf_ingress.observability import (
    FIXES_DROPPED_TOTAL,
    UNREGISTERED_TAG_TOTAL,
    DropReason,
)

logger = structlog.get_logger(__name__)

DEVICE_RANGING_ADAPTER_VERSION = "0.1.0"
DEVICE_RANGING_PROFILES: frozenset[str] = frozenset({"device_ranging_v1"})

_UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def unix_ns_to_utc(unix_ns: int) -> datetime:
    """Convert positive integer Unix nanoseconds to a timezone-aware UTC datetime."""
    if unix_ns <= 0:
        raise ValueError(f"Unix nanosecond timestamp must be positive, got {unix_ns}")
    seconds, rem_ns = divmod(unix_ns, 1_000_000_000)
    return _UNIX_EPOCH + timedelta(seconds=seconds, microseconds=rem_ns // 1_000)


__all__ = [
    "DEVICE_RANGING_ADAPTER_VERSION",
    "DEVICE_RANGING_PROFILES",
    "AiomqttMessageStream",
    "DeviceRangingMqttAdapter",
    "DeviceRangingMqttConfig",
    "DeviceRangingV1Epoch",
    "DeviceRangingV1Payload",
    "DeviceRangingV1Range",
    "InMemoryMqttMessageStream",
    "MqttMessageStream",
    "build_device_ranging_mqtt_adapter",
    "unix_ns_to_utc",
]


class DeviceRangingMqttConfig(BaseModel):
    """Adapter-specific configuration validated when ``kind == "device_ranging_mqtt"``."""

    model_config = ConfigDict(extra="forbid")

    payload_profile: str = "device_ranging_v1"
    broker_url: str = "mqtt://localhost:1883"
    topic: str = "rf/devices/+/ranges"
    username: str | None = None
    password: str | None = None
    nominal_rate_hz: float = Field(default=1.0, gt=0.0)
    uwb_default_sigma_m: float = Field(default=0.12, gt=0.0)
    ble_channel_sounding_default_sigma_m: float = Field(default=0.50, gt=0.0)
    ble_rssi_default_sigma_m: float = Field(default=2.0, gt=0.0)

    @model_validator(mode="after")
    def _validate_profile(self) -> DeviceRangingMqttConfig:
        if self.payload_profile not in DEVICE_RANGING_PROFILES:
            known = ", ".join(sorted(DEVICE_RANGING_PROFILES))
            raise ValueError(
                f"Unknown device_ranging_mqtt payload_profile {self.payload_profile!r}; "
                f"expected one of: {known}"
            )
        return self


class DeviceRangingV1Range(BaseModel):
    """One anchor range entry inside a ``device_ranging.v1`` epoch."""

    model_config = ConfigDict(extra="forbid")

    anchor_id: str = ""
    uwb_address_hex: str | None = None
    distance_m: float = Field(ge=0.0)
    variance_m2: float | None = Field(default=None, gt=0.0)
    technology: RangingTechnology = RangingTechnology.UWB
    azimuth_deg: float | None = None
    azimuth_variance_deg2: float | None = Field(default=None, gt=0.0)
    elevation_deg: float | None = None
    los_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_finite_and_anchor_ref(self) -> DeviceRangingV1Range:
        if not math.isfinite(self.distance_m):
            raise ValueError("distance_m must be finite")
        if self.variance_m2 is not None and not math.isfinite(self.variance_m2):
            raise ValueError("variance_m2 must be finite")
        if self.azimuth_deg is not None and not math.isfinite(self.azimuth_deg):
            raise ValueError("azimuth_deg must be finite")
        if self.elevation_deg is not None and not math.isfinite(self.elevation_deg):
            raise ValueError("elevation_deg must be finite")
        if not self.anchor_id and not self.uwb_address_hex:
            raise ValueError("Range entry requires either anchor_id or uwb_address_hex")
        return self


class DeviceRangingV1Epoch(BaseModel):
    """One ranging epoch inside a ``device_ranging.v1`` micro-batch."""

    model_config = ConfigDict(extra="forbid")

    epoch_seq: int = Field(ge=0)
    measured_at_unix_ns: int = Field(gt=0)
    ranges: list[DeviceRangingV1Range] = Field(min_length=1)


class DeviceRangingV1Payload(BaseModel):
    """Versioned ``device_ranging.v1`` MQTT batch payload published by a wearable."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["device_ranging.v1"] = Field(alias="schema")
    tag_id: str = Field(min_length=1, max_length=128)
    sent_at_unix_ns: int = Field(gt=0)
    wear_state: WearState = WearState.UNSPECIFIED
    motion_state: MotionState = MotionState.UNSPECIFIED
    battery_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    epochs: list[DeviceRangingV1Epoch] = Field(min_length=1)


class DeviceRangingMqttAdapter:
    """MQTT :class:`RangeAdapter` unpacking ``device_ranging.v1`` micro-batches."""

    def __init__(
        self,
        *,
        adapter_id: str,
        deps: AdapterDeps,
        stream: MqttMessageStream,
        default_sigma_by_tech: dict[RangingTechnology, float] | None = None,
        nominal_rate_hz: float = 1.0,
        ntp_synced: bool = True,
        latency_s: float = 0.0,
    ) -> None:
        self.adapter_id: str = adapter_id
        self._deps: AdapterDeps = deps
        self._stream: MqttMessageStream = stream
        self._default_sigma_by_tech: dict[RangingTechnology, float] = default_sigma_by_tech or {
            RangingTechnology.UWB: 0.12,
            RangingTechnology.BLE_CHANNEL_SOUNDING: 0.50,
            RangingTechnology.BLE_RSSI: 2.0,
        }
        self._closed: bool = False
        self._last_report_at: datetime | None = None
        self._malformed_reports: int = 0
        self._unregistered_reports: int = 0

        self.capability: AdapterCapabilityDescriptor = AdapterCapabilityDescriptor(
            adapter_id=self.adapter_id,
            adapter_kind=AdapterKind.DEVICE_RANGING,
            adapter_version=DEVICE_RANGING_ADAPTER_VERSION,
            fix_kinds=(FixKind.POSITION, FixKind.ROOM),
            nominal_rate_hz=nominal_rate_hz,
            reports_covariance=True,
            reports_wear_state=True,
            reports_z=True,
            reports_motion_state=True,
            ntp_synced=ntp_synced,
            latency_s=latency_s,
        )

    def health(self) -> AdapterHealth:
        return AdapterHealth(
            adapter_id=self.adapter_id,
            connected=not self._closed,
            last_report_at=self._last_report_at,
            malformed_reports=self._malformed_reports,
            unregistered_reports=self._unregistered_reports,
        )

    async def close(self) -> None:
        self._closed = True
        await self._stream.close()

    def _resolve_registered_tag_id(
        self,
        wire_tag_id: str,
        snapshot: SiteConfigSnapshot | None,
    ) -> str | None:
        if snapshot is None:
            return wire_tag_id
        by_tag = snapshot.device_by_tag_id(wire_tag_id)
        if by_tag is not None:
            return by_tag.tag_id
        by_hw = snapshot.device_by_hardware_key(self.adapter_id, wire_tag_id)
        if by_hw is not None:
            return by_hw.tag_id
        self._unregistered_reports += 1
        UNREGISTERED_TAG_TOTAL.labels(adapter_id=self.adapter_id).inc()
        FIXES_DROPPED_TOTAL.labels(reason=DropReason.UNREGISTERED_TAG).inc()
        return None

    def _resolve_anchor_id(
        self,
        entry: DeviceRangingV1Range,
        snapshot: SiteConfigSnapshot | None,
    ) -> str | None:
        if entry.anchor_id:
            return entry.anchor_id
        if entry.uwb_address_hex and snapshot is not None:
            wanted = entry.uwb_address_hex.strip().upper()
            for node in snapshot.nodes:
                if node.uwb_address_hex and node.uwb_address_hex.strip().upper() == wanted:
                    return node.anchor_id
        return None

    def _variance_for(self, entry: DeviceRangingV1Range) -> float:
        if entry.variance_m2 is not None:
            return float(entry.variance_m2)
        sigma = self._default_sigma_by_tech.get(entry.technology, 0.50)
        return sigma * sigma

    def _record_malformed(self, reason: str) -> None:
        self._malformed_reports += 1
        FIXES_DROPPED_TOTAL.labels(reason=DropReason.MALFORMED_INPUT).inc()
        logger.warning(
            "rf_device_ranging_malformed_payload",
            adapter_id=self.adapter_id,
            reason=reason,
        )

    async def ranges(self) -> AsyncIterator[RangeMeasurement]:
        async for _topic, raw_payload in self._stream.messages():
            if self._closed:
                break
            try:
                decoded = json.loads(raw_payload.decode("utf-8"))
                payload = DeviceRangingV1Payload.model_validate(decoded)
                sent_at = unix_ns_to_utc(payload.sent_at_unix_ns)
            except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
                self._record_malformed(type(exc).__name__)
                continue

            snapshot = self._deps.site_config.snapshot()
            canonical_tag_id = self._resolve_registered_tag_id(payload.tag_id, snapshot)
            if canonical_tag_id is None:
                continue

            for epoch in payload.epochs:
                if self._closed:
                    break
                try:
                    measured_at = unix_ns_to_utc(epoch.measured_at_unix_ns)
                except ValueError as exc:
                    self._record_malformed(type(exc).__name__)
                    continue

                for entry in epoch.ranges:
                    if self._closed:
                        break
                    anchor_id = self._resolve_anchor_id(entry, snapshot)
                    if anchor_id is None:
                        self._record_malformed("unresolved_anchor_id")
                        continue
                    az_frame: AzimuthReferenceFrame | None = (
                        "device_local" if entry.azimuth_deg is not None else None
                    )
                    self._last_report_at = self._deps.clock()
                    yield RangeMeasurement(
                        tag_id=canonical_tag_id,
                        anchor_id=anchor_id,
                        distance_m=entry.distance_m,
                        variance_m2=self._variance_for(entry),
                        technology=entry.technology,
                        measured_at=measured_at,
                        los_confidence=entry.los_confidence,
                        azimuth_deg=entry.azimuth_deg,
                        azimuth_variance_deg2=entry.azimuth_variance_deg2,
                        elevation_deg=entry.elevation_deg,
                        azimuth_reference_frame=az_frame,
                        wear_state=payload.wear_state,
                        motion_state=payload.motion_state,
                        battery_fraction=payload.battery_fraction,
                        adapter_id=self.adapter_id,
                        epoch_seq=epoch.epoch_seq,
                        sent_at=sent_at,
                    )


def build_device_ranging_mqtt_adapter(
    config: AdapterConfig,
    deps: AdapterDeps,
) -> DeviceRangingMqttAdapter:
    """Registry factory for ``kind: device_ranging_mqtt``."""
    raw = config.extra_fields()
    if config.payload_profile is not None:
        raw["payload_profile"] = config.payload_profile
    parsed = DeviceRangingMqttConfig.model_validate(raw)
    stream = AiomqttMessageStream(
        broker_url=parsed.broker_url,
        topic=parsed.topic,
        username=parsed.username,
        password=parsed.password,
    )
    return DeviceRangingMqttAdapter(
        adapter_id=config.id,
        deps=deps,
        stream=stream,
        default_sigma_by_tech={
            RangingTechnology.UWB: parsed.uwb_default_sigma_m,
            RangingTechnology.BLE_CHANNEL_SOUNDING: parsed.ble_channel_sounding_default_sigma_m,
            RangingTechnology.BLE_RSSI: parsed.ble_rssi_default_sigma_m,
        },
        nominal_rate_hz=parsed.nominal_rate_hz,
        ntp_synced=config.ntp_synced,
        latency_s=config.latency_s,
    )

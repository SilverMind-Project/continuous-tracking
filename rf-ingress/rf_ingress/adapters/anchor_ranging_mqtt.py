"""Anchor-side UWB ranging MQTT adapter (``fira_anchor_v1`` payload profile).

Consumes per-anchor range reports published over LAN MQTT by mains-powered FiRa
MAC 1.3 UWB anchors (on ``rf/anchors/+/ranges``) for standalone deep-sleep tags.
Reporting ranges from wall-powered anchors keeps battery tags UWB-only without
waking Wi-Fi on each ranging round.

Vendor/wire payload schemas stay strictly inside this module. Emitted
:class:`rf_ingress.domain.RangeMeasurement` values carry ``technology = uwb``,
honest variance floored against datasheet overconfidence, and charging-safe
``wear_state`` (a charging tag always reports ``WearState.OFF_BODY``).
"""

from __future__ import annotations

import json
import math
from collections.abc import AsyncIterator, Mapping
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

ANCHOR_RANGING_ADAPTER_VERSION = "0.1.0"
ANCHOR_RANGING_PROFILES: frozenset[str] = frozenset({"fira_anchor_v1"})

_UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _unix_ns_to_utc(unix_ns: int) -> datetime:
    """Convert positive integer Unix nanoseconds to a timezone-aware UTC datetime."""
    if unix_ns <= 0:
        raise ValueError(f"Unix nanosecond timestamp must be positive, got {unix_ns}")
    seconds, rem_ns = divmod(unix_ns, 1_000_000_000)
    return _UNIX_EPOCH + timedelta(seconds=seconds, microseconds=rem_ns // 1_000)


__all__ = [
    "ANCHOR_RANGING_ADAPTER_VERSION",
    "ANCHOR_RANGING_PROFILES",
    "AiomqttMessageStream",
    "AnchorNodeCalibrationConfig",
    "AnchorRangingMqttAdapter",
    "AnchorRangingMqttConfig",
    "FiraAnchorReportV1",
    "InMemoryMqttMessageStream",
    "MqttMessageStream",
    "build_anchor_ranging_mqtt_adapter",
]


class AnchorNodeCalibrationConfig(BaseModel):
    """Optional per-anchor bench calibration overrides."""

    model_config = ConfigDict(extra="forbid")

    sigma_m: float | None = Field(default=None, gt=0.0)
    range_bias_m: float = 0.0
    calibrated_floor_plan_azimuth: bool | None = None

    @model_validator(mode="after")
    def _validate_finite(self) -> AnchorNodeCalibrationConfig:
        if self.sigma_m is not None and not math.isfinite(self.sigma_m):
            raise ValueError("sigma_m must be finite")
        if not math.isfinite(self.range_bias_m):
            raise ValueError("range_bias_m must be finite")
        return self


class AnchorRangingMqttConfig(BaseModel):
    """Adapter-specific configuration validated when ``kind == "anchor_ranging_mqtt"``."""

    model_config = ConfigDict(extra="forbid")

    payload_profile: str = "fira_anchor_v1"
    broker_url: str = "mqtt://localhost:1883"
    topic: str = "rf/anchors/+/ranges"
    username: str | None = None
    password: str | None = None
    nominal_rate_hz: float = Field(default=1.0, gt=0.0)
    # Conservative worn-tag default (includes body shadowing) until bench-calibrated.
    uwb_default_sigma_m: float = Field(default=0.25, gt=0.0)
    sigma_floor_m: float = Field(default=0.10, gt=0.0)
    max_range_m: float = Field(default=25.0, gt=0.0)
    calibrated_floor_plan_azimuth: bool = False
    nodes: dict[str, AnchorNodeCalibrationConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_profile_and_bounds(self) -> AnchorRangingMqttConfig:
        if self.payload_profile not in ANCHOR_RANGING_PROFILES:
            known = ", ".join(sorted(ANCHOR_RANGING_PROFILES))
            raise ValueError(
                f"Unknown anchor_ranging_mqtt payload_profile {self.payload_profile!r}; "
                f"expected one of: {known}"
            )
        for field_name in (
            "nominal_rate_hz",
            "uwb_default_sigma_m",
            "sigma_floor_m",
            "max_range_m",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if self.uwb_default_sigma_m < self.sigma_floor_m:
            raise ValueError(
                "uwb_default_sigma_m must be >= sigma_floor_m "
                f"({self.uwb_default_sigma_m} < {self.sigma_floor_m})"
            )
        return self


class FiraAnchorReportV1(BaseModel):
    """Versioned ``fira_anchor.v1`` MQTT range report published by one fixed UWB anchor."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["fira_anchor.v1"] = Field(alias="schema")
    anchor_id: str = Field(default="", max_length=128)
    uwb_address_hex: str | None = Field(default=None, max_length=16)
    tag_id: str = Field(default="", max_length=128)
    hardware_key: str | None = Field(default=None, max_length=128)
    epoch_seq: int | None = Field(default=None, ge=0)
    measured_at_unix_ns: int = Field(gt=0)
    sent_at_unix_ns: int | None = Field(default=None, gt=0)
    distance_m: float = Field(ge=0.0)
    variance_m2: float | None = Field(default=None, gt=0.0)
    los_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    azimuth_deg: float | None = None
    azimuth_variance_deg2: float | None = Field(default=None, gt=0.0)
    elevation_deg: float | None = None
    wear_state: WearState = WearState.UNSPECIFIED
    motion_state: MotionState = MotionState.UNSPECIFIED
    charging: bool = False
    battery_fraction: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_refs_and_floats(self) -> FiraAnchorReportV1:
        for field_name in (
            "distance_m",
            "variance_m2",
            "los_confidence",
            "azimuth_deg",
            "azimuth_variance_deg2",
            "elevation_deg",
            "battery_fraction",
        ):
            value = getattr(self, field_name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        has_anchor_id = bool(self.anchor_id.strip())
        has_uwb_addr = bool(self.uwb_address_hex and self.uwb_address_hex.strip())
        if not has_anchor_id and not has_uwb_addr:
            raise ValueError("Report requires either anchor_id or uwb_address_hex")
        has_tag_id = bool(self.tag_id.strip())
        has_hw_key = bool(self.hardware_key and self.hardware_key.strip())
        if not has_tag_id and not has_hw_key:
            raise ValueError("Report requires either tag_id or hardware_key")
        return self


class AnchorRangingMqttAdapter:
    """MQTT :class:`RangeAdapter` unpacking ``fira_anchor.v1`` anchor range reports."""

    def __init__(
        self,
        *,
        adapter_id: str,
        deps: AdapterDeps,
        stream: MqttMessageStream,
        uwb_default_sigma_m: float = 0.25,
        sigma_floor_m: float = 0.10,
        max_range_m: float = 25.0,
        calibrated_floor_plan_azimuth: bool = False,
        node_calibrations: Mapping[str, AnchorNodeCalibrationConfig] | None = None,
        nominal_rate_hz: float = 1.0,
        ntp_synced: bool = True,
        latency_s: float = 0.0,
    ) -> None:
        if not math.isfinite(sigma_floor_m) or sigma_floor_m <= 0.0:
            raise ValueError(f"sigma_floor_m must be finite and > 0, got {sigma_floor_m!r}")
        if not math.isfinite(uwb_default_sigma_m) or uwb_default_sigma_m < sigma_floor_m:
            raise ValueError(
                f"uwb_default_sigma_m must be finite and >= sigma_floor_m, "
                f"got {uwb_default_sigma_m!r}"
            )
        if not math.isfinite(max_range_m) or max_range_m <= 0.0:
            raise ValueError(f"max_range_m must be finite and > 0, got {max_range_m!r}")

        self.adapter_id: str = adapter_id
        self._deps: AdapterDeps = deps
        self._stream: MqttMessageStream = stream
        self._uwb_default_sigma_m: float = uwb_default_sigma_m
        self._sigma_floor_m: float = sigma_floor_m
        self._max_range_m: float = max_range_m
        self._calibrated_floor_plan_azimuth: bool = calibrated_floor_plan_azimuth
        self._node_calibrations: dict[str, AnchorNodeCalibrationConfig] = dict(
            node_calibrations or {}
        )
        self._closed: bool = False
        self._last_report_at: datetime | None = None
        self._malformed_reports: int = 0
        self._unregistered_reports: int = 0

        self.capability: AdapterCapabilityDescriptor = AdapterCapabilityDescriptor(
            adapter_id=self.adapter_id,
            adapter_kind=AdapterKind.ANCHOR_RANGING,
            adapter_version=ANCHOR_RANGING_ADAPTER_VERSION,
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
        payload: FiraAnchorReportV1,
        snapshot: SiteConfigSnapshot | None,
    ) -> str | None:
        hw_key = payload.hardware_key.strip() if payload.hardware_key else ""
        tag_id = payload.tag_id.strip()
        if snapshot is None:
            return tag_id or hw_key
        if hw_key:
            by_hw = snapshot.device_by_hardware_key(self.adapter_id, hw_key)
            if by_hw is not None:
                return by_hw.tag_id
        if tag_id:
            by_tag = snapshot.device_by_tag_id(tag_id)
            if by_tag is not None:
                return by_tag.tag_id
            by_tag_as_hw = snapshot.device_by_hardware_key(self.adapter_id, tag_id)
            if by_tag_as_hw is not None:
                return by_tag_as_hw.tag_id
        self._unregistered_reports += 1
        UNREGISTERED_TAG_TOTAL.labels(adapter_id=self.adapter_id).inc()
        FIXES_DROPPED_TOTAL.labels(reason=DropReason.UNREGISTERED_TAG).inc()
        return None

    def _resolve_anchor_id(
        self,
        payload: FiraAnchorReportV1,
        snapshot: SiteConfigSnapshot | None,
    ) -> str | None:
        anchor_id = payload.anchor_id.strip()
        if anchor_id:
            return anchor_id
        if payload.uwb_address_hex and snapshot is not None:
            wanted = payload.uwb_address_hex.strip().upper()
            for node in snapshot.nodes:
                if node.uwb_address_hex and node.uwb_address_hex.strip().upper() == wanted:
                    return node.anchor_id
        return None

    def _variance_for(self, payload: FiraAnchorReportV1, anchor_id: str) -> float:
        floor_var = self._sigma_floor_m * self._sigma_floor_m
        if payload.variance_m2 is not None:
            return max(float(payload.variance_m2), floor_var)
        node_cal = self._node_calibrations.get(anchor_id)
        sigma_m = (
            node_cal.sigma_m
            if node_cal is not None and node_cal.sigma_m is not None
            else self._uwb_default_sigma_m
        )
        return max(sigma_m * sigma_m, floor_var)

    def _azimuth_frame_for(
        self,
        payload: FiraAnchorReportV1,
        anchor_id: str,
    ) -> AzimuthReferenceFrame | None:
        if payload.azimuth_deg is None:
            return None
        node_cal = self._node_calibrations.get(anchor_id)
        calibrated = (
            node_cal.calibrated_floor_plan_azimuth
            if node_cal is not None and node_cal.calibrated_floor_plan_azimuth is not None
            else self._calibrated_floor_plan_azimuth
        )
        if calibrated and payload.azimuth_variance_deg2 is not None:
            return "floor_plan"
        return "device_local"

    def _record_malformed(self, reason: str) -> None:
        self._malformed_reports += 1
        FIXES_DROPPED_TOTAL.labels(reason=DropReason.MALFORMED_INPUT).inc()
        logger.warning(
            "rf_anchor_ranging_malformed_payload",
            adapter_id=self.adapter_id,
            reason=reason,
        )

    async def ranges(self) -> AsyncIterator[RangeMeasurement]:
        async for _topic, raw_payload in self._stream.messages():
            if self._closed:
                break
            try:
                decoded = json.loads(raw_payload.decode("utf-8"))
                payload = FiraAnchorReportV1.model_validate(decoded)
                measured_at = _unix_ns_to_utc(payload.measured_at_unix_ns)
                sent_at = (
                    _unix_ns_to_utc(payload.sent_at_unix_ns)
                    if payload.sent_at_unix_ns is not None
                    else None
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
                self._record_malformed(type(exc).__name__)
                continue

            snapshot = self._deps.site_config.snapshot()
            canonical_tag_id = self._resolve_registered_tag_id(payload, snapshot)
            if canonical_tag_id is None:
                continue

            anchor_id = self._resolve_anchor_id(payload, snapshot)
            if anchor_id is None:
                self._record_malformed("unresolved_anchor_id")
                continue

            node_cal = self._node_calibrations.get(anchor_id)
            bias_m = node_cal.range_bias_m if node_cal is not None else 0.0
            corrected_distance_m = max(0.0, payload.distance_m - bias_m)
            if corrected_distance_m > self._max_range_m:
                FIXES_DROPPED_TOTAL.labels(reason=DropReason.RANGE_OUT_OF_BOUNDS).inc()
                continue

            # A tag docked on its charger must never be reported as on-body evidence.
            effective_wear_state = WearState.OFF_BODY if payload.charging else payload.wear_state
            self._last_report_at = self._deps.clock()

            yield RangeMeasurement(
                tag_id=canonical_tag_id,
                anchor_id=anchor_id,
                distance_m=corrected_distance_m,
                variance_m2=self._variance_for(payload, anchor_id),
                technology=RangingTechnology.UWB,
                measured_at=measured_at,
                los_confidence=payload.los_confidence,
                azimuth_deg=payload.azimuth_deg,
                azimuth_variance_deg2=payload.azimuth_variance_deg2,
                elevation_deg=payload.elevation_deg,
                azimuth_reference_frame=self._azimuth_frame_for(payload, anchor_id),
                wear_state=effective_wear_state,
                motion_state=payload.motion_state,
                battery_fraction=payload.battery_fraction,
                adapter_id=self.adapter_id,
                epoch_seq=payload.epoch_seq,
                sent_at=sent_at,
            )


def build_anchor_ranging_mqtt_adapter(
    config: AdapterConfig,
    deps: AdapterDeps,
) -> AnchorRangingMqttAdapter:
    """Registry factory for ``kind: anchor_ranging_mqtt``."""
    raw = config.extra_fields()
    if config.payload_profile is not None:
        raw["payload_profile"] = config.payload_profile
    parsed = AnchorRangingMqttConfig.model_validate(raw)
    stream = AiomqttMessageStream(
        broker_url=parsed.broker_url,
        topic=parsed.topic,
        username=parsed.username,
        password=parsed.password,
    )
    return AnchorRangingMqttAdapter(
        adapter_id=config.id,
        deps=deps,
        stream=stream,
        uwb_default_sigma_m=parsed.uwb_default_sigma_m,
        sigma_floor_m=parsed.sigma_floor_m,
        max_range_m=parsed.max_range_m,
        calibrated_floor_plan_azimuth=parsed.calibrated_floor_plan_azimuth,
        node_calibrations=parsed.nodes,
        nominal_rate_hz=parsed.nominal_rate_hz,
        ntp_synced=config.ntp_synced,
        latency_s=config.latency_s,
    )

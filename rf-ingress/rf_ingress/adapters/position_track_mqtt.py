"""MQTT adapter for anonymous position sensors (``position_track_v1`` payload profile).

Consumes JSON track reports published by edge mmWave radar and room LiDAR gateways on
``sensors/position/+/tracks``. Enforces strict Pydantic schemas (``extra="forbid"``),
topic-vs-payload ``sensor_id`` matching, fail-closed site-config sensor registration, and
finite positive-definite 2x2 covariances before yielding :class:`AnonymousPositionTrackReport`.
"""

from __future__ import annotations

import json
import math
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Literal

import structlog
from cts_contracts.position_sensors import is_valid_sensor_id
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rf_ingress.adapters._mqtt_transport import (
    AiomqttMessageStream,
    MqttMessageStream,
)
from rf_ingress.adapters.base import AdapterDeps, AdapterHealth
from rf_ingress.config import AdapterConfig
from rf_ingress.domain import AnonymousPositionTrackReport, PositionTrackFrame
from rf_ingress.observability import (
    POSITION_TRACKS_DROPPED_TOTAL,
    PositionTrackDropReason,
)

logger = structlog.get_logger(__name__)

POSITION_TRACK_PROFILES: frozenset[str] = frozenset({"position_track_v1"})
_UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_MIN_COV_DET_M4 = 1e-9


def _unix_ns_to_utc(unix_ns: int) -> datetime:
    if unix_ns <= 0:
        raise ValueError(f"Unix nanosecond timestamp must be positive, got {unix_ns}")
    seconds, rem_ns = divmod(unix_ns, 1_000_000_000)
    return _UNIX_EPOCH + timedelta(seconds=seconds, microseconds=rem_ns // 1_000)


def _topic_sensor_id(topic: str) -> str | None:
    parts = topic.split("/")
    if (
        len(parts) == 4
        and parts[0] == "sensors"
        and parts[1] == "position"
        and parts[3] == "tracks"
    ):
        seg = parts[2].strip()
        if seg and seg != "+":
            return seg
    return None


class PositionTrackMqttConfig(BaseModel):
    """Adapter-specific configuration validated when ``kind == "position_track_mqtt"``."""

    model_config = ConfigDict(extra="forbid")

    payload_profile: str = "position_track_v1"
    broker_url: str = "mqtt://localhost:1883"
    topic: str = "sensors/position/+/tracks"
    username: str | None = None
    password: str | None = None

    @model_validator(mode="after")
    def _validate_profile(self) -> PositionTrackMqttConfig:
        if self.payload_profile not in POSITION_TRACK_PROFILES:
            known = ", ".join(sorted(POSITION_TRACK_PROFILES))
            raise ValueError(
                f"Unknown position_track_mqtt payload_profile {self.payload_profile!r}; "
                f"expected one of: {known}"
            )
        return self


class PositionTrackItemV1(BaseModel):
    """Single target track inside a ``position_track.v1`` MQTT payload."""

    model_config = ConfigDict(extra="forbid")

    sensor_track_id: str = Field(min_length=1, max_length=128)
    measured_at_unix_ns: int = Field(gt=0)
    x_m: float
    y_m: float
    cov_xx_m2: float
    cov_xy_m2: float = 0.0
    cov_yy_m2: float
    speed_m_s: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _validate_floats(self) -> PositionTrackItemV1:
        for field_name in ("x_m", "y_m", "cov_xx_m2", "cov_xy_m2", "cov_yy_m2", "speed_m_s"):
            val = getattr(self, field_name)
            if val is not None and not math.isfinite(val):
                raise ValueError(f"{field_name} must be finite")
        return self


class PositionTrackEnvelopeV1(BaseModel):
    """Versioned ``position_track.v1`` batch published by a position-sensor gateway."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["position_track.v1"] = Field(alias="schema")
    sensor_id: str = Field(min_length=1, max_length=64)
    floor_plan_id: str = ""
    frame: PositionTrackFrame = "floor_plan"
    sent_at_unix_ns: int | None = Field(default=None, gt=0)
    tracks: list[PositionTrackItemV1]

    @model_validator(mode="after")
    def _validate_sensor_id(self) -> PositionTrackEnvelopeV1:
        if not is_valid_sensor_id(self.sensor_id):
            raise ValueError(f"Invalid sensor_id slug: {self.sensor_id!r}")
        return self


class PositionTrackMqttAdapter:
    """MQTT :class:`PositionTrackAdapter` unpacking ``position_track.v1`` payloads."""

    def __init__(
        self,
        *,
        adapter_id: str,
        deps: AdapterDeps,
        stream: MqttMessageStream,
    ) -> None:
        self.adapter_id: str = adapter_id
        self._deps: AdapterDeps = deps
        self._stream: MqttMessageStream = stream
        self._closed: bool = False
        self._last_report_at: datetime | None = None
        self._malformed_reports: int = 0
        self._unregistered_reports: int = 0

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

    def _record_malformed(self, reason: str) -> None:
        self._malformed_reports += 1
        POSITION_TRACKS_DROPPED_TOTAL.labels(reason=PositionTrackDropReason.MALFORMED_INPUT).inc()
        logger.warning(
            "rf_position_track_malformed_payload",
            adapter_id=self.adapter_id,
            reason=reason,
        )

    async def position_tracks(self) -> AsyncIterator[AnonymousPositionTrackReport]:
        async for topic, raw_payload in self._stream.messages():
            if self._closed:
                break
            try:
                decoded = json.loads(raw_payload.decode("utf-8"))
                envelope = PositionTrackEnvelopeV1.model_validate(decoded)
                sent_at = (
                    _unix_ns_to_utc(envelope.sent_at_unix_ns)
                    if envelope.sent_at_unix_ns is not None
                    else None
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
                self._record_malformed(type(exc).__name__)
                continue

            topic_sid = _topic_sensor_id(topic)
            if topic_sid is not None and topic_sid != envelope.sensor_id:
                self._malformed_reports += 1
                POSITION_TRACKS_DROPPED_TOTAL.labels(
                    reason=PositionTrackDropReason.TOPIC_SENSOR_MISMATCH
                ).inc()
                logger.warning(
                    "rf_position_track_topic_sensor_mismatch",
                    adapter_id=self.adapter_id,
                )
                continue

            snapshot = self._deps.site_config.snapshot()
            if snapshot is None:
                self._unregistered_reports += 1
                POSITION_TRACKS_DROPPED_TOTAL.labels(
                    reason=PositionTrackDropReason.NO_SITE_CONFIG
                ).inc()
                continue

            if snapshot.position_sensor(envelope.sensor_id) is None:
                self._unregistered_reports += 1
                POSITION_TRACKS_DROPPED_TOTAL.labels(
                    reason=PositionTrackDropReason.UNREGISTERED_SENSOR
                ).inc()
                continue

            for item in envelope.tracks:
                det = item.cov_xx_m2 * item.cov_yy_m2 - item.cov_xy_m2 * item.cov_xy_m2
                if item.cov_xx_m2 <= 0.0 or item.cov_yy_m2 <= 0.0 or det <= _MIN_COV_DET_M4:
                    self._malformed_reports += 1
                    POSITION_TRACKS_DROPPED_TOTAL.labels(
                        reason=PositionTrackDropReason.INVALID_COVARIANCE
                    ).inc()
                    continue

                try:
                    measured_at = _unix_ns_to_utc(item.measured_at_unix_ns)
                except ValueError as exc:
                    self._record_malformed(type(exc).__name__)
                    continue

                self._last_report_at = self._deps.clock()
                yield AnonymousPositionTrackReport(
                    sensor_id=envelope.sensor_id,
                    sensor_track_id=item.sensor_track_id,
                    measured_at=measured_at,
                    x_m=item.x_m,
                    y_m=item.y_m,
                    cov_xx_m2=item.cov_xx_m2,
                    cov_xy_m2=item.cov_xy_m2,
                    cov_yy_m2=item.cov_yy_m2,
                    speed_m_s=item.speed_m_s,
                    frame=envelope.frame,
                    floor_plan_id=envelope.floor_plan_id,
                    adapter_id=self.adapter_id,
                    sent_at=sent_at,
                )


def build_position_track_mqtt_adapter(
    config: AdapterConfig,
    deps: AdapterDeps,
) -> PositionTrackMqttAdapter:
    """Registry factory for ``kind: position_track_mqtt``."""
    raw = config.extra_fields()
    if config.payload_profile is not None:
        raw["payload_profile"] = config.payload_profile
    parsed = PositionTrackMqttConfig.model_validate(raw)
    stream = AiomqttMessageStream(
        broker_url=parsed.broker_url,
        topic=parsed.topic,
        username=parsed.username,
        password=parsed.password,
    )
    return PositionTrackMqttAdapter(
        adapter_id=config.id,
        deps=deps,
        stream=stream,
    )

"""Unit tests for anonymous position-track transform, MQTT validation, and runtime wiring."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from rf_ingress.adapters._mqtt_transport import InMemoryMqttMessageStream
from rf_ingress.adapters.base import AdapterDeps
from rf_ingress.adapters.position_track_mqtt import PositionTrackMqttAdapter
from rf_ingress.adapters.registry import build_default_registry
from rf_ingress.config import AdapterConfig
from rf_ingress.domain import (
    AnonymousPositionTrackReport,
    PositionSensorConfig,
    SiteConfigSnapshot,
)
from rf_ingress.observability import (
    POSITION_TRACKS_DROPPED_TOTAL,
    POSITION_TRACKS_PUBLISHED_TOTAL,
    PositionTrackDropReason,
)
from rf_ingress.pipeline import RfIngressRuntime
from rf_ingress.position_tracks import (
    DEFAULT_POSITION_TRACK_RETENTION_HOURS,
    InMemoryPositionTrackPublisher,
    PositionTrackDropError,
    transform_position_track,
)
from rf_ingress.publish import InMemoryTagFixPublisher, compute_minid_ms, datetime_to_unix_ns
from rf_ingress.site_config import InMemorySiteConfigProvider, parse_site_config_payload

_T0 = datetime(2026, 11, 25, 14, 0, 0, tzinfo=UTC)


def _site(yaw_deg: float | None = 90.0) -> SiteConfigSnapshot:
    return SiteConfigSnapshot(
        floor_plan_id="fp-main",
        position_sensors=(
            PositionSensorConfig(
                sensor_id="radar-bath-1",
                room_id="bathroom",
                x_m=2.0,
                y_m=3.0,
                yaw_deg=yaw_deg,
                calibration_residual_m=0.10,
            ),
        ),
        version="sha256:v1",
    )


def test_parse_site_config_includes_position_sensors() -> None:
    snap = parse_site_config_payload(
        {
            "floor_plan_id": "fp-main",
            "nodes": [],
            "bindings": [],
            "devices": [],
            "position_sensors": [
                {
                    "sensor_id": "radar-bath-1",
                    "room_id": "2",
                    "x_m": 2.0,
                    "y_m": 3.0,
                    "yaw_deg": 90.0,
                    "calibration_residual_m": 0.12,
                }
            ],
            "version": "v1",
        }
    )
    sensor = snap.position_sensor("radar-bath-1")
    assert sensor is not None
    assert sensor.x_m == pytest.approx(2.0)
    assert sensor.yaw_deg == pytest.approx(90.0)
    assert sensor.calibration_residual_m == pytest.approx(0.12)
    assert snap.position_sensor("missing") is None


def test_transform_floor_plan_and_sensor_local_yaw_rotation() -> None:
    sensor = _site(yaw_deg=90.0).position_sensor("radar-bath-1")
    assert sensor is not None

    fp_report = AnonymousPositionTrackReport(
        sensor_id="radar-bath-1",
        sensor_track_id="t-1",
        measured_at=_T0,
        x_m=2.45,
        y_m=4.10,
        cov_xx_m2=0.04,
        cov_xy_m2=0.0,
        cov_yy_m2=0.04,
        frame="floor_plan",
    )
    emitted_fp = transform_position_track(fp_report, sensor, site_floor_plan_id="fp-main")
    assert emitted_fp.x_mm == 2450
    assert emitted_fp.y_mm == 4100
    assert emitted_fp.floor_plan_id == "fp-main"
    assert emitted_fp.cov_xx_m2 == pytest.approx(0.04)

    # 90 deg CCW rotation: (x=1.0, y=0.5) -> (-0.5, 1.0), plus mount (2.0, 3.0) -> (1.5, 4.0).
    # Covariance R @ [[0.09, 0.01], [0.01, 0.01]] @ R^T -> [[0.01, -0.01], [-0.01, 0.09]].
    local_report = AnonymousPositionTrackReport(
        sensor_id="radar-bath-1",
        sensor_track_id="t-2",
        measured_at=_T0,
        x_m=1.0,
        y_m=0.5,
        cov_xx_m2=0.09,
        cov_xy_m2=0.01,
        cov_yy_m2=0.01,
        frame="sensor_local",
        speed_m_s=0.4,
    )
    emitted_local = transform_position_track(local_report, sensor, site_floor_plan_id="fp-main")
    assert emitted_local.x_mm == 1500
    assert emitted_local.y_mm == 4000
    assert emitted_local.cov_xx_m2 == pytest.approx(0.01)
    assert emitted_local.cov_xy_m2 == pytest.approx(-0.01)
    assert emitted_local.cov_yy_m2 == pytest.approx(0.09)
    assert emitted_local.speed_m_s == pytest.approx(0.4)


def test_transform_rejects_missing_yaw_non_pd_cov_and_wrong_floor_plan() -> None:
    sensor_no_yaw = _site(yaw_deg=None).position_sensor("radar-bath-1")
    assert sensor_no_yaw is not None

    with pytest.raises(PositionTrackDropError) as exc_yaw:
        transform_position_track(
            AnonymousPositionTrackReport(
                sensor_id="radar-bath-1",
                sensor_track_id="t-1",
                measured_at=_T0,
                x_m=1.0,
                y_m=0.0,
                cov_xx_m2=0.04,
                cov_xy_m2=0.0,
                cov_yy_m2=0.04,
                frame="sensor_local",
            ),
            sensor_no_yaw,
            site_floor_plan_id="fp-main",
        )
    assert exc_yaw.value.reason == PositionTrackDropReason.MISSING_SENSOR_YAW

    with pytest.raises(PositionTrackDropError) as exc_cov:
        transform_position_track(
            AnonymousPositionTrackReport(
                sensor_id="radar-bath-1",
                sensor_track_id="t-1",
                measured_at=_T0,
                x_m=1.0,
                y_m=0.0,
                cov_xx_m2=0.04,
                cov_xy_m2=0.04,
                cov_yy_m2=0.04,
            ),
            sensor_no_yaw,
            site_floor_plan_id="fp-main",
        )
    assert exc_cov.value.reason == PositionTrackDropReason.INVALID_COVARIANCE

    with pytest.raises(PositionTrackDropError) as exc_fp:
        transform_position_track(
            AnonymousPositionTrackReport(
                sensor_id="radar-bath-1",
                sensor_track_id="t-1",
                measured_at=_T0,
                x_m=1.0,
                y_m=0.0,
                cov_xx_m2=0.04,
                cov_xy_m2=0.0,
                cov_yy_m2=0.04,
                floor_plan_id="other-floor",
            ),
            sensor_no_yaw,
            site_floor_plan_id="fp-main",
        )
    assert exc_fp.value.reason == PositionTrackDropReason.FLOOR_PLAN_MISMATCH


def test_unknown_payload_profile_fails_registry_startup() -> None:
    registry = build_default_registry()
    deps = AdapterDeps(site_config=InMemorySiteConfigProvider())
    with pytest.raises(ValidationError, match="Unknown position_track_mqtt payload_profile"):
        registry.build(
            AdapterConfig(
                id="pos-1",
                kind="position_track_mqtt",
                payload_profile="unknown_profile_v2",
            ),
            deps,
        )


@pytest.mark.asyncio
async def test_mqtt_adapter_drops_malformed_mismatch_unregistered_and_no_config() -> None:
    provider = InMemorySiteConfigProvider(None)
    ns = datetime_to_unix_ns(_T0)

    def _msg(topic: str, body: dict[str, object]) -> tuple[str, bytes]:
        return topic, json.dumps(body).encode("utf-8")

    valid_track = {
        "sensor_track_id": "t-01",
        "measured_at_unix_ns": ns,
        "x_m": 2.45,
        "y_m": 4.10,
        "cov_xx_m2": 0.0225,
        "cov_xy_m2": 0.0,
        "cov_yy_m2": 0.0225,
        "speed_m_s": 0.35,
    }

    # 1. Fail closed before first site-config snapshot.
    stream_no_cfg = InMemoryMqttMessageStream(
        [
            _msg(
                "sensors/position/radar-bath-1/tracks",
                {
                    "schema": "position_track.v1",
                    "sensor_id": "radar-bath-1",
                    "tracks": [valid_track],
                },
            )
        ]
    )
    adapter_no_cfg = PositionTrackMqttAdapter(
        adapter_id="mqtt-pos",
        deps=AdapterDeps(site_config=provider, clock=lambda: _T0),
        stream=stream_no_cfg,
    )
    before_no_cfg = POSITION_TRACKS_DROPPED_TOTAL.labels(
        reason=PositionTrackDropReason.NO_SITE_CONFIG
    )._value.get()
    emitted_none = [r async for r in adapter_no_cfg.position_tracks()]
    assert emitted_none == []
    assert (
        POSITION_TRACKS_DROPPED_TOTAL.labels(
            reason=PositionTrackDropReason.NO_SITE_CONFIG
        )._value.get()
        > before_no_cfg
    )

    # 2. Loaded site-config: extra-forbid key, topic mismatch, unregistered, singular cov, valid.
    provider.set_snapshot(_site())
    stream = InMemoryMqttMessageStream(
        [
            (
                "sensors/position/radar-bath-1/tracks",
                json.dumps(
                    {
                        "schema": "position_track.v1",
                        "sensor_id": "radar-bath-1",
                        " floor_plan_id": "fp-main",
                        "tracks": [valid_track],
                    }
                ).encode("utf-8"),
            ),
            _msg(
                "sensors/position/wrong-topic/tracks",
                {
                    "schema": "position_track.v1",
                    "sensor_id": "radar-bath-1",
                    "tracks": [valid_track],
                },
            ),
            _msg(
                "sensors/position/unregistered-1/tracks",
                {
                    "schema": "position_track.v1",
                    "sensor_id": "unregistered-1",
                    "tracks": [valid_track],
                },
            ),
            _msg(
                "sensors/position/radar-bath-1/tracks",
                {
                    "schema": "position_track.v1",
                    "sensor_id": "radar-bath-1",
                    "tracks": [
                        {
                            **valid_track,
                            "sensor_track_id": "bad-cov",
                            "cov_xx_m2": 0.04,
                            "cov_xy_m2": 0.04,
                            "cov_yy_m2": 0.04,
                        },
                        valid_track,
                    ],
                },
            ),
        ]
    )
    before_topic = POSITION_TRACKS_DROPPED_TOTAL.labels(
        reason=PositionTrackDropReason.TOPIC_SENSOR_MISMATCH
    )._value.get()
    before_unreg = POSITION_TRACKS_DROPPED_TOTAL.labels(
        reason=PositionTrackDropReason.UNREGISTERED_SENSOR
    )._value.get()
    before_cov = POSITION_TRACKS_DROPPED_TOTAL.labels(
        reason=PositionTrackDropReason.INVALID_COVARIANCE
    )._value.get()
    before_mal = POSITION_TRACKS_DROPPED_TOTAL.labels(
        reason=PositionTrackDropReason.MALFORMED_INPUT
    )._value.get()

    adapter = PositionTrackMqttAdapter(
        adapter_id="mqtt-pos",
        deps=AdapterDeps(site_config=provider, clock=lambda: _T0),
        stream=stream,
    )
    reports = [r async for r in adapter.position_tracks()]
    assert len(reports) == 1
    assert reports[0].sensor_id == "radar-bath-1"
    assert reports[0].sensor_track_id == "t-01"
    assert adapter.health().malformed_reports == 3
    assert adapter.health().unregistered_reports == 1
    assert (
        POSITION_TRACKS_DROPPED_TOTAL.labels(
            reason=PositionTrackDropReason.MALFORMED_INPUT
        )._value.get()
        == before_mal + 1
    )
    assert (
        POSITION_TRACKS_DROPPED_TOTAL.labels(
            reason=PositionTrackDropReason.TOPIC_SENSOR_MISMATCH
        )._value.get()
        == before_topic + 1
    )
    assert (
        POSITION_TRACKS_DROPPED_TOTAL.labels(
            reason=PositionTrackDropReason.UNREGISTERED_SENSOR
        )._value.get()
        == before_unreg + 1
    )
    assert (
        POSITION_TRACKS_DROPPED_TOTAL.labels(
            reason=PositionTrackDropReason.INVALID_COVARIANCE
        )._value.get()
        == before_cov + 1
    )


@pytest.mark.asyncio
async def test_runtime_publishes_anonymous_position_tracks_with_1h_minid_and_no_wearable_caps() -> (
    None
):
    now = _T0 + timedelta(seconds=10)
    provider = InMemorySiteConfigProvider(_site())
    stream = InMemoryMqttMessageStream(
        [
            (
                "sensors/position/radar-bath-1/tracks",
                json.dumps(
                    {
                        "schema": "position_track.v1",
                        "sensor_id": "radar-bath-1",
                        "sent_at_unix_ns": datetime_to_unix_ns(now),
                        "tracks": [
                            {
                                "sensor_track_id": "t-01",
                                "measured_at_unix_ns": datetime_to_unix_ns(_T0),
                                "x_m": 2.45,
                                "y_m": 4.10,
                                "cov_xx_m2": 0.0225,
                                "cov_xy_m2": 0.0,
                                "cov_yy_m2": 0.0225,
                                "speed_m_s": 0.35,
                            }
                        ],
                    }
                ).encode("utf-8"),
            )
        ]
    )
    adapter = PositionTrackMqttAdapter(
        adapter_id="mqtt-pos",
        deps=AdapterDeps(site_config=provider, clock=lambda: now),
        stream=stream,
    )
    tag_pub = InMemoryTagFixPublisher(clock=lambda: now)
    pos_pub = InMemoryPositionTrackPublisher(
        retention_hours=DEFAULT_POSITION_TRACK_RETENTION_HOURS,
        clock=lambda: now,
    )
    before_pub = POSITION_TRACKS_PUBLISHED_TOTAL.labels(adapter_id="mqtt-pos")._value.get()

    runtime = RfIngressRuntime(
        adapters=[adapter],
        site_config=provider,
        publisher=tag_pub,
        position_track_publisher=pos_pub,
        clock=lambda: now,
    )
    fixes = await runtime.run_once()
    assert fixes == []
    assert tag_pub.fixes == []
    assert tag_pub.capabilities == []
    assert len(pos_pub.tracks) == 1

    pub = pos_pub.tracks[0]
    assert pub.stream == "position.tracks"
    assert pub.minid == compute_minid_ms(now, 1.0)
    assert pub.proto.sensor_id == "radar-bath-1"
    assert pub.proto.sensor_track_id == "t-01"
    assert pub.proto.x_mm == 2450
    assert pub.proto.y_mm == 4100
    assert pub.proto.has_speed is True
    assert pub.proto.speed_m_s == pytest.approx(0.35)
    assert pub.proto.floor_plan_id == "fp-main"
    assert runtime.health_snapshot().publisher_healthy is True
    assert (
        POSITION_TRACKS_PUBLISHED_TOTAL.labels(adapter_id="mqtt-pos")._value.get() == before_pub + 1
    )

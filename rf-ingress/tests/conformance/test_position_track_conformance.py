"""Conformance suite for anonymous PositionTrackAdapter implementations."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import pytest
from cts_contracts.position_sensors import is_valid_sensor_id

from rf_ingress.adapters._mqtt_transport import InMemoryMqttMessageStream
from rf_ingress.adapters.base import AdapterDeps, PositionTrackAdapter
from rf_ingress.adapters.position_track_mqtt import PositionTrackMqttAdapter
from rf_ingress.adapters.simulator import PositionTrackSimulatorAdapter
from rf_ingress.domain import (
    AnonymousPositionTrackReport,
    PositionSensorConfig,
    SiteConfigSnapshot,
)
from rf_ingress.observability import POSITION_TRACKS_DROPPED_TOTAL, PositionTrackDropReason
from rf_ingress.publish import datetime_to_unix_ns
from rf_ingress.site_config import InMemorySiteConfigProvider

_MONOTONIC_TOLERANCE_S = 1.0


@dataclass(frozen=True)
class PositionTrackConformanceContext:
    adapter: PositionTrackAdapter
    expected_registered_sensors: frozenset[str]
    expected_unregistered_sensors: frozenset[str] = frozenset()
    expects_malformed_drop: bool = False


def _site() -> SiteConfigSnapshot:
    return SiteConfigSnapshot(
        floor_plan_id="fp-main",
        position_sensors=(
            PositionSensorConfig(
                sensor_id="radar-bath-1",
                room_id="bathroom",
                x_m=2.0,
                y_m=3.0,
                yaw_deg=90.0,
                calibration_residual_m=0.10,
            ),
        ),
        version="v1",
    )


def _build_simulator_case() -> PositionTrackConformanceContext:
    provider = InMemorySiteConfigProvider(_site())
    adapter = PositionTrackSimulatorAdapter(
        adapter_id="pos-sim-1",
        deps=AdapterDeps(site_config=provider),
        seed=7,
        steps=4,
        frame="sensor_local",
        inject_unregistered=True,
        inject_malformed=True,
    )
    return PositionTrackConformanceContext(
        adapter=adapter,
        expected_registered_sensors=frozenset({"radar-bath-1"}),
        expected_unregistered_sensors=frozenset({"radar-unknown"}),
        expects_malformed_drop=True,
    )


def _build_mqtt_case() -> PositionTrackConformanceContext:
    provider = InMemorySiteConfigProvider(_site())
    t0 = datetime.now().astimezone()
    ns1 = datetime_to_unix_ns(t0)
    ns2 = ns1 + 500_000_000
    stream = InMemoryMqttMessageStream(
        [
            ("sensors/position/radar-bath-1/tracks", b"{bad-json"),
            (
                "sensors/position/radar-unknown/tracks",
                json.dumps(
                    {
                        "schema": "position_track.v1",
                        "sensor_id": "radar-unknown",
                        "tracks": [
                            {
                                "sensor_track_id": "u1",
                                "measured_at_unix_ns": ns1,
                                "x_m": 1.0,
                                "y_m": 1.0,
                                "cov_xx_m2": 0.04,
                                "cov_xy_m2": 0.0,
                                "cov_yy_m2": 0.04,
                            }
                        ],
                    }
                ).encode("utf-8"),
            ),
            (
                "sensors/position/radar-bath-1/tracks",
                json.dumps(
                    {
                        "schema": "position_track.v1",
                        "sensor_id": "radar-bath-1",
                        "tracks": [
                            {
                                "sensor_track_id": "t1",
                                "measured_at_unix_ns": ns1,
                                "x_m": 2.1,
                                "y_m": 3.1,
                                "cov_xx_m2": 0.04,
                                "cov_xy_m2": 0.0,
                                "cov_yy_m2": 0.04,
                            },
                            {
                                "sensor_track_id": "t1",
                                "measured_at_unix_ns": ns2,
                                "x_m": 2.2,
                                "y_m": 3.2,
                                "cov_xx_m2": 0.04,
                                "cov_xy_m2": 0.0,
                                "cov_yy_m2": 0.04,
                            },
                        ],
                    }
                ).encode("utf-8"),
            ),
        ]
    )
    adapter = PositionTrackMqttAdapter(
        adapter_id="pos-mqtt-1",
        deps=AdapterDeps(site_config=provider),
        stream=stream,
    )
    return PositionTrackConformanceContext(
        adapter=adapter,
        expected_registered_sensors=frozenset({"radar-bath-1"}),
        expected_unregistered_sensors=frozenset({"radar-unknown"}),
        expects_malformed_drop=True,
    )


_CASES: list[tuple[str, Callable[[], PositionTrackConformanceContext]]] = [
    ("position_track_simulator", _build_simulator_case),
    ("position_track_mqtt", _build_mqtt_case),
]


@pytest.fixture(params=[c[0] for c in _CASES])
def pos_conformance_ctx(request: pytest.FixtureRequest) -> PositionTrackConformanceContext:
    factory = dict(_CASES)[request.param]
    return factory()


@pytest.mark.asyncio
async def test_position_conformance_registered_sensors_and_unregistered_count(
    pos_conformance_ctx: PositionTrackConformanceContext,
) -> None:
    before_unreg = POSITION_TRACKS_DROPPED_TOTAL.labels(
        reason=PositionTrackDropReason.UNREGISTERED_SENSOR
    )._value.get()
    reports: list[AnonymousPositionTrackReport] = [
        r async for r in pos_conformance_ctx.adapter.position_tracks()
    ]
    assert len(reports) > 0
    seen_sensors = {r.sensor_id for r in reports}
    assert seen_sensors.issubset(pos_conformance_ctx.expected_registered_sensors)
    assert seen_sensors.isdisjoint(pos_conformance_ctx.expected_unregistered_sensors)
    assert all(is_valid_sensor_id(r.sensor_id) for r in reports)
    if pos_conformance_ctx.expected_unregistered_sensors:
        after_unreg = POSITION_TRACKS_DROPPED_TOTAL.labels(
            reason=PositionTrackDropReason.UNREGISTERED_SENSOR
        )._value.get()
        assert after_unreg > before_unreg
        assert pos_conformance_ctx.adapter.health().unregistered_reports > 0


@pytest.mark.asyncio
async def test_position_conformance_timestamps_and_positive_definite_cov(
    pos_conformance_ctx: PositionTrackConformanceContext,
) -> None:
    last_by_track: dict[tuple[str, str], datetime] = {}
    async for report in pos_conformance_ctx.adapter.position_tracks():
        assert report.measured_at.tzinfo is not None
        assert report.measured_at.utcoffset() is not None
        assert math.isfinite(report.x_m) and math.isfinite(report.y_m)
        det = report.cov_xx_m2 * report.cov_yy_m2 - report.cov_xy_m2 * report.cov_xy_m2
        assert report.cov_xx_m2 > 0.0 and report.cov_yy_m2 > 0.0 and det > 1e-9
        key = (report.sensor_id, report.sensor_track_id)
        prev = last_by_track.get(key)
        if prev is not None:
            assert (report.measured_at - prev).total_seconds() >= -_MONOTONIC_TOLERANCE_S
        last_by_track[key] = report.measured_at


@pytest.mark.asyncio
async def test_position_conformance_malformed_counted_and_clean_close(
    pos_conformance_ctx: PositionTrackConformanceContext,
) -> None:
    before_mal = POSITION_TRACKS_DROPPED_TOTAL.labels(
        reason=PositionTrackDropReason.MALFORMED_INPUT
    )._value.get()
    reports = [r async for r in pos_conformance_ctx.adapter.position_tracks()]
    assert len(reports) > 0
    if pos_conformance_ctx.expects_malformed_drop:
        after_mal = POSITION_TRACKS_DROPPED_TOTAL.labels(
            reason=PositionTrackDropReason.MALFORMED_INPUT
        )._value.get()
        assert after_mal > before_mal
        assert pos_conformance_ctx.adapter.health().malformed_reports > 0

    adapter = pos_conformance_ctx.adapter
    assert adapter.health().connected is True
    await adapter.close()
    assert adapter.health().connected is False
    after_close = [r async for r in adapter.position_tracks()]
    assert after_close == []

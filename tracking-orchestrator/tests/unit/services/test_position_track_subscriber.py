"""Tests for PositionTrackSubscriber decoding, validation, dead-lettering, and buffering."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from prometheus_client import CollectorRegistry

from app.domain import PositionSensor, WearableSiteConfig
from app.observability import metrics as metrics_module
from app.observability.metrics import DeadLetterStream, PositionTrackIngestResult
from app.proto.continuoustracking.v1 import position_sensor_pb2
from app.services.position_track_buffer import PositionTrackBuffer
from app.services.position_track_subscriber import (
    FIELD_POSITION_TRACK,
    PositionTrackSubscriber,
    decode_position_track,
)
from app.services.wearable_site_config_map import WearableSiteConfigMap

_T0 = datetime(2026, 11, 25, 10, 0, 0, tzinfo=UTC)


def _site_config() -> WearableSiteConfig:
    return WearableSiteConfig(
        floor_plan_id="fp-main",
        position_sensors=(
            PositionSensor(
                sensor_id="radar-bath-1",
                room_id="bathroom",
                x_mm=2000,
                y_mm=3000,
                yaw_deg=0.0,
                calibration_residual_m=0.12,
            ),
        ),
        version="v1",
    )


def _encode_track(
    *,
    sensor_id: str = "radar-bath-1",
    sensor_track_id: str = "t-01",
    measured_at: datetime = _T0,
    x_mm: int = 2450,
    y_mm: int = 4100,
    cov_xx: float = 0.04,
    cov_xy: float = 0.0,
    cov_yy: float = 0.04,
    has_speed: bool = False,
    speed_m_s: float = 0.0,
    floor_plan_id: str = "fp-main",
) -> bytes:
    msg = position_sensor_pb2.AnonymousPositionTrack(
        sensor_id=sensor_id,
        sensor_track_id=sensor_track_id,
        measured_at_unix_ns=int(measured_at.timestamp() * 1e9),
        x_mm=x_mm,
        y_mm=y_mm,
        cov_xx_m2=cov_xx,
        cov_xy_m2=cov_xy,
        cov_yy_m2=cov_yy,
        has_speed=has_speed,
        speed_m_s=speed_m_s,
        floor_plan_id=floor_plan_id,
    )
    return msg.SerializeToString()


def test_decodes_valid_anonymous_position_track_and_buffers_in_time_order() -> None:
    cfg = _site_config()
    buf = PositionTrackBuffer(max_age_s=10.0, track_gap_s=2.0, max_tracks=16)
    sub = PositionTrackSubscriber(
        redis_client=None,
        buffer=buf,
        site_config_map=WearableSiteConfigMap(cfg),
        max_track_age_s=5.0,
    )

    later = _encode_track(measured_at=_T0 + timedelta(seconds=1.5), x_mm=2600, y_mm=4300)
    earlier = _encode_track(
        measured_at=_T0,
        x_mm=2450,
        y_mm=4100,
        has_speed=True,
        speed_m_s=0.45,
    )

    assert sub.handle_fields({FIELD_POSITION_TRACK: later}, now=_T0 + timedelta(seconds=2)) == (
        PositionTrackIngestResult.ACCEPTED
    )
    assert sub.handle_fields({FIELD_POSITION_TRACK: earlier}, now=_T0 + timedelta(seconds=2)) == (
        PositionTrackIngestResult.ACCEPTED
    )

    taken = buf.take_round_tracks(
        now=_T0 + timedelta(seconds=2),
        max_age_s=5.0,
        min_track_age_s=1.0,
    )
    assert len(taken) == 1
    assert taken[0].x_m == pytest.approx(2.6)
    assert taken[0].y_m == pytest.approx(4.3)
    assert taken[0].calibration_residual_m == pytest.approx(0.12)

    obs, result = decode_position_track(
        earlier, site_config=cfg, now=_T0 + timedelta(seconds=1), max_track_age_s=5.0
    )
    assert result == PositionTrackIngestResult.ACCEPTED
    assert obs is not None
    assert obs.x_m == pytest.approx(2.45)
    assert obs.y_m == pytest.approx(4.10)
    assert obs.speed_m_s == pytest.approx(0.45)


def test_rejects_non_positive_definite_covariance_and_dead_letters_malformed_proto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = CollectorRegistry()
    test_metrics = metrics_module.build_metrics(registry)
    monkeypatch.setattr(metrics_module, "metrics", test_metrics)

    buf = PositionTrackBuffer(max_age_s=10.0, track_gap_s=2.0, max_tracks=16)
    sub = PositionTrackSubscriber(
        redis_client=None,
        buffer=buf,
        site_config_map=WearableSiteConfigMap(_site_config()),
        max_track_age_s=5.0,
    )

    singular = _encode_track(cov_xx=0.04, cov_xy=0.04, cov_yy=0.04)
    negative_var = _encode_track(cov_xx=-0.04, cov_xy=0.0, cov_yy=0.04)
    nan_cov = _encode_track(cov_xx=float("nan"))

    for payload in (singular, negative_var, nan_cov, b"\xff\xff\xff\xff"):
        res = sub.handle_fields({FIELD_POSITION_TRACK: payload}, now=_T0)
        assert res == PositionTrackIngestResult.DROPPED_MALFORMED

    assert sub.handle_fields({b"other": b"value"}, now=_T0) == (
        PositionTrackIngestResult.DROPPED_MALFORMED
    )
    assert (
        sub.handle_fields(
            {FIELD_POSITION_TRACK: _encode_track(sensor_id="unknown-radar")},
            now=_T0,
        )
        == PositionTrackIngestResult.DROPPED_UNREGISTERED
    )
    assert (
        sub.handle_fields(
            {FIELD_POSITION_TRACK: _encode_track(floor_plan_id="other-plan")},
            now=_T0,
        )
        == PositionTrackIngestResult.DROPPED_FLOOR_PLAN
    )
    assert (
        sub.handle_fields(
            {FIELD_POSITION_TRACK: _encode_track(measured_at=_T0 - timedelta(seconds=10))},
            now=_T0,
        )
        == PositionTrackIngestResult.DROPPED_STALE
    )

    assert len(buf) == 0
    assert test_metrics.cts_messages_dead_lettered_total.labels(
        stream=str(DeadLetterStream.POSITION_TRACKS)
    )._value.get() == pytest.approx(5.0)
    assert test_metrics.cts_position_tracks_total.labels(
        result=str(PositionTrackIngestResult.DROPPED_UNREGISTERED)
    )._value.get() == pytest.approx(1.0)
    assert test_metrics.cts_position_tracks_total.labels(
        result=str(PositionTrackIngestResult.DROPPED_FLOOR_PLAN)
    )._value.get() == pytest.approx(1.0)
    assert test_metrics.cts_position_tracks_total.labels(
        result=str(PositionTrackIngestResult.DROPPED_STALE)
    )._value.get() == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_subscriber_loop_survives_malformed_entry_and_xacks() -> None:
    class _FakeRedis:
        def __init__(self) -> None:
            self.acked: list[tuple[str, str, Any]] = []
            self._reads = 0

        async def xgroup_create(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        async def xreadgroup(self, **_kwargs: Any) -> list[Any]:
            self._reads += 1
            if self._reads == 1:
                return [
                    (
                        b"position.tracks",
                        [
                            (b"1-0", {FIELD_POSITION_TRACK: b"not-a-proto"}),
                            (
                                b"2-0",
                                {
                                    FIELD_POSITION_TRACK: _encode_track(
                                        measured_at=datetime.now(UTC)
                                    )
                                },
                            ),
                        ],
                    )
                ]
            raise asyncio.CancelledError

        async def xack(self, stream: str, group: str, msg_id: Any) -> None:
            self.acked.append((stream, group, msg_id))

    fake_redis = _FakeRedis()
    buf = PositionTrackBuffer(max_age_s=10.0, track_gap_s=2.0, max_tracks=16)
    sub = PositionTrackSubscriber(
        redis_client=fake_redis,
        buffer=buf,
        site_config_map=WearableSiteConfigMap(_site_config()),
        max_track_age_s=5.0,
    )
    sub._running = True
    with pytest.raises(asyncio.CancelledError):
        await sub._consume_loop()

    assert [item[2] for item in fake_redis.acked] == [b"1-0", b"2-0"]
    assert len(buf) == 1

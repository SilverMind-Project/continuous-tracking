"""Unit tests for BurstClockOffsetTracker and AdapterClockSkewEstimator."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from cts_contracts.wearables import (
    AdapterKind,
    DeviceKind,
    FixKind,
    MotionState,
    RangingTechnology,
    WearState,
)

from rf_ingress.adapters.base import AdapterDeps, AdapterHealth
from rf_ingress.clock import AdapterClockSkewEstimator, BurstClockOffsetTracker
from rf_ingress.domain import (
    AdapterCapabilityDescriptor,
    RangeMeasurement,
    RfNodeGeometry,
    SiteBinding,
    SiteConfigSnapshot,
    SiteDevice,
)
from rf_ingress.observability import DEVICE_CLOCK_OFFSET_MS
from rf_ingress.pipeline import RfIngressRuntime
from rf_ingress.publish import InMemoryTagFixPublisher, datetime_to_unix_ns
from rf_ingress.site_config import InMemorySiteConfigProvider

_T0 = datetime(2026, 11, 19, 12, 0, 0, tzinfo=UTC)


class _StubRangeAdapter:
    def __init__(self, adapter_id: str) -> None:
        self.adapter_id = adapter_id
        self.capability = AdapterCapabilityDescriptor(
            adapter_id=adapter_id,
            adapter_kind=AdapterKind.DEVICE_RANGING,
            adapter_version="0.1.0",
            fix_kinds=(FixKind.POSITION,),
            nominal_rate_hz=1.0,
            reports_covariance=True,
            reports_wear_state=True,
            reports_z=True,
            reports_motion_state=True,
            ntp_synced=True,
            latency_s=0.0,
        )

    def health(self) -> AdapterHealth:
        return AdapterHealth(adapter_id=self.adapter_id, connected=True, last_report_at=_T0)

    async def close(self) -> None:
        return None


async def test_12s_burst_in_four_3s_batches_reconstructs_within_60ms_of_ground_truth() -> None:
    """Four 3 s batches with +150 ms watch skew and 50-800 ms Wi-Fi jitter stay within 60 ms."""
    anchors = (
        RfNodeGeometry(anchor_id="a1", x_m=0.0, y_m=0.0, z_m=2.4, adapter_id="watch-1"),
        RfNodeGeometry(anchor_id="a2", x_m=6.0, y_m=0.0, z_m=1.4, adapter_id="watch-1"),
        RfNodeGeometry(anchor_id="a3", x_m=6.0, y_m=5.0, z_m=2.5, adapter_id="watch-1"),
        RfNodeGeometry(anchor_id="a4", x_m=0.0, y_m=5.0, z_m=1.5, adapter_id="watch-1"),
    )
    snap = SiteConfigSnapshot(
        floor_plan_id="household:1",
        nodes=anchors,
        devices=(
            SiteDevice(
                tag_id="tag-watch-1",
                adapter_id="watch-1",
                hardware_key="watch:1",
                device_kind=DeviceKind.WATCH,
                is_active=True,
            ),
        ),
        bindings=(
            SiteBinding(
                tag_id="tag-watch-1",
                member_id="caregiver-1",
                valid_from=_T0 - timedelta(hours=1),
                valid_to=None,
            ),
        ),
        version="burst:v1",
    )
    provider = InMemorySiteConfigProvider(snap)
    publisher = InMemoryTagFixPublisher()
    adapter = _StubRangeAdapter("watch-1")
    _ = AdapterDeps(site_config=provider)

    runtime = RfIngressRuntime(
        adapters=[],
        site_config=provider,
        publisher=publisher,
    )

    watch_skew_s = 0.150  # Watch clock is +150 ms ahead of UTC.
    batch_jitters_s = [0.050, 0.420, 0.800, 0.180]  # 50 ms to 800 ms Wi-Fi queueing jitter.
    true_tag_xyz = (3.0, 2.5, 1.0)
    true_epoch_times: dict[int, datetime] = {}

    for batch_idx, jitter_s in enumerate(batch_jitters_s):
        batch_start_s = batch_idx * 3
        batch_end_s = batch_start_s + 2
        true_send = _T0 + timedelta(seconds=batch_end_s)
        sent_at_watch = true_send + timedelta(seconds=watch_skew_s)
        broker_arrival = true_send + timedelta(seconds=jitter_s)

        for offset_in_batch in range(3):
            epoch_seq = batch_start_s + offset_in_batch + 1
            true_capture = _T0 + timedelta(seconds=batch_start_s + offset_in_batch)
            true_epoch_times[epoch_seq] = true_capture
            measured_at_watch = true_capture + timedelta(seconds=watch_skew_s)

            for node in anchors:
                dist = (
                    (true_tag_xyz[0] - node.x_m) ** 2
                    + (true_tag_xyz[1] - node.y_m) ** 2
                    + (true_tag_xyz[2] - node.z_m) ** 2
                ) ** 0.5
                meas = RangeMeasurement(
                    tag_id="tag-watch-1",
                    anchor_id=node.anchor_id,
                    distance_m=dist,
                    variance_m2=0.12**2,
                    technology=RangingTechnology.UWB,
                    measured_at=measured_at_watch,
                    wear_state=WearState.ON_BODY,
                    motion_state=MotionState.MOVING,
                    adapter_id="watch-1",
                    epoch_seq=epoch_seq,
                    sent_at=sent_at_watch,
                )
                await runtime.handle_range_measurement(
                    adapter,
                    meas,
                    arrival_at=broker_arrival,  # type: ignore[arg-type]
                )

    assert len(publisher.fixes) == 12
    for pub in publisher.fixes:
        expected_ns = datetime_to_unix_ns(true_epoch_times[pub.proto.epoch_seq])
        error_ms = abs(pub.proto.measured_at_unix_ns - expected_ns) / 1_000_000.0
        assert error_ms <= 60.0, (
            f"Epoch {pub.proto.epoch_seq} timestamp error {error_ms:.1f} ms > 60 ms"
        )

    # Metric exported in milliseconds: min(jitter) - skew = 50 - 150 = -100 ms.
    offset_ms = DEVICE_CLOCK_OFFSET_MS.labels(tag_id="tag-watch-1")._value.get()
    assert abs(offset_ms - (-100.0)) < 1.0


def test_burst_tracker_window_and_adapter_skew_median() -> None:
    tracker = BurstClockOffsetTracker(window_size=3)
    tracker.observe_batch("t1", sent_at=_T0, arrival_at=_T0 + timedelta(milliseconds=300))
    tracker.observe_batch(
        "t1",
        sent_at=_T0 + timedelta(seconds=3),
        arrival_at=_T0 + timedelta(seconds=3, milliseconds=80),
    )
    assert tracker.offset_seconds("t1") == 0.08

    skew = AdapterClockSkewEstimator(window_s=60.0, warn_threshold_s=0.5)
    skew.observe("ad-1", device_time=_T0, arrival_time=_T0 + timedelta(seconds=0.2))
    skew.observe(
        "ad-1", device_time=_T0 + timedelta(seconds=1), arrival_time=_T0 + timedelta(seconds=1.8)
    )
    skew.observe(
        "ad-1", device_time=_T0 + timedelta(seconds=2), arrival_time=_T0 + timedelta(seconds=2.4)
    )
    current = skew.current_skew_seconds("ad-1")
    assert current is not None
    assert abs(current - 0.4) < 1e-6

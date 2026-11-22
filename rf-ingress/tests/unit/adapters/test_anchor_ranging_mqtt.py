"""Unit tests for the anchor_ranging_mqtt adapter (fira_anchor_v1)."""

from __future__ import annotations

import json
import math
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from cts_contracts.wearables import (
    AdapterKind,
    DeviceKind,
    FixKind,
    MotionState,
    RangingTechnology,
    RfNodeRole,
    WearState,
)
from pydantic import ValidationError

from rf_ingress.adapters._mqtt_transport import InMemoryMqttMessageStream
from rf_ingress.adapters.anchor_ranging_mqtt import (
    AnchorNodeCalibrationConfig,
    AnchorRangingMqttAdapter,
    AnchorRangingMqttConfig,
    FiraAnchorReportV1,
)
from rf_ingress.adapters.base import AdapterDeps
from rf_ingress.adapters.registry import build_default_registry
from rf_ingress.bench import BenchGroundTruthPoint, replay_fira_anchor_bench_session
from rf_ingress.config import AdapterConfig
from rf_ingress.domain import (
    RfNodeGeometry,
    SiteBinding,
    SiteConfigSnapshot,
    SiteDevice,
)
from rf_ingress.observability import FIXES_DROPPED_TOTAL, UNREGISTERED_TAG_TOTAL, DropReason
from rf_ingress.pipeline import RfIngressRuntime
from rf_ingress.publish import InMemoryTagFixPublisher, datetime_to_unix_ns
from rf_ingress.site_config import InMemorySiteConfigProvider

_T0 = datetime(2026, 11, 22, 12, 0, 0, tzinfo=UTC)
_TAG_HW_KEY = "uwb:AA:BB:CC:01"
_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "schemas" / "fira_anchor_v1.fixture.json"
)


def _uwb_site_config() -> SiteConfigSnapshot:
    return SiteConfigSnapshot(
        floor_plan_id="household:1",
        nodes=(
            RfNodeGeometry(
                anchor_id="living-room-uwb-1",
                x_m=0.2,
                y_m=0.2,
                z_m=2.3,
                adapter_id="uwb-home",
                node_role=RfNodeRole.RANGING_ANCHOR,
                technology=RangingTechnology.UWB,
                room_id="room-living",
                uwb_address_hex="00:01",
                uwb_channel=9,
                uwb_preamble_index=10,
            ),
            RfNodeGeometry(
                anchor_id="living-room-uwb-2",
                x_m=5.8,
                y_m=0.2,
                z_m=1.3,
                adapter_id="uwb-home",
                node_role=RfNodeRole.RANGING_ANCHOR,
                technology=RangingTechnology.UWB,
                room_id="room-living",
                uwb_address_hex="00:02",
                uwb_channel=9,
                uwb_preamble_index=10,
            ),
            RfNodeGeometry(
                anchor_id="living-room-uwb-3",
                x_m=5.8,
                y_m=4.8,
                z_m=2.4,
                adapter_id="uwb-home",
                node_role=RfNodeRole.RANGING_ANCHOR,
                technology=RangingTechnology.UWB,
                room_id="room-living",
                uwb_address_hex="00:03",
                uwb_channel=9,
                uwb_preamble_index=10,
            ),
            RfNodeGeometry(
                anchor_id="living-room-uwb-4",
                x_m=0.2,
                y_m=4.8,
                z_m=1.2,
                adapter_id="uwb-home",
                node_role=RfNodeRole.RANGING_ANCHOR,
                technology=RangingTechnology.UWB,
                room_id="room-living",
                uwb_address_hex="00:04",
                uwb_channel=9,
                uwb_preamble_index=10,
            ),
        ),
        devices=(
            SiteDevice(
                tag_id="senior-uwb-clip-1",
                adapter_id="uwb-home",
                hardware_key=_TAG_HW_KEY,
                device_kind=DeviceKind.UWB_TAG,
                is_active=True,
            ),
        ),
        bindings=(
            SiteBinding(
                tag_id="senior-uwb-clip-1",
                member_id="resident-1",
                valid_from=_T0 - timedelta(hours=1),
                valid_to=None,
            ),
        ),
        version="uwb:v1",
    )


def _report_msg(
    *,
    anchor_id: str = "living-room-uwb-1",
    uwb_address_hex: str | None = None,
    tag_id: str = "",
    hardware_key: str | None = _TAG_HW_KEY,
    epoch_seq: int | None = 1,
    measured_at: datetime = _T0,
    sent_at: datetime | None = None,
    distance_m: float = 2.4,
    variance_m2: float | None = None,
    wear_state: str = "on_body",
    motion_state: str = "moving",
    charging: bool = False,
    battery_fraction: float | None = 0.85,
    azimuth_deg: float | None = None,
    azimuth_variance_deg2: float | None = None,
) -> tuple[str, bytes]:
    body: dict[str, object] = {
        "schema": "fira_anchor.v1",
        "measured_at_unix_ns": datetime_to_unix_ns(measured_at),
        "distance_m": distance_m,
        "wear_state": wear_state,
        "motion_state": motion_state,
        "charging": charging,
    }
    if anchor_id:
        body["anchor_id"] = anchor_id
    if uwb_address_hex is not None:
        body["uwb_address_hex"] = uwb_address_hex
    if tag_id:
        body["tag_id"] = tag_id
    if hardware_key is not None:
        body["hardware_key"] = hardware_key
    if epoch_seq is not None:
        body["epoch_seq"] = epoch_seq
    if sent_at is not None:
        body["sent_at_unix_ns"] = datetime_to_unix_ns(sent_at)
    if variance_m2 is not None:
        body["variance_m2"] = variance_m2
    if battery_fraction is not None:
        body["battery_fraction"] = battery_fraction
    if azimuth_deg is not None:
        body["azimuth_deg"] = azimuth_deg
    if azimuth_variance_deg2 is not None:
        body["azimuth_variance_deg2"] = azimuth_variance_deg2
    topic_anchor = anchor_id or (uwb_address_hex or "unknown")
    return (f"rf/anchors/{topic_anchor}/ranges", json.dumps(body).encode("utf-8"))


def test_shared_schema_fixture_validates_all_valid_and_rejects_all_invalid() -> None:
    fixture = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    assert fixture["schema"] == "fira_anchor.v1"
    assert len(fixture["valid_samples"]) >= 4
    assert len(fixture["invalid_samples"]) >= 5

    for sample in fixture["valid_samples"]:
        parsed = FiraAnchorReportV1.model_validate(sample["payload"])
        assert parsed.schema_version == "fira_anchor.v1"

    for sample in fixture["invalid_samples"]:
        with pytest.raises(ValidationError):
            FiraAnchorReportV1.model_validate(sample["payload"])


def test_config_and_registry_validation() -> None:
    with pytest.raises(ValidationError, match="Unknown anchor_ranging_mqtt payload_profile"):
        AnchorRangingMqttConfig(payload_profile="unknown_v0")

    with pytest.raises(ValidationError, match="uwb_default_sigma_m must be >= sigma_floor_m"):
        AnchorRangingMqttConfig(uwb_default_sigma_m=0.05, sigma_floor_m=0.10)

    registry = build_default_registry()
    assert "anchor_ranging_mqtt" in registry.registered_kinds()
    deps = AdapterDeps(site_config=InMemorySiteConfigProvider(_uwb_site_config()))
    cfg = AdapterConfig(
        id="uwb-home",
        kind="anchor_ranging_mqtt",
        payload_profile="fira_anchor_v1",
        broker_url="mqtt://127.0.0.1:1883",
        topic="rf/anchors/+/ranges",
        uwb_default_sigma_m=0.22,
        sigma_floor_m=0.10,
        nodes={"living-room-uwb-1": {"sigma_m": 0.18, "range_bias_m": 0.03}},
    )
    adapter = registry.build(cfg, deps)
    assert isinstance(adapter, AnchorRangingMqttAdapter)
    assert adapter.capability.adapter_kind == AdapterKind.ANCHOR_RANGING
    assert adapter.capability.reports_wear_state is True
    assert adapter.capability.reports_motion_state is True


async def test_resolves_hardware_key_uwb_address_and_applies_variance_floor_and_bias() -> None:
    provider = InMemorySiteConfigProvider(_uwb_site_config())
    stream = InMemoryMqttMessageStream(
        [
            _report_msg(
                anchor_id="",
                uwb_address_hex="00:02",
                hardware_key=_TAG_HW_KEY,
                distance_m=3.15,
                variance_m2=0.0001,  # overconfident datasheet value must be floored
                epoch_seq=10,
                sent_at=_T0 + timedelta(milliseconds=30),
            ),
            _report_msg(
                anchor_id="living-room-uwb-1",
                hardware_key=_TAG_HW_KEY,
                distance_m=2.55,
                variance_m2=None,
                epoch_seq=10,
            ),
        ]
    )
    adapter = AnchorRangingMqttAdapter(
        adapter_id="uwb-home",
        deps=AdapterDeps(site_config=provider, clock=lambda: _T0),
        stream=stream,
        uwb_default_sigma_m=0.25,
        sigma_floor_m=0.12,
        node_calibrations={
            "living-room-uwb-1": AnchorNodeCalibrationConfig(sigma_m=0.18, range_bias_m=0.05),
        },
    )
    emitted = [m async for m in adapter.ranges()]
    assert len(emitted) == 2

    first, second = emitted
    assert first.tag_id == "senior-uwb-clip-1"
    assert first.anchor_id == "living-room-uwb-2"
    assert first.technology == RangingTechnology.UWB
    assert first.variance_m2 == pytest.approx(0.12 * 0.12)
    assert first.sent_at == _T0 + timedelta(milliseconds=30)

    assert second.anchor_id == "living-room-uwb-1"
    assert second.distance_m == pytest.approx(2.50)
    assert second.variance_m2 == pytest.approx(0.18 * 0.18)


async def test_charging_tag_always_emits_off_body_through_runtime() -> None:
    snap = _uwb_site_config()
    provider = InMemorySiteConfigProvider(snap)
    # 3 anchors report a tag sitting on its charger; raw payload erroneously says on_body.
    stream = InMemoryMqttMessageStream(
        [
            _report_msg(
                anchor_id="living-room-uwb-1",
                distance_m=math.hypot(2.0 - 0.2, 2.0 - 0.2, 1.0 - 2.3),
                epoch_seq=7,
                wear_state="on_body",
                motion_state="still",
                charging=True,
                battery_fraction=0.98,
            ),
            _report_msg(
                anchor_id="living-room-uwb-2",
                distance_m=math.hypot(2.0 - 5.8, 2.0 - 0.2, 1.0 - 1.3),
                epoch_seq=7,
                wear_state="on_body",
                motion_state="still",
                charging=True,
                battery_fraction=0.98,
            ),
            _report_msg(
                anchor_id="living-room-uwb-3",
                distance_m=math.hypot(2.0 - 5.8, 2.0 - 4.8, 1.0 - 2.4),
                epoch_seq=7,
                wear_state="on_body",
                motion_state="still",
                charging=True,
                battery_fraction=0.98,
            ),
        ]
    )
    adapter = AnchorRangingMqttAdapter(
        adapter_id="uwb-home",
        deps=AdapterDeps(site_config=provider, clock=lambda: _T0),
        stream=stream,
    )
    publisher = InMemoryTagFixPublisher()
    runtime = RfIngressRuntime(
        adapters=[adapter],
        site_config=provider,
        publisher=publisher,
        clock=lambda: _T0,
    )
    await runtime.run_once()
    assert len(publisher.fixes) == 1
    fix = publisher.fixes[0].fix
    assert fix.kind == FixKind.POSITION
    assert fix.wear_state == WearState.OFF_BODY
    assert fix.motion_state == MotionState.STILL
    assert fix.battery_fraction == pytest.approx(0.98)


async def test_drops_malformed_unregistered_unresolved_and_out_of_bounds() -> None:
    provider = InMemorySiteConfigProvider(_uwb_site_config())
    before_malformed = FIXES_DROPPED_TOTAL.labels(reason=DropReason.MALFORMED_INPUT)._value.get()
    before_unreg = UNREGISTERED_TAG_TOTAL.labels(adapter_id="uwb-home")._value.get()
    before_oob = FIXES_DROPPED_TOTAL.labels(reason=DropReason.RANGE_OUT_OF_BOUNDS)._value.get()

    stream = InMemoryMqttMessageStream(
        [
            ("rf/anchors/a/ranges", b"not-json"),
            _report_msg(anchor_id="", uwb_address_hex="99:99", epoch_seq=1),
            _report_msg(hardware_key="uwb:00:00:00:99", epoch_seq=2),
            _report_msg(distance_m=42.0, epoch_seq=3),
            _report_msg(distance_m=2.2, epoch_seq=4),
        ]
    )
    adapter = AnchorRangingMqttAdapter(
        adapter_id="uwb-home",
        deps=AdapterDeps(site_config=provider, clock=lambda: _T0),
        stream=stream,
        max_range_m=20.0,
    )
    emitted = [m async for m in adapter.ranges()]
    assert len(emitted) == 1
    assert emitted[0].epoch_seq == 4
    assert (
        FIXES_DROPPED_TOTAL.labels(reason=DropReason.MALFORMED_INPUT)._value.get()
        == before_malformed + 2
    )
    assert UNREGISTERED_TAG_TOTAL.labels(adapter_id="uwb-home")._value.get() == before_unreg + 1
    assert FIXES_DROPPED_TOTAL.labels(reason=DropReason.RANGE_OUT_OF_BOUNDS)._value.get() == (
        before_oob + 1
    )


async def test_azimuth_frame_stays_device_local_unless_anchor_explicitly_calibrated() -> None:
    provider = InMemorySiteConfigProvider(_uwb_site_config())
    stream = InMemoryMqttMessageStream(
        [
            _report_msg(
                anchor_id="living-room-uwb-1",
                azimuth_deg=35.0,
                azimuth_variance_deg2=16.0,
                epoch_seq=1,
            ),
            _report_msg(
                anchor_id="living-room-uwb-2",
                azimuth_deg=-20.0,
                azimuth_variance_deg2=16.0,
                epoch_seq=1,
            ),
        ]
    )
    adapter = AnchorRangingMqttAdapter(
        adapter_id="uwb-home",
        deps=AdapterDeps(site_config=provider, clock=lambda: _T0),
        stream=stream,
        calibrated_floor_plan_azimuth=False,
        node_calibrations={
            "living-room-uwb-2": AnchorNodeCalibrationConfig(calibrated_floor_plan_azimuth=True),
        },
    )
    emitted = [m async for m in adapter.ranges()]
    assert emitted[0].azimuth_reference_frame == "device_local"
    assert emitted[1].azimuth_reference_frame == "floor_plan"


async def test_bench_session_replay_passes_accuracy_and_matches_direct_solver_within_10pct() -> (
    None
):
    """Simulate a 12-point, 4-direction worn-tag UWB bench session and verify parity."""
    snap = _uwb_site_config()
    rng = random.Random(20261122)

    grid_points = [(x, y) for x in (1.2, 2.4, 3.6, 4.8) for y in (1.2, 2.5, 3.8)]
    directions = (
        ("N", (0.0, 1.0)),
        ("E", (1.0, 0.0)),
        ("S", (0.0, -1.0)),
        ("W", (-1.0, 0.0)),
    )
    epochs_per_dir = 5  # 12 points * 4 directions * 5 epochs = 240 epochs
    tag_z_m = 1.05

    ground_truth: list[BenchGroundTruthPoint] = []
    mqtt_messages: list[tuple[str, bytes]] = []
    epoch_seq = 0
    cursor = _T0

    for pt_idx, (gx, gy) in enumerate(grid_points):
        for dir_name, (fx, fy) in directions:
            start_at = cursor
            end_at = cursor + timedelta(seconds=epochs_per_dir)
            ground_truth.append(
                BenchGroundTruthPoint(
                    point_id=f"P{pt_idx + 1:02d}-{dir_name}",
                    x_m=gx,
                    y_m=gy,
                    z_m=tag_z_m,
                    start_at=start_at,
                    end_at=end_at,
                    facing_dir=dir_name,
                )
            )
            for step in range(epochs_per_dir):
                epoch_seq += 1
                ts = start_at + timedelta(seconds=step)
                for node in snap.nodes:
                    # 5% single-anchor packet dropout.
                    if rng.random() < 0.05:
                        continue
                    dx = node.x_m - gx
                    dy = node.y_m - gy
                    dz = node.z_m - tag_z_m
                    true_d = math.sqrt(dx * dx + dy * dy + dz * dz)
                    # Directional torso shadowing when anchor is behind wearer.
                    horiz = max(math.hypot(dx, dy), 1e-6)
                    cos_angle = (dx * fx + dy * fy) / horiz
                    if cos_angle < -0.35:
                        noisy_d = max(0.05, true_d + 0.14 + rng.gauss(0.0, 0.10))
                    else:
                        noisy_d = max(0.05, true_d + rng.gauss(0.0, 0.07))
                    mqtt_messages.append(
                        _report_msg(
                            anchor_id=node.anchor_id,
                            hardware_key=_TAG_HW_KEY,
                            epoch_seq=epoch_seq,
                            measured_at=ts,
                            sent_at=ts + timedelta(milliseconds=20),
                            distance_m=noisy_d,
                            wear_state="on_body",
                            motion_state="still",
                            charging=False,
                            battery_fraction=0.90,
                        )
                    )
            cursor = end_at

    parity = await replay_fira_anchor_bench_session(
        mqtt_messages,
        site_config=snap,
        ground_truth_points=ground_truth,
        expected_epochs=epoch_seq,
        adapter_id="uwb-home",
        uwb_default_sigma_m=0.18,
    )
    assert parity.direct_stats.passed is True
    assert parity.adapter_stats.passed is True
    assert parity.adapter_stats.median_error_m is not None
    assert parity.adapter_stats.median_error_m <= 0.30
    assert parity.adapter_stats.p95_error_m is not None
    assert parity.adapter_stats.p95_error_m <= 1.00
    assert parity.adapter_stats.availability_fraction >= 0.80
    assert parity.within_tolerance is True
    assert parity.median_rel_diff is not None and parity.median_rel_diff <= 0.10
    assert parity.p95_rel_diff is not None and parity.p95_rel_diff <= 0.10
    assert parity.availability_rel_diff <= 0.10

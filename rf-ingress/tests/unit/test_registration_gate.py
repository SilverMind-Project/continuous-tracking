"""Unit tests for RegistrationGate binding boundaries and unregistered drops."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from cts_contracts.wearables import DeviceKind

from rf_ingress.domain import SiteBinding, SiteConfigSnapshot, SiteDevice
from rf_ingress.observability import UNREGISTERED_TAG_TOTAL, DropReason
from rf_ingress.registration import RegistrationGate

_T0 = datetime(2026, 11, 19, 12, 0, 0, tzinfo=UTC)


def _snapshot() -> SiteConfigSnapshot:
    return SiteConfigSnapshot(
        floor_plan_id="household:1",
        devices=(
            SiteDevice(
                tag_id="tag-active",
                adapter_id="sim-1",
                hardware_key="hw-1",
                device_kind=DeviceKind.BLE_BEACON,
                is_active=True,
            ),
            SiteDevice(
                tag_id="tag-inactive",
                adapter_id="sim-1",
                hardware_key="hw-2",
                device_kind=DeviceKind.BLE_BEACON,
                is_active=False,
            ),
        ),
        bindings=(
            SiteBinding(
                tag_id="tag-active",
                member_id="resident-1",
                valid_from=_T0,
                valid_to=_T0 + timedelta(hours=2),
            ),
        ),
        version="reg:v1",
    )


def test_registration_gate_enforces_half_open_binding_boundaries() -> None:
    gate = RegistrationGate()
    snap = _snapshot()

    before = gate.check(
        snapshot=snap,
        tag_id="tag-active",
        adapter_id="sim-1",
        measured_at=_T0 - timedelta(microseconds=1),
    )
    assert before.allowed is False
    assert before.drop_reason == DropReason.UNBOUND_TAG

    at_start = gate.check(
        snapshot=snap,
        tag_id="tag-active",
        adapter_id="sim-1",
        measured_at=_T0,
    )
    assert at_start.allowed is True

    inside = gate.check(
        snapshot=snap,
        tag_id="tag-active",
        adapter_id="sim-1",
        measured_at=_T0 + timedelta(hours=1),
    )
    assert inside.allowed is True

    at_end = gate.check(
        snapshot=snap,
        tag_id="tag-active",
        adapter_id="sim-1",
        measured_at=_T0 + timedelta(hours=2),
    )
    assert at_end.allowed is False
    assert at_end.drop_reason == DropReason.UNBOUND_TAG


def test_registration_gate_drops_missing_config_inactive_and_unregistered() -> None:
    gate = RegistrationGate()
    snap = _snapshot()

    no_cfg = gate.check(snapshot=None, tag_id="tag-active", adapter_id="sim-1", measured_at=_T0)
    assert no_cfg.allowed is False
    assert no_cfg.drop_reason == DropReason.NO_SITE_CONFIG

    inactive = gate.check(snapshot=snap, tag_id="tag-inactive", adapter_id="sim-1", measured_at=_T0)
    assert inactive.allowed is False
    assert inactive.drop_reason == DropReason.INACTIVE_DEVICE

    before_cnt = UNREGISTERED_TAG_TOTAL.labels(adapter_id="sim-1")._value.get()
    unreg = gate.check(snapshot=snap, tag_id="tag-unknown", adapter_id="sim-1", measured_at=_T0)
    after_cnt = UNREGISTERED_TAG_TOTAL.labels(adapter_id="sim-1")._value.get()
    assert unreg.allowed is False
    assert unreg.drop_reason == DropReason.UNREGISTERED_TAG
    assert after_cnt == before_cnt + 1

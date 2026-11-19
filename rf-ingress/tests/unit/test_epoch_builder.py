"""Unit tests for EpochBuilder and core room-level fallback."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from cts_contracts.wearables import MotionState, RangingTechnology, RfNodeRole, WearState

from rf_ingress.domain import RangeMeasurement, RfNodeGeometry
from rf_ingress.solver.epoch import EpochBuilder, room_fallback_from_epoch

_T0 = datetime(2026, 11, 19, 12, 0, 0, tzinfo=UTC)


def _meas(
    anchor_id: str,
    distance_m: float,
    dt_s: float = 0.0,
    *,
    tag_id: str = "tag-1",
    epoch_seq: int | None = None,
    variance_m2: float = 0.01,
) -> RangeMeasurement:
    return RangeMeasurement(
        tag_id=tag_id,
        anchor_id=anchor_id,
        distance_m=distance_m,
        variance_m2=variance_m2,
        technology=RangingTechnology.UWB,
        measured_at=_T0 + timedelta(seconds=dt_s),
        wear_state=WearState.ON_BODY,
        motion_state=MotionState.MOVING,
        adapter_id="sim-1",
        epoch_seq=epoch_seq,
    )


def test_window_closing_and_duplicate_anchor_keeps_latest() -> None:
    builder = EpochBuilder(epoch_window_s=0.25)
    assert builder.add(_meas("a1", 2.0, 0.00)) == []
    assert builder.add(_meas("a2", 3.0, 0.05)) == []
    assert builder.add(_meas("a1", 2.2, 0.10)) == []

    emitted = builder.add(_meas("a1", 2.5, 0.30))
    assert len(emitted) == 1
    epoch = emitted[0]
    by_anchor = {m.anchor_id: m for m in epoch.measurements}
    assert set(by_anchor.keys()) == {"a1", "a2"}
    assert by_anchor["a1"].distance_m == 2.2

    remaining = builder.flush_expired(_T0 + timedelta(seconds=0.60))
    assert len(remaining) == 1
    assert remaining[0].measurements[0].distance_m == 2.5


def test_explicit_epoch_seq_groups_and_flushes_on_advance() -> None:
    builder = EpochBuilder(epoch_window_s=0.25)
    assert builder.add(_meas("a1", 1.0, 0.0, epoch_seq=7)) == []
    assert builder.add(_meas("a2", 2.0, 0.4, epoch_seq=7)) == []
    emitted = builder.add(_meas("a1", 1.1, 1.0, epoch_seq=8))
    assert len(emitted) == 1
    assert emitted[0].epoch_seq == 7
    assert len(emitted[0].measurements) == 2


def test_emits_immediately_when_all_expected_nodes_report() -> None:
    builder = EpochBuilder(epoch_window_s=0.25)
    builder.set_expected_nodes("sim-1", frozenset({"a1", "a2", "a3"}))
    assert builder.add(_meas("a1", 1.0, 0.0)) == []
    assert builder.add(_meas("a2", 2.0, 0.02)) == []
    emitted = builder.add(_meas("a3", 3.0, 0.04))
    assert len(emitted) == 1
    assert len(emitted[0].measurements) == 3
    assert builder.open_epoch_count() == 0


def test_site_config_version_change_drops_in_flight_epochs() -> None:
    builder = EpochBuilder(epoch_window_s=0.25)
    assert builder.on_site_config_version("sha256:aaa") == 0
    builder.add(_meas("a1", 1.0, 0.0, tag_id="tag-1"))
    builder.add(_meas("a1", 2.0, 0.0, tag_id="tag-2"))
    assert builder.open_epoch_count() == 2

    # Same version keeps open epochs.
    assert builder.on_site_config_version("sha256:aaa") == 0
    assert builder.open_epoch_count() == 2

    # Changed version drops all in-flight epochs so two revisions never mix.
    dropped = builder.on_site_config_version("sha256:bbb")
    assert dropped == 2
    assert builder.open_epoch_count() == 0


def test_room_fallback_picks_nearest_node_and_computes_margin_probability() -> None:
    nodes = {
        "r1": RfNodeGeometry(
            anchor_id="r1",
            x_m=1.0,
            y_m=1.0,
            z_m=1.5,
            node_role=RfNodeRole.RECEIVER,
            technology=RangingTechnology.BLE_RSSI,
            room_id="kitchen",
        ),
        "r2": RfNodeGeometry(
            anchor_id="r2",
            x_m=7.0,
            y_m=1.0,
            z_m=1.5,
            node_role=RfNodeRole.RECEIVER,
            technology=RangingTechnology.BLE_RSSI,
            room_id="hallway",
        ),
    }
    builder = EpochBuilder(epoch_window_s=0.25)
    builder.add(_meas("r1", 1.2, 0.0, variance_m2=1.0))
    builder.add(_meas("r2", 6.5, 0.01, variance_m2=4.0))
    epoch_wide = builder.flush_all()[0]

    builder.add(_meas("r1", 3.0, 1.0, variance_m2=2.0))
    builder.add(_meas("r2", 3.1, 1.01, variance_m2=2.0))
    epoch_tie = builder.flush_all()[0]

    fb_wide = room_fallback_from_epoch(epoch_wide, nodes)
    fb_tie = room_fallback_from_epoch(epoch_tie, nodes)
    assert fb_wide is not None and fb_tie is not None
    assert fb_wide.room_id == "kitchen"
    assert fb_wide.probability is not None and fb_tie.probability is not None
    assert 0.5 <= fb_tie.probability < fb_wide.probability <= 0.99

"""Unit tests for EpochBuilder and core room-level fallback."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

from cts_contracts.wearables import MotionState, RangingTechnology, RfNodeRole, WearState

from rf_ingress.domain import RangeMeasurement, RfNodeGeometry
from rf_ingress.solver.epoch import EpochBuilder, infer_epoch_room, room_fallback_from_epoch

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


def _two_room_layout() -> dict[str, str | None]:
    return {
        "k1": "room_kitchen",
        "k2": "room_kitchen",
        "k3": "room_kitchen",
        "k4": "room_kitchen",
        "l1": "room_living",
        "l2": "room_living",
        "l3": "room_living",
        "l4": "room_living",
    }


def test_multi_room_adapter_closes_early_when_all_anchors_in_single_room_report() -> None:
    builder = EpochBuilder(epoch_window_s=0.25)
    builder.on_site_config_version("v1", node_rooms_by_adapter={"sim-1": _two_room_layout()})
    for anchor_id in ("k1", "k2", "k3"):
        assert builder.add(_meas(anchor_id, 2.0, 0.0, epoch_seq=1), now=_T0) == []
    emitted = builder.add(_meas("k4", 2.0, 0.0, epoch_seq=1), now=_T0)
    assert len(emitted) == 1
    assert {m.anchor_id for m in emitted[0].measurements} == {"k1", "k2", "k3", "k4"}
    assert builder.open_epoch_count() == 0


def test_room_completion_waits_when_an_anchor_outside_the_room_is_heard() -> None:
    builder = EpochBuilder(epoch_window_s=0.25)
    builder.on_site_config_version("v1", node_rooms_by_adapter={"sim-1": _two_room_layout()})
    for anchor_id in ("k1", "l1", "k2", "k3", "k4"):
        assert builder.add(_meas(anchor_id, 2.0, 0.0, epoch_seq=1), now=_T0) == []
    # Closed by idle expiry instead, with every anchor heard.
    emitted = builder.flush_expired(_T0 + timedelta(seconds=0.30))
    assert len(emitted) == 1
    assert len(emitted[0].measurements) == 5


def test_room_with_fewer_than_three_anchors_never_closes_early() -> None:
    builder = EpochBuilder(epoch_window_s=0.25)
    builder.on_site_config_version(
        "v1",
        node_rooms_by_adapter={"sim-1": {"h1": "hall", "h2": "hall", "k1": "room_kitchen"}},
    )
    assert builder.add(_meas("h1", 1.0, 0.0, epoch_seq=1), now=_T0) == []
    assert builder.add(_meas("h2", 1.0, 0.0, epoch_seq=1), now=_T0) == []
    assert builder.open_epoch_count() == 1


def test_site_config_reload_replaces_room_grouping() -> None:
    builder = EpochBuilder(epoch_window_s=0.25)
    builder.on_site_config_version("v1", node_rooms_by_adapter={"sim-1": _two_room_layout()})
    # Revision v2 moves k4 into the living room, so k1..k3 alone now complete the kitchen.
    layout_v2 = {**_two_room_layout(), "k4": "room_living"}
    builder.on_site_config_version("v2", node_rooms_by_adapter={"sim-1": layout_v2})
    assert builder.add(_meas("k1", 2.0, 0.0, epoch_seq=1), now=_T0) == []
    assert builder.add(_meas("k2", 2.0, 0.0, epoch_seq=1), now=_T0) == []
    emitted = builder.add(_meas("k3", 2.0, 0.0, epoch_seq=1), now=_T0)
    assert len(emitted) == 1


def test_expiry_runs_on_server_idle_time_not_device_measured_at() -> None:
    """A batched epoch captured seconds before arrival is not expired on arrival."""
    builder = EpochBuilder(epoch_window_s=0.25)
    arrival = _T0 + timedelta(seconds=3.0)
    assert builder.add(_meas("a1", 1.0, 0.0, epoch_seq=5), now=arrival) == []
    # A flush tick between two measurements of the same batch must not split it.
    assert builder.flush_expired(arrival + timedelta(seconds=0.01)) == []
    assert (
        builder.add(_meas("a2", 2.0, 0.0, epoch_seq=5), now=arrival + timedelta(seconds=0.02)) == []
    )
    assert builder.flush_expired(arrival + timedelta(seconds=0.20)) == []
    emitted = builder.flush_expired(arrival + timedelta(seconds=0.28))
    assert len(emitted) == 1
    assert {m.anchor_id for m in emitted[0].measurements} == {"a1", "a2"}


def test_epoch_that_keeps_receiving_is_bounded_by_max_open_s() -> None:
    builder = EpochBuilder(epoch_window_s=0.25, max_open_s=1.0)
    for step in range(5):
        builder.add(_meas("a1", 1.0, 0.0, epoch_seq=9), now=_T0 + timedelta(seconds=0.2 * step))
    assert builder.flush_expired(_T0 + timedelta(seconds=0.9)) == []
    assert len(builder.flush_expired(_T0 + timedelta(seconds=1.0))) == 1


def test_late_straggler_for_closed_sequence_is_dropped_not_reopened() -> None:
    late: list[int] = []
    builder = EpochBuilder(epoch_window_s=0.25, on_late_measurement=lambda: late.append(1))
    builder.on_site_config_version("v1", node_rooms_by_adapter={"sim-1": _two_room_layout()})
    for anchor_id in ("k1", "k2", "k3"):
        builder.add(_meas(anchor_id, 2.0, 0.0, epoch_seq=7), now=_T0)
    assert len(builder.add(_meas("k4", 2.0, 0.0, epoch_seq=7), now=_T0)) == 1

    # A living-room anchor reporting the same round late would otherwise open a
    # duplicate 1-anchor epoch whose room fallback names the wrong room.
    assert builder.add(_meas("l1", 4.0, 0.0, epoch_seq=7), now=_T0 + timedelta(seconds=0.1)) == []
    assert builder.open_epoch_count() == 0
    assert late == [1]

    # The same straggler while the next round is open must not close that round.
    builder.add(_meas("k1", 2.0, 0.5, epoch_seq=8), now=_T0 + timedelta(seconds=0.5))
    assert builder.add(_meas("l2", 4.0, 0.0, epoch_seq=7), now=_T0 + timedelta(seconds=0.55)) == []
    assert builder.open_epoch_count() == 1
    assert late == [1, 1]


def test_restarted_sequence_counter_is_not_treated_as_straggler() -> None:
    late: list[int] = []
    builder = EpochBuilder(
        epoch_window_s=0.25, straggler_horizon_s=1.0, on_late_measurement=lambda: late.append(1)
    )
    builder.add(_meas("a1", 1.0, 0.0, epoch_seq=41), now=_T0)
    assert len(builder.flush_expired(_T0 + timedelta(seconds=0.3))) == 1
    # Device restart: the counter starts again below the last closed sequence.
    assert builder.add(_meas("a1", 1.0, 0.4, epoch_seq=1), now=_T0 + timedelta(seconds=0.4)) == []
    assert builder.open_epoch_count() == 1
    assert len(builder.flush_expired(_T0 + timedelta(seconds=0.7))) == 1
    # A repeat of the last closed sequence after the horizon is a new round, not a straggler.
    assert builder.add(_meas("a1", 1.0, 2.0, epoch_seq=1), now=_T0 + timedelta(seconds=2.0)) == []
    assert builder.open_epoch_count() == 1
    assert late == []


def test_infer_epoch_room_selects_nearest_inlier_anchor_room_with_honest_probability() -> None:
    nodes = {
        "k1": RfNodeGeometry(anchor_id="k1", x_m=0.0, y_m=0.0, z_m=2.2, room_id="room_kitchen"),
        "k2": RfNodeGeometry(anchor_id="k2", x_m=4.0, y_m=0.0, z_m=1.3, room_id="room_kitchen"),
        "k3": RfNodeGeometry(anchor_id="k3", x_m=4.0, y_m=4.0, z_m=2.2, room_id="room_kitchen"),
        "l1": RfNodeGeometry(anchor_id="l1", x_m=4.5, y_m=4.0, z_m=1.3, room_id="room_living"),
    }
    builder = EpochBuilder(epoch_window_s=0.25)
    for anchor_id, dist in (("k1", 2.0), ("k2", 2.5), ("k3", 3.0), ("l1", 0.4)):
        builder.add(_meas(anchor_id, dist, 0.0, epoch_seq=3))
    epoch = builder.flush_all()[0]

    nearest = infer_epoch_room(epoch, nodes)
    assert nearest is not None
    assert nearest[0] == "room_living"
    # Excluding the solver's dropped outlier anchor moves attribution to the inliers.
    inliers = infer_epoch_room(epoch, nodes, exclude_anchor_id="l1")
    assert inliers is not None
    assert inliers[0] == "room_kitchen"
    assert 0.0 < inliers[1] < 1.0

    no_rooms = {
        aid: RfNodeGeometry(anchor_id=aid, x_m=n.x_m, y_m=n.y_m, z_m=n.z_m)
        for aid, n in nodes.items()
    }
    assert infer_epoch_room(epoch, no_rooms) is None


def test_batched_round_spanning_two_rooms_is_not_truncated_by_room_completion() -> None:
    """A batch lists one room first; the rest of the same message must not be stragglers."""
    late: list[int] = []
    builder = EpochBuilder(epoch_window_s=0.25, on_late_measurement=lambda: late.append(1))
    builder.on_site_config_version("v1", node_rooms_by_adapter={"sim-1": _two_room_layout()})
    sent_at = _T0 + timedelta(seconds=2.0)
    arrival = _T0 + timedelta(seconds=2.05)
    for anchor_id in ("k1", "k2", "k3", "k4", "l1", "l2", "l3"):
        measurement = replace(_meas(anchor_id, 3.0, 0.0, epoch_seq=1), sent_at=sent_at)
        assert builder.add(measurement, now=arrival) == []
    emitted = builder.flush_expired(arrival + timedelta(seconds=0.3))
    assert len(emitted) == 1
    assert len(emitted[0].measurements) == 7
    assert late == []

"""Unit tests for WalkingBoutSegmenter."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.trajectory.gait import GaitConfig, WalkingBout, WalkingBoutSegmenter

_T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
_IDENTITY = "alice"
_PH = "ph-001"


def _t(seconds: float) -> datetime:
    return _T0 + timedelta(seconds=seconds)


def _seg(cfg: GaitConfig | None = None) -> WalkingBoutSegmenter:
    return WalkingBoutSegmenter(cfg or GaitConfig())


def _ingest_series(
    seg: WalkingBoutSegmenter,
    speeds: list[float | None],
    interval_s: float = 1.0,
    ph_id: str = _PH,
    identity_id: str = _IDENTITY,
    x_start: float = 0.0,
    step_x: float = 0.5,
) -> list[WalkingBout]:
    """Feed a list of speeds to the segmenter; return any emitted bouts."""
    bouts: list[WalkingBout] = []
    for i, speed in enumerate(speeds):
        t = _t(i * interval_s)
        x = x_start + i * step_x
        result = seg.ingest(
            ph_id=ph_id,
            identity_id=identity_id,
            captured_at=t,
            floor_speed_m_s=speed,
            floor_x_m=x,
            floor_y_m=0.0,
            room_name="hallway",
        )
        if result is not None:
            bouts.append(result)
    return bouts


class TestOpenClose:
    def test_sustained_walking_produces_bout_on_flush(self) -> None:
        seg = _seg()
        # 5 s at 0.6 m/s — should stay open
        bouts = _ingest_series(seg, [0.6] * 5)
        assert bouts == []
        bout = seg.flush_ph(_PH, _t(5.0))
        assert bout is not None
        assert bout.identity_id == _IDENTITY
        assert bout.sample_count == 5
        assert bout.median_speed_m_s == pytest.approx(0.6)

    def test_bout_closes_after_grace_window(self) -> None:
        cfg = GaitConfig(bout_min_speed_m_s=0.3, bout_close_grace_s=2.0, bout_min_duration_s=3.0)
        seg = _seg(cfg)
        # 5 s walking, then 3 s below threshold (> grace 2 s) closes the bout
        speeds: list[float | None] = [0.6] * 5 + [0.1] * 3
        bouts = _ingest_series(seg, speeds)
        assert len(bouts) == 1
        assert bouts[0].sample_count == 5

    def test_grace_gap_bridges_bout(self) -> None:
        """1.5 s dip below threshold does NOT split the bout."""
        cfg = GaitConfig(bout_min_speed_m_s=0.3, bout_close_grace_s=2.0, bout_min_duration_s=3.0)
        seg = _seg(cfg)
        # 4 s walking, 1 s below (inside grace), 4 s walking, then flush
        speeds: list[float | None] = [0.6] * 4 + [0.1] * 1 + [0.6] * 4
        _ingest_series(seg, speeds)
        bout = seg.flush_ph(_PH, _t(9.0))
        assert bout is not None
        # All 8 above-threshold samples should be in one bout
        assert bout.sample_count == 8


class TestDiscard:
    def test_short_bout_discarded(self) -> None:
        cfg = GaitConfig(bout_min_duration_s=3.0, bout_close_grace_s=2.0)
        seg = _seg(cfg)
        # 2 s walking (< 3 s min) then long pause
        speeds: list[float | None] = [0.6] * 2 + [0.0] * 4
        bouts = _ingest_series(seg, speeds)
        assert bouts == []
        assert seg.flush_ph(_PH, _t(6.0)) is None

    def test_low_median_bout_discarded(self) -> None:
        cfg = GaitConfig(bout_min_duration_s=3.0, min_median_speed_m_s=0.2, bout_close_grace_s=2.0)
        seg = _seg(cfg)
        # 5 s at 0.15 m/s (median below 0.2 threshold)
        speeds: list[float | None] = [0.15] * 5 + [0.0] * 4
        bouts = _ingest_series(seg, speeds, interval_s=1.0)
        assert bouts == []
        assert seg.flush_ph(_PH, _t(9.0)) is None


class TestGlitchSamples:
    def test_glitch_excluded_bout_stays_open(self) -> None:
        cfg = GaitConfig(max_plausible_speed_m_s=2.5, bout_min_duration_s=3.0)
        seg = _seg(cfg)
        # 3 s normal, 1 glitch, 3 s normal
        speeds: list[float | None] = [0.6, 0.6, 0.6, 5.0, 0.6, 0.6, 0.6]
        _ingest_series(seg, speeds)
        bout = seg.flush_ph(_PH, _t(7.0))
        assert bout is not None
        # Glitch sample excluded from count
        assert bout.sample_count == 6

    def test_glitch_does_not_reset_bout(self) -> None:
        seg = _seg()
        speeds: list[float | None] = [0.6] * 5 + [99.0]
        _ingest_series(seg, speeds)
        # Bout still open after glitch
        bout = seg.flush_ph(_PH, _t(6.0))
        assert bout is not None


class TestUncalibrated:
    def test_none_speed_ignored(self) -> None:
        seg = _seg()
        speeds: list[float | None] = [None, None, 0.6, None, 0.6, None]
        _ingest_series(seg, speeds)
        # No sustained bout — only 2 above-threshold samples in isolation
        bout = seg.flush_ph(_PH, _t(6.0))
        # Duration < bout_min_duration_s so discarded
        assert bout is None

    def test_none_frames_do_not_close_bout(self) -> None:
        cfg = GaitConfig(bout_min_duration_s=3.0)
        seg = _seg(cfg)
        # 5 s walking, then None frames (should not close the bout via grace logic)
        speeds: list[float | None] = [0.6] * 5 + [None, None]
        _ingest_series(seg, speeds)
        bout = seg.flush_ph(_PH, _t(7.0))
        assert bout is not None
        assert bout.sample_count == 5


class TestDistance:
    def test_displacement_beats_speed_x_time(self) -> None:
        """Square-path fixture: integrated displacement != speed x time."""
        cfg = GaitConfig(bout_min_duration_s=3.0)
        seg = _seg(cfg)
        # Walk a 1 m x 1 m square over 4 steps: (0,0)→(1,0)→(1,1)→(0,1)→(0,0)
        positions = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        for i, (x, y) in enumerate(positions):
            seg.ingest(
                ph_id=_PH,
                identity_id=_IDENTITY,
                captured_at=_t(float(i)),
                floor_speed_m_s=0.6,
                floor_x_m=x,
                floor_y_m=y,
                room_name="hallway",
            )
        bout = seg.flush_ph(_PH, _t(5.0))
        assert bout is not None
        # Integrated displacement: 4 sides x 1 m = 4.0 m
        assert bout.distance_m == pytest.approx(4.0, abs=0.01)
        # speed x time would be 0.6 x 5 = 3.0 m — different from displacement
        assert bout.distance_m != pytest.approx(0.6 * 5.0, abs=0.1)


class TestPHClose:
    def test_flush_ph_clears_state(self) -> None:
        seg = _seg()
        _ingest_series(seg, [0.6] * 5)
        seg.flush_ph(_PH, _t(5.0))
        # After flush, state is cleared; second flush returns None
        assert seg.flush_ph(_PH, _t(6.0)) is None

    def test_flush_all(self) -> None:
        seg = _seg()
        _ingest_series(seg, [0.6] * 5, ph_id="ph-a", identity_id="alice")
        _ingest_series(seg, [0.6] * 5, ph_id="ph-b", identity_id="bob")
        bouts = seg.flush_all(_t(5.0))
        assert len(bouts) == 2
        identities = {b.identity_id for b in bouts}
        assert identities == {"alice", "bob"}


class TestIdempotentBoutId:
    def test_stable_bout_id(self) -> None:
        seg = _seg()
        _ingest_series(seg, [0.6] * 5)
        bout = seg.flush_ph(_PH, _t(5.0))
        assert bout is not None
        # Calling bout_id twice gives the same value
        assert bout.bout_id == bout.bout_id
        # Another bout starting at the same time with the same identity produces
        # the same UUID (idempotent re-processing guarantee)
        bout2 = WalkingBout(
            identity_id=bout.identity_id,
            started_at=bout.started_at,
            ended_at=bout.ended_at,
            duration_s=bout.duration_s,
            distance_m=bout.distance_m,
            median_speed_m_s=bout.median_speed_m_s,
            p95_speed_m_s=bout.p95_speed_m_s,
            sample_count=bout.sample_count,
            rooms=bout.rooms,
        )
        assert bout2.bout_id == bout.bout_id

"""Tests for PositionTrackBuffer confirmation, watermark consumption, gap reset, and bounds."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.domain import AnonymousPositionObservation
from app.services.position_track_buffer import PositionTrackBuffer

_T0 = datetime(2026, 11, 25, 10, 0, 0, tzinfo=UTC)


def _obs(
    *,
    sensor_id: str = "radar-1",
    track_id: str = "t-1",
    offset_s: float = 0.0,
    x_m: float = 2.0,
    y_m: float = 3.0,
) -> AnonymousPositionObservation:
    return AnonymousPositionObservation(
        sensor_id=sensor_id,
        sensor_track_id=track_id,
        measured_at=_T0 + timedelta(seconds=offset_s),
        x_m=x_m,
        y_m=y_m,
        cov_m2=(0.04, 0.0, 0.0, 0.04),
        calibration_residual_m=0.10,
    )


def test_buffer_confirmation_peek_and_single_consumption() -> None:
    buf = PositionTrackBuffer(max_age_s=10.0, track_gap_s=2.0, max_tracks=8)
    buf.add(_obs(offset_s=0.0, x_m=1.0))
    buf.add(_obs(offset_s=0.5, x_m=1.1))
    # Duplicate timestamp ignored cleanly.
    buf.add(_obs(offset_s=0.5, x_m=9.9))

    now_unconfirmed = _T0 + timedelta(seconds=0.6)
    assert buf.peek_round_tracks(now=now_unconfirmed, max_age_s=1.5, min_track_age_s=1.0) == ()
    assert buf.take_round_tracks(now=now_unconfirmed, max_age_s=1.5, min_track_age_s=1.0) == ()

    buf.add(_obs(offset_s=1.2, x_m=1.3))
    now_confirmed = _T0 + timedelta(seconds=1.3)
    peeked = buf.peek_round_tracks(now=now_confirmed, max_age_s=1.5, min_track_age_s=1.0)
    assert len(peeked) == 1
    assert peeked[0].x_m == pytest.approx(1.3)

    taken = buf.take_round_tracks(now=now_confirmed, max_age_s=1.5, min_track_age_s=1.0)
    assert len(taken) == 1
    assert taken[0].x_m == pytest.approx(1.3)

    # Same confirmed sample is never returned a second time.
    assert buf.take_round_tracks(now=now_confirmed, max_age_s=1.5, min_track_age_s=1.0) == ()


def test_track_gap_resets_confirmation_age_and_max_tracks_evicts_oldest() -> None:
    buf = PositionTrackBuffer(max_age_s=10.0, track_gap_s=2.0, max_tracks=2)
    buf.add(_obs(track_id="t-1", offset_s=0.0, x_m=1.0))
    buf.add(_obs(track_id="t-1", offset_s=1.2, x_m=1.2))
    assert (
        len(
            buf.take_round_tracks(
                now=_T0 + timedelta(seconds=1.3),
                max_age_s=1.5,
                min_track_age_s=1.0,
            )
        )
        == 1
    )

    # A gap > track_gap_s resets track start age so the new target waits for confirmation.
    buf.add(_obs(track_id="t-1", offset_s=4.0, x_m=5.0))
    assert (
        buf.take_round_tracks(
            now=_T0 + timedelta(seconds=4.1),
            max_age_s=1.5,
            min_track_age_s=1.0,
        )
        == ()
    )
    buf.add(_obs(track_id="t-1", offset_s=5.1, x_m=5.1))
    after_gap = buf.take_round_tracks(
        now=_T0 + timedelta(seconds=5.2),
        max_age_s=1.5,
        min_track_age_s=1.0,
    )
    assert len(after_gap) == 1
    assert after_gap[0].x_m == pytest.approx(5.1)

    # Track bound evicts least-recent track when capacity is exceeded.
    buf.add(_obs(track_id="t-2", offset_s=6.0))
    buf.add(_obs(track_id="t-3", offset_s=7.0))
    assert len(buf) == 2
    assert ("radar-1", "t-1") not in buf._tracks

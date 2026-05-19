"""Tests for IdentityCommitter."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.tracking.identity_committer import CommitDecision, IdentityCommitter


def _committer(window_s: float = 3.0) -> IdentityCommitter:
    return IdentityCommitter(commit_window_s=window_s)


class TestIdentityCommitter:
    def test_flush_empty(self) -> None:
        c = _committer()
        assert c.flush() == []

    def test_flush_before_window_expires(self) -> None:
        c = _committer(window_s=60.0)
        now = datetime.now(UTC)
        c.ingest("gt-1", "alice", 0.9)
        # Window hasn't expired — nothing flushed.
        assert c.flush(now=now) == []

    def test_flush_after_window_expires(self) -> None:
        c = _committer(window_s=3.0)
        past = datetime.now(UTC) - timedelta(seconds=10)
        c._buffer["gt-1"] = [(past, "alice", 0.9, "face")]
        decisions = c.flush()
        assert len(decisions) == 1
        assert decisions[0].identity_id == "alice"
        assert decisions[0].buffered is True
        # Buffer cleared after flush.
        assert "gt-1" not in c._buffer

    def test_flush_takes_latest_when_all_non_none(self) -> None:
        c = _committer(window_s=3.0)
        t0 = datetime.now(UTC) - timedelta(seconds=10)
        t1 = t0 + timedelta(seconds=1)
        t2 = t1 + timedelta(seconds=1)
        c._buffer["gt-1"] = [
            (t0, "alice", 0.7, "reid"),
            (t1, "alice", 0.8, "reid"),
            (t2, "alice", 0.9, "face"),
        ]
        decisions = c.flush()
        assert decisions[0].identity_id == "alice"
        assert decisions[0].confidence == pytest.approx(0.9)

    def test_flush_prefers_non_none_over_trailing_none(self) -> None:
        """When the latest entry is None but earlier entries in the window
        confirmed an identity, the confirmed identity must win.

        Regression guard for the race where face commits alice then quiet
        frames (before Fix 1 was applied to the resolver) produced None,
        causing flush() to clear the valid assignment.
        """
        c = _committer(window_s=3.0)
        t0 = datetime.now(UTC) - timedelta(seconds=5)
        t1 = t0 + timedelta(seconds=1)
        t2 = t1 + timedelta(seconds=1)
        c._buffer["gt-1"] = [
            (t0, None, 0.4, ""),      # before face fired
            (t1, "alice", 0.92, "face_high_confidence"),
            (t2, None, 0.57, ""),     # maintenance bug (pre-fix) produced None
        ]
        decisions = c.flush()
        assert decisions[0].identity_id == "alice", (
            "flush() must not clear a valid identity when the latest entry is None "
            "but an earlier entry in the same window confirmed an identity"
        )

    def test_flush_emits_none_when_all_none(self) -> None:
        """Genuine UNKNOWN result: all entries in the window are None."""
        c = _committer(window_s=3.0)
        t0 = datetime.now(UTC) - timedelta(seconds=5)
        t1 = t0 + timedelta(seconds=1)
        c._buffer["gt-1"] = [
            (t0, None, 0.3, ""),
            (t1, None, 0.4, ""),
        ]
        decisions = c.flush()
        assert decisions[0].identity_id is None

    def test_high_confidence_face_fast_path(self) -> None:
        c = _committer()
        result = c.check_high_confidence_face("gt-1", "alice", 0.9)
        assert result is not None
        assert result.identity_id == "alice"
        assert result.buffered is False

    def test_high_confidence_face_below_threshold(self) -> None:
        c = _committer()
        result = c.check_high_confidence_face("gt-1", "alice", 0.5)
        assert result is None

    def test_clear_track_removes_buffer(self) -> None:
        c = _committer()
        c.ingest("gt-1", "alice", 0.9)
        c.clear_track("gt-1")
        assert "gt-1" not in c._buffer

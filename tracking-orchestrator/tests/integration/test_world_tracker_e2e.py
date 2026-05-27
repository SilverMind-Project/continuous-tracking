"""N8: M1 world tracker end-to-end replay tests.

Replays recorded FrameReady fixtures through the full pipeline and asserts
correct PH lifecycle: one PH across two cameras, two PHs across two rooms.

These tests require testcontainer Postgres + Redis and replay fixtures.
Marked ``@pytest.mark.integration`` — skipped by default; CI opts in.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "frame_replays"


def _read_frames(path: Path) -> list[bytes]:
    """Read a length-prefixed protobuf FrameReady stream."""
    frames = []
    with open(path, "rb") as f:
        while True:
            len_bytes = f.read(4)
            if not len_bytes:
                break
            (msg_len,) = struct.unpack(">I", len_bytes)
            frame_bytes = f.read(msg_len)
            if len(frame_bytes) != msg_len:
                break
            frames.append(frame_bytes)
    return frames


@pytest.mark.integration
@pytest.mark.skip(reason="Requires testcontainer Postgres + Redis + replay fixtures")
class TestWorldTrackerE2EOnePersonTwoCameras:
    """Replay two_cameras_one_room.bin and assert single-PH lifecycle."""

    async def test_single_ph_covers_both_cameras(self):
        """One person across two cameras produces exactly one PH."""
        fixture = FIXTURES_DIR / "two_cameras_one_room.bin"
        if not fixture.exists():
            pytest.skip(f"Fixture not found: {fixture}")

        frames = _read_frames(fixture)
        assert len(frames) > 0, "Fixture should contain at least one frame"

        # TODO: wire testcontainer Postgres + Redis, run frames through pipeline
        # - assert exactly one PH in repository
        # - assert PH observation IDs cover both camera IDs
        # - assert PH current_identity_id is committed or UNKNOWN with valid posterior


@pytest.mark.integration
@pytest.mark.skip(reason="Requires testcontainer Postgres + Redis + replay fixtures")
class TestWorldTrackerE2ETwoPeopleTwoRooms:
    """Replay two_rooms_two_people.bin and assert two-PH separation."""

    async def test_two_phs_do_not_merge(self):
        """Two people in separate rooms produce exactly two PHs with correct room attribution."""
        fixture = FIXTURES_DIR / "two_rooms_two_people.bin"
        if not fixture.exists():
            pytest.skip(f"Fixture not found: {fixture}")

        frames = _read_frames(fixture)
        assert len(frames) > 0, "Fixture should contain at least one frame"

        # TODO: wire testcontainer Postgres + Redis, run frames through pipeline
        # - assert exactly two PHs in repository
        # - assert they do not merge
        # - assert room associations match recording

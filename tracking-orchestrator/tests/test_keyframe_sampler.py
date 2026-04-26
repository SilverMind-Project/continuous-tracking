"""Unit tests for KeyframeSampler.

Tests cover:
- First call always samples (no prior timer).
- Second call within the interval returns None.
- Second call after the interval returns a keyframe.
- trigger_sample always returns a keyframe regardless of interval.
- trigger_sample does not reset the periodic timer.
- Periodic and trigger samples have different expiry durations.
- Multiple tracklets are tracked independently.
- reset_tracklet clears the periodic timer.
- Keyframes are persisted in the repository.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.sampling.keyframe_sampler import KeyframeSampler, SamplerConfig
from app.storage.base import InMemoryKeyframeRepository


@pytest.fixture()
def repo() -> InMemoryKeyframeRepository:
    return InMemoryKeyframeRepository()


@pytest.fixture()
def sampler(repo: InMemoryKeyframeRepository) -> KeyframeSampler:
    config = SamplerConfig(
        keyframe_min_interval_s=30.0,
        periodic_expires_hours=72,
        trigger_expires_days=30,
    )
    return KeyframeSampler(repo=repo, config=config)


_T0 = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
_ANNS: dict[str, object] = {"bbox": [10, 20, 100, 200]}


async def test_first_sample_always_taken(
    sampler: KeyframeSampler,
    repo: InMemoryKeyframeRepository,
) -> None:
    kf = await sampler.maybe_sample("tl-001", "gt-001", "cam-a", "frame.jpg", _T0, _ANNS)
    assert kf is not None
    assert kf.tracklet_id == "tl-001"
    assert kf.tag_reason == "periodic"

    stored = await repo.list_keyframes(tracklet_id="tl-001")
    assert len(stored) == 1


async def test_within_interval_returns_none(sampler: KeyframeSampler) -> None:
    await sampler.maybe_sample("tl-001", "gt-001", "cam-a", "f1.jpg", _T0, _ANNS)
    # 15 seconds later — within the 30s interval.
    kf = await sampler.maybe_sample(
        "tl-001", "gt-001", "cam-a", "f2.jpg", _T0 + timedelta(seconds=15), _ANNS
    )
    assert kf is None


async def test_after_interval_returns_keyframe(
    sampler: KeyframeSampler,
    repo: InMemoryKeyframeRepository,
) -> None:
    await sampler.maybe_sample("tl-001", "gt-001", "cam-a", "f1.jpg", _T0, _ANNS)
    t2 = _T0 + timedelta(seconds=31)
    kf = await sampler.maybe_sample("tl-001", "gt-001", "cam-a", "f2.jpg", t2, _ANNS)
    assert kf is not None
    assert kf.captured_at == t2

    stored = await repo.list_keyframes(tracklet_id="tl-001")
    assert len(stored) == 2


async def test_trigger_sample_always_returns(
    sampler: KeyframeSampler,
    repo: InMemoryKeyframeRepository,
) -> None:
    # Take a periodic sample first.
    await sampler.maybe_sample("tl-001", "gt-001", "cam-a", "f1.jpg", _T0, _ANNS)
    # Trigger 5 seconds later — well within the 30s periodic interval.
    t2 = _T0 + timedelta(seconds=5)
    kf = await sampler.trigger_sample(
        "tl-001", "gt-001", "cam-a", "f2.jpg", t2, _ANNS, tag_reason="identity_changed"
    )
    assert kf is not None
    assert kf.tag_reason == "identity_changed"

    stored = await repo.list_keyframes(tracklet_id="tl-001")
    assert len(stored) == 2


async def test_trigger_does_not_reset_periodic_timer(sampler: KeyframeSampler) -> None:
    await sampler.maybe_sample("tl-001", "gt-001", "cam-a", "f1.jpg", _T0, _ANNS)
    t_trigger = _T0 + timedelta(seconds=5)
    await sampler.trigger_sample("tl-001", "gt-001", "cam-a", "f2.jpg", t_trigger, _ANNS, "hazard")
    # The periodic timer was set at T0; 29s total — should still be None.
    t_check = _T0 + timedelta(seconds=29)
    kf = await sampler.maybe_sample("tl-001", "gt-001", "cam-a", "f3.jpg", t_check, _ANNS)
    assert kf is None


async def test_periodic_expiry_is_shorter_than_trigger(sampler: KeyframeSampler) -> None:
    periodic = await sampler.maybe_sample("tl-001", "gt-001", "cam-a", "f1.jpg", _T0, _ANNS)
    assert periodic is not None
    periodic_duration = (periodic.expires_at - _T0).total_seconds()

    trigger = await sampler.trigger_sample(
        "tl-002", "gt-002", "cam-b", "f2.jpg", _T0, _ANNS, "identity_changed"
    )
    trigger_duration = (trigger.expires_at - _T0).total_seconds()

    assert periodic_duration < trigger_duration
    assert abs(periodic_duration - 72 * 3600) < 1
    assert abs(trigger_duration - 30 * 86400) < 1


async def test_multiple_tracklets_independent(
    sampler: KeyframeSampler,
    repo: InMemoryKeyframeRepository,
) -> None:
    # Sample tracklet 1.
    await sampler.maybe_sample("tl-001", "gt-001", "cam-a", "f1.jpg", _T0, _ANNS)
    # 15s later: tl-001 is still within interval, tl-002 has no timer yet.
    t2 = _T0 + timedelta(seconds=15)
    kf1 = await sampler.maybe_sample("tl-001", "gt-001", "cam-a", "f2.jpg", t2, _ANNS)
    kf2 = await sampler.maybe_sample("tl-002", "gt-002", "cam-b", "f3.jpg", t2, _ANNS)

    assert kf1 is None  # within interval
    assert kf2 is not None  # first sample for tl-002


async def test_reset_tracklet_clears_timer(sampler: KeyframeSampler) -> None:
    await sampler.maybe_sample("tl-001", "gt-001", "cam-a", "f1.jpg", _T0, _ANNS)
    sampler.reset_tracklet("tl-001")
    # After reset, the next call should sample even if within the old interval.
    t2 = _T0 + timedelta(seconds=5)
    kf = await sampler.maybe_sample("tl-001", "gt-001", "cam-a", "f2.jpg", t2, _ANNS)
    assert kf is not None


async def test_keyframe_stored_with_correct_fields(
    sampler: KeyframeSampler,
    repo: InMemoryKeyframeRepository,
) -> None:
    anns = {"bbox": [1, 2, 3, 4], "person_id": "alice"}
    kf = await sampler.maybe_sample("tl-001", "gt-001", "cam-x", "frames/x.jpg", _T0, anns)
    assert kf is not None
    stored = await repo.list_keyframes(tracklet_id="tl-001")
    assert len(stored) == 1
    s = stored[0]
    assert s.camera_id == "cam-x"
    assert s.minio_key == "frames/x.jpg"
    assert s.annotations == anns
    assert s.tag_reason == "periodic"


async def test_keyframe_direct_lookup_and_retention_update(
    sampler: KeyframeSampler,
    repo: InMemoryKeyframeRepository,
) -> None:
    """Issue #34: O(1) lookup/update methods back dashboard keyframe routes."""
    sample = await sampler.trigger_sample(
        tracklet_id="tl-retain",
        global_track_id="gt-retain",
        camera_id="cam-1",
        minio_key="frames/retain.jpg",
        captured_at=_T0,
        annotations={},
        tag_reason="identity_changed",
    )
    assert sample is not None

    fetched = await repo.get_keyframe(sample.keyframe_id)
    assert fetched == sample

    new_expiry = _T0 + timedelta(days=30)
    assert await repo.update_retention(sample.keyframe_id, new_expiry)
    retained = await repo.get_keyframe(sample.keyframe_id)
    assert retained is not None
    assert retained.expires_at == new_expiry
    assert not await repo.update_retention("missing", new_expiry)

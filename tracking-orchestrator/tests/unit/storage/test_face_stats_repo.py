"""InMemoryFaceObservationStatsRepository: append and time-range queries."""

from __future__ import annotations

from datetime import timedelta

from app.storage.face_stats import InMemoryFaceObservationStatsRepository
from tests.fixtures.face_stats import T0 as _T0
from tests.fixtures.face_stats import face_stat


async def test_list_between_is_half_open_and_oldest_first() -> None:
    repo = InMemoryFaceObservationStatsRepository()
    await repo.append_many([face_stat(30), face_stat(0), face_stat(60), face_stat(10)])

    rows = await repo.list_between(_T0, _T0 + timedelta(seconds=60))

    assert [r.observed_at for r in rows] == [
        _T0,
        _T0 + timedelta(seconds=10),
        _T0 + timedelta(seconds=30),
    ]


async def test_camera_filter_returns_only_that_camera() -> None:
    repo = InMemoryFaceObservationStatsRepository()
    await repo.append_many([face_stat(0, "cam-1"), face_stat(1, "cam-2"), face_stat(2, "cam-1")])

    rows = await repo.list_between(_T0, _T0 + timedelta(hours=1), camera_id="cam-2")

    assert [r.camera_id for r in rows] == ["cam-2"]


async def test_append_empty_is_noop() -> None:
    repo = InMemoryFaceObservationStatsRepository()
    await repo.append_many([])
    assert await repo.list_between(_T0 - timedelta(days=1), _T0 + timedelta(days=1)) == []

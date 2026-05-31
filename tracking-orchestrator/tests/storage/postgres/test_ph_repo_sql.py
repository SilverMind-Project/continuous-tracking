"""PostgresPHRepository SQL shape tests."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from app.domain import PersonHypothesis
from app.storage.postgres.ph_repo import PostgresPHRepository


class _Acquire:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _Conn:
        return self._conn

    async def __aexit__(self, *_args: object) -> None:
        return None


class _Pool:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    def acquire(self) -> _Acquire:
        return _Acquire(self._conn)


class _Conn:
    def __init__(self, fetchrow_result: dict[str, object] | None = None) -> None:
        self.fetchrow_result = fetchrow_result
        self.fetch_sql = ""
        self.execute_sql = ""
        self.execute_args: tuple[object, ...] = ()

    async def fetchrow(self, _sql: str, _ph_id: str) -> dict[str, object] | None:
        if self.fetchrow_result is not None:
            return self.fetchrow_result
        return {"last_seen_at": datetime.now(UTC), "state_mean": [1.0, 2.0, 0.0, 0.0]}

    async def fetch(self, sql: str, *_args: object) -> list[dict[str, object]]:
        self.fetch_sql = sql
        return []

    async def execute(self, sql: str, *args: object) -> str:
        self.execute_sql = sql
        self.execute_args = args
        return "OK"


@pytest.mark.asyncio
async def test_get_co_present_casts_reference_time_for_interval_math() -> None:
    conn = _Conn()
    repo = PostgresPHRepository(_Pool(conn))

    result = await repo.get_co_present("ph-1")

    assert result == []
    assert "$2::timestamptz - INTERVAL '30 seconds'" in conn.fetch_sql
    assert "$2::timestamptz + INTERVAL '30 seconds'" in conn.fetch_sql
    assert "closed_at IS NULL" in conn.fetch_sql
    assert "state_mean[1]::double precision" in conn.fetch_sql
    assert "state_mean[2]::double precision" in conn.fetch_sql


def _ph_row(metadata: object) -> dict[str, object]:
    now = datetime.now(UTC)
    return {
        "ph_id": "ph-1",
        "state_mean": [1.0, 2.0, 0.0, 0.0],
        "state_cov": [0.1] * 16,
        "born_at": now,
        "last_seen_at": now,
        "last_seen_camera": "cam-1",
        "observation_count": 2,
        "current_identity_id": None,
        "current_identity_committed_at": None,
        "gallery_mean": None,
        "height_m": None,
        "active_cameras": ["cam-1"],
        "closed_at": None,
        "metadata": metadata,
        "mean_quality": 0.0,
    }


def _domain_ph(metadata: Any = None) -> PersonHypothesis:
    now = datetime.now(UTC)
    return PersonHypothesis(
        ph_id="ph-1",
        state_mean=(1.0, 2.0, 0.0, 0.0),
        state_cov=(0.1,) * 16,
        born_at=now,
        last_seen_at=now,
        last_seen_camera="cam-1",
        observation_count=2,
        active_cameras=frozenset({"cam-1"}),
        metadata={} if metadata is None else metadata,
    )


@pytest.mark.asyncio
async def test_get_decodes_json_metadata_text_at_repository_boundary() -> None:
    repo = PostgresPHRepository(_Pool(_Conn(_ph_row('{"display_name": "Bob"}'))))

    ph = await repo.get("ph-1")

    assert ph is not None
    assert ph.metadata == {"display_name": "Bob"}


@pytest.mark.asyncio
async def test_get_rejects_non_object_metadata() -> None:
    repo = PostgresPHRepository(_Pool(_Conn(_ph_row('["not", "an", "object"]'))))

    with pytest.raises(TypeError, match=r"person_hypotheses\.metadata"):
        await repo.get("ph-1")


@pytest.mark.asyncio
async def test_save_casts_metadata_parameter_to_jsonb() -> None:
    conn = _Conn()
    repo = PostgresPHRepository(_Pool(conn))

    await repo.save(_domain_ph(metadata={"display_name": "Bob"}))

    assert "$14::jsonb" in conn.execute_sql
    assert conn.execute_args[13] == '{"display_name": "Bob"}'


@pytest.mark.asyncio
async def test_save_rejects_non_mapping_metadata() -> None:
    repo = PostgresPHRepository(_Pool(_Conn()))

    with pytest.raises(TypeError, match=r"PersonHypothesis\.metadata"):
        await repo.save(_domain_ph(metadata="{}"))

"""Postgres implementation of SettingsRepository.

The orchestrator's ``cameras`` and ``streams`` tables exist primarily to
anchor foreign keys from tracking data (events, detections, tracklets,
keyframes, trajectories). The authoritative camera registry lives in
cognitive-companion; the orchestrator only needs a row to exist so its
FK-constrained inserts succeed.

Uses asyncpg with ``$N`` positional placeholders.
"""

from __future__ import annotations

import json

import asyncpg
from structlog import get_logger

from ...domain import CameraConfig, StreamConfig
from ..base import SettingsRepository

logger = get_logger(__name__)


_SQL_UPSERT_CAMERA = """
INSERT INTO continuous_tracking.cameras
    (camera_id, name, rtsp_url, location, floor_plan, is_active)
VALUES ($1, $2, $3, $4, $5::jsonb, $6)
ON CONFLICT (camera_id) DO UPDATE SET
    name       = EXCLUDED.name,
    rtsp_url   = EXCLUDED.rtsp_url,
    location   = EXCLUDED.location,
    floor_plan = EXCLUDED.floor_plan,
    is_active  = EXCLUDED.is_active,
    updated_at = now()
"""

_SQL_GET_CAMERA = """
SELECT camera_id, name, rtsp_url, location, floor_plan, is_active
FROM continuous_tracking.cameras
WHERE camera_id = $1
"""

_SQL_LIST_CAMERAS = """
SELECT camera_id, name, rtsp_url, location, floor_plan, is_active
FROM continuous_tracking.cameras
ORDER BY camera_id
"""

_SQL_UPSERT_STREAM = """
INSERT INTO continuous_tracking.streams
    (stream_id, camera_id, frame_rate, resolution_width, resolution_height, is_active)
VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT (stream_id) DO UPDATE SET
    camera_id         = EXCLUDED.camera_id,
    frame_rate        = EXCLUDED.frame_rate,
    resolution_width  = EXCLUDED.resolution_width,
    resolution_height = EXCLUDED.resolution_height,
    is_active         = EXCLUDED.is_active,
    updated_at        = now()
"""

_SQL_GET_STREAM = """
SELECT stream_id, camera_id, frame_rate, resolution_width, resolution_height, is_active
FROM continuous_tracking.streams
WHERE stream_id = $1
"""

_SQL_LIST_STREAMS = """
SELECT stream_id, camera_id, frame_rate, resolution_width, resolution_height, is_active
FROM continuous_tracking.streams
ORDER BY stream_id
"""


class PostgresSettingsRepository(SettingsRepository):
    """Postgres-backed camera and stream configuration store."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get_camera_config(self, camera_id: str) -> CameraConfig | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(_SQL_GET_CAMERA, camera_id)
        if row is None:
            return None
        return _row_to_camera(row)

    async def save_camera_config(self, config: CameraConfig) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                _SQL_UPSERT_CAMERA,
                config.camera_id,
                config.name,
                config.rtsp_url,
                config.location,
                json.dumps(config.floor_plan),
                config.is_active,
            )

    async def list_camera_configs(self) -> list[CameraConfig]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_SQL_LIST_CAMERAS)
        return [_row_to_camera(row) for row in rows]

    async def get_stream_config(self, stream_id: str) -> StreamConfig | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(_SQL_GET_STREAM, stream_id)
        if row is None:
            return None
        return _row_to_stream(row)

    async def save_stream_config(self, config: StreamConfig) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                _SQL_UPSERT_STREAM,
                config.stream_id,
                config.camera_id,
                config.frame_rate,
                config.resolution_width,
                config.resolution_height,
                config.is_active,
            )

    async def list_stream_configs(self) -> list[StreamConfig]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_SQL_LIST_STREAMS)
        return [_row_to_stream(row) for row in rows]


def _row_to_camera(row: asyncpg.Record) -> CameraConfig:
    floor_plan_raw = row["floor_plan"]
    floor_plan = json.loads(floor_plan_raw) if isinstance(floor_plan_raw, str) else floor_plan_raw
    return CameraConfig(
        camera_id=row["camera_id"],
        name=row["name"] or "",
        rtsp_url=row["rtsp_url"] or "",
        location=row["location"] or "",
        floor_plan=floor_plan or {},
        is_active=row["is_active"],
    )


def _row_to_stream(row: asyncpg.Record) -> StreamConfig:
    return StreamConfig(
        stream_id=row["stream_id"],
        camera_id=row["camera_id"],
        frame_rate=row["frame_rate"],
        resolution_width=row["resolution_width"],
        resolution_height=row["resolution_height"],
        is_active=row["is_active"],
    )

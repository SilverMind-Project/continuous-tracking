"""Postgres-backed CameraTopologyRepository and CoPresenceRepository."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from structlog import get_logger

if TYPE_CHECKING:
    import asyncpg

from ...domain import CameraTopologyEdge, CoPresenceLink

logger = get_logger(__name__)


class PostgresCameraTopologyRepository:
    """Postgres-backed store for camera adjacency topology edges."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @staticmethod
    def _row_to_edge(row: asyncpg.Record) -> CameraTopologyEdge:
        return CameraTopologyEdge(
            from_camera=row["from_camera"],
            to_camera=row["to_camera"],
            observation_count=row["observation_count"],
            mean_transit_s=row["mean_transit_s"],
            variance_transit_s2=row["variance_transit_s2"],
            last_updated_at=row["last_updated_at"],
        )

    async def get_edge(self, from_camera: str, to_camera: str) -> CameraTopologyEdge | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT from_camera, to_camera, observation_count, mean_transit_s, "
                "variance_transit_s2, last_updated_at "
                "FROM continuous_tracking.camera_topology_edges "
                "WHERE from_camera = $1 AND to_camera = $2",
                from_camera,
                to_camera,
            )
        if row is None:
            return None
        return self._row_to_edge(row)

    async def upsert_edge(self, edge: CameraTopologyEdge) -> None:
        now = edge.last_updated_at or datetime.now(UTC)
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO continuous_tracking.camera_topology_edges "
                "(from_camera, to_camera, observation_count, mean_transit_s, "
                " variance_transit_s2, last_updated_at) "
                "VALUES ($1, $2, $3, $4, $5, $6) "
                "ON CONFLICT (from_camera, to_camera) DO UPDATE SET "
                "observation_count = EXCLUDED.observation_count, "
                "mean_transit_s = EXCLUDED.mean_transit_s, "
                "variance_transit_s2 = EXCLUDED.variance_transit_s2, "
                "last_updated_at = EXCLUDED.last_updated_at",
                edge.from_camera,
                edge.to_camera,
                edge.observation_count,
                edge.mean_transit_s,
                edge.variance_transit_s2,
                now,
            )

    async def list_edges(self) -> list[CameraTopologyEdge]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT from_camera, to_camera, observation_count, mean_transit_s, "
                "variance_transit_s2, last_updated_at "
                "FROM continuous_tracking.camera_topology_edges "
                "ORDER BY from_camera, to_camera"
            )
        return [self._row_to_edge(row) for row in rows]

    async def list_edges_from(self, from_camera: str) -> list[CameraTopologyEdge]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT from_camera, to_camera, observation_count, mean_transit_s, "
                "variance_transit_s2, last_updated_at "
                "FROM continuous_tracking.camera_topology_edges "
                "WHERE from_camera = $1 ORDER BY to_camera",
                from_camera,
            )
        return [self._row_to_edge(row) for row in rows]


class PostgresCoPresenceRepository:
    """Postgres-backed store for identity-level co-presence links."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @staticmethod
    def _row_to_link(row: asyncpg.Record) -> CoPresenceLink:
        return CoPresenceLink(
            id=str(row["id"]),
            group_id=row["group_id"],
            ph_id_a=str(row["ph_id_a"]),
            ph_id_b=str(row["ph_id_b"]),
            identity_id=row["identity_id"],
            first_observed_at=row["first_observed_at"],
            last_observed_at=row["last_observed_at"],
            observation_count=row["observation_count"],
        )

    async def upsert_link(self, link: CoPresenceLink) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO continuous_tracking.co_presence_links "
                "(id, group_id, ph_id_a, ph_id_b, identity_id, first_observed_at, "
                " last_observed_at, observation_count) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8) "
                "ON CONFLICT (ph_id_a, ph_id_b) DO UPDATE SET "
                "last_observed_at = EXCLUDED.last_observed_at, "
                "observation_count = continuous_tracking.co_presence_links.observation_count + 1, "
                "identity_id = EXCLUDED.identity_id",
                link.id,
                link.group_id,
                link.ph_id_a,
                link.ph_id_b,
                link.identity_id,
                link.first_observed_at,
                link.last_observed_at,
                link.observation_count,
            )

    async def list_by_group(self, group_id: str) -> list[CoPresenceLink]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, group_id, ph_id_a, ph_id_b, identity_id, "
                "first_observed_at, last_observed_at, observation_count "
                "FROM continuous_tracking.co_presence_links "
                "WHERE group_id = $1 ORDER BY last_observed_at DESC",
                group_id,
            )
        return [self._row_to_link(row) for row in rows]

    async def list_by_identity(self, identity_id: str) -> list[CoPresenceLink]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, group_id, ph_id_a, ph_id_b, identity_id, "
                "first_observed_at, last_observed_at, observation_count "
                "FROM continuous_tracking.co_presence_links "
                "WHERE identity_id = $1 ORDER BY last_observed_at DESC",
                identity_id,
            )
        return [self._row_to_link(row) for row in rows]

    async def list_by_ph(self, ph_id: str) -> list[CoPresenceLink]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, group_id, ph_id_a, ph_id_b, identity_id, "
                "first_observed_at, last_observed_at, observation_count "
                "FROM continuous_tracking.co_presence_links "
                "WHERE ph_id_a = $1 OR ph_id_b = $1 "
                "ORDER BY last_observed_at DESC",
                ph_id,
            )
        return [self._row_to_link(row) for row in rows]

    async def get_active_link(self, ph_id_a: str, ph_id_b: str) -> CoPresenceLink | None:
        aid, bid = sorted([ph_id_a, ph_id_b])
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, group_id, ph_id_a, ph_id_b, identity_id, "
                "first_observed_at, last_observed_at, observation_count "
                "FROM continuous_tracking.co_presence_links "
                "WHERE ph_id_a = $1 AND ph_id_b = $2",
                aid,
                bid,
            )
        if row is None:
            return None
        return self._row_to_link(row)

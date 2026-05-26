"""Person Hypothesis and World Observation persistence.

Protocol + InMemory + Postgres triplet following the project pattern.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from ...domain import PersonHypothesis, WorldObservation


class PHRepositoryProtocol(ABC):
    """Persist Person Hypotheses."""

    @abstractmethod
    async def save(self, ph: PersonHypothesis) -> None:
        """Insert or update a Person Hypothesis."""

    @abstractmethod
    async def get(self, ph_id: str) -> PersonHypothesis | None:
        """Retrieve a PH by ID."""

    @abstractmethod
    async def list_open(self) -> list[PersonHypothesis]:
        """List all PHs that are not closed."""

    @abstractmethod
    async def list_closed_since(self, since: datetime, limit: int = 100) -> list[PersonHypothesis]:
        """List recently closed PHs for continuation candidate matching."""

    @abstractmethod
    async def update_identity(
        self,
        ph_id: str,
        identity_id: str | None,
        committed_at: datetime,
    ) -> None:
        """Update the current identity assignment for a PH."""


class WorldObservationRepository(ABC):
    """Persist individual world observations linked to a PH."""

    @abstractmethod
    async def save(self, observation: WorldObservation, ph_id: str) -> None:
        """Store an observation linked to a PH."""

    @abstractmethod
    async def list_by_ph(self, ph_id: str, limit: int = 50) -> list[WorldObservation]:
        """List recent observations for a PH."""


class PHObservationRepository(ABC):
    """Combined repository for PH lifecycle + observation persistence."""

    @abstractmethod
    async def save_ph(self, ph: PersonHypothesis) -> None:
        """Insert or update a PH."""

    @abstractmethod
    async def save_observation(self, obs: WorldObservation, ph_id: str) -> None:
        """Store an observation."""

    @abstractmethod
    async def list_open_phs(self) -> list[PersonHypothesis]:
        """List all open PHs."""

    @abstractmethod
    async def list_closed_phs_since(
        self, since: datetime, limit: int = 100
    ) -> list[PersonHypothesis]:
        """List recently closed PHs."""

    @abstractmethod
    async def update_ph_identity(
        self,
        ph_id: str,
        identity_id: str | None,
        committed_at: datetime,
    ) -> None:
        """Update identity on a PH."""


# ---------------------------------------------------------------------------
# In-memory implementations
# ---------------------------------------------------------------------------


class InMemoryPHRepository(PHRepositoryProtocol):
    """In-memory store for Person Hypotheses."""

    def __init__(self) -> None:
        self._phs: dict[str, PersonHypothesis] = {}

    async def save(self, ph: PersonHypothesis) -> None:
        self._phs[ph.ph_id] = ph

    async def get(self, ph_id: str) -> PersonHypothesis | None:
        return self._phs.get(ph_id)

    async def list_open(self) -> list[PersonHypothesis]:
        return [ph for ph in self._phs.values() if ph.closed_at is None]

    async def list_closed_since(self, since: datetime, limit: int = 100) -> list[PersonHypothesis]:
        closed = [
            ph for ph in self._phs.values() if ph.closed_at is not None and ph.closed_at >= since
        ]
        closed.sort(key=lambda ph: ph.closed_at, reverse=True)  # type: ignore[arg-type,return-value]
        return closed[:limit]

    async def update_identity(
        self,
        ph_id: str,
        identity_id: str | None,
        committed_at: datetime,
    ) -> None:
        ph = self._phs.get(ph_id)
        if ph is not None:
            self._phs[ph_id] = PersonHypothesis(
                ph_id=ph.ph_id,
                state_mean=ph.state_mean,
                state_cov=ph.state_cov,
                born_at=ph.born_at,
                last_seen_at=ph.last_seen_at,
                last_seen_camera=ph.last_seen_camera,
                observation_count=ph.observation_count,
                current_identity_id=identity_id,
                current_identity_committed_at=committed_at,
                gallery_mean=ph.gallery_mean,
                height_estimate_m=ph.height_estimate_m,
                active_cameras=ph.active_cameras,
                closed_at=ph.closed_at,
                last_floor_speed_m_s=ph.last_floor_speed_m_s,
                last_posture=ph.last_posture,
                metadata=ph.metadata,
            )


class InMemoryWorldObservationRepository(WorldObservationRepository):
    """In-memory store for world observations."""

    def __init__(self) -> None:
        self._observations: dict[str, list[WorldObservation]] = {}

    async def save(self, observation: WorldObservation, ph_id: str) -> None:
        self._observations.setdefault(ph_id, []).append(observation)

    async def list_by_ph(self, ph_id: str, limit: int = 50) -> list[WorldObservation]:
        obs_list = self._observations.get(ph_id, [])
        return obs_list[-limit:]


# ---------------------------------------------------------------------------
# Postgres implementations
# ---------------------------------------------------------------------------


class PostgresPHRepository(PHRepositoryProtocol):
    """Postgres-backed Person Hypothesis repository."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def save(self, ph: PersonHypothesis) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO continuous_tracking.person_hypotheses AS ph (
                    ph_id, born_at, closed_at, last_seen_at, last_seen_camera,
                    observation_count, current_identity_id,
                    current_identity_committed_at,
                    state_mean, state_cov, gallery_mean, height_m,
                    active_cameras, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                ON CONFLICT (ph_id) DO UPDATE SET
                    closed_at = COALESCE(EXCLUDED.closed_at, ph.closed_at),
                    last_seen_at = GREATEST(EXCLUDED.last_seen_at, ph.last_seen_at),
                    last_seen_camera = EXCLUDED.last_seen_camera,
                    observation_count = EXCLUDED.observation_count,
                    current_identity_id = COALESCE(
                        EXCLUDED.current_identity_id, ph.current_identity_id
                    ),
                    current_identity_committed_at = COALESCE(
                        EXCLUDED.current_identity_committed_at,
                        ph.current_identity_committed_at
                    ),
                    state_mean = EXCLUDED.state_mean,
                    state_cov = EXCLUDED.state_cov,
                    gallery_mean = EXCLUDED.gallery_mean,
                    height_m = EXCLUDED.height_m,
                    active_cameras = EXCLUDED.active_cameras,
                    metadata = EXCLUDED.metadata
                """,
                ph.ph_id,
                ph.born_at,
                ph.closed_at,
                ph.last_seen_at,
                ph.last_seen_camera,
                ph.observation_count,
                ph.current_identity_id,
                ph.current_identity_committed_at,
                list(ph.state_mean),
                list(ph.state_cov),
                ph.gallery_mean,
                ph.height_estimate_m,
                list(ph.active_cameras),
                ph.metadata,
            )

    async def get(self, ph_id: str) -> PersonHypothesis | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM continuous_tracking.person_hypotheses WHERE ph_id = $1",
                ph_id,
            )
        return _row_to_ph(row) if row else None

    async def list_open(self) -> list[PersonHypothesis]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM continuous_tracking.person_hypotheses WHERE closed_at IS NULL "
                "ORDER BY last_seen_at DESC"
            )
        return [_row_to_ph(row) for row in rows]

    async def list_closed_since(self, since: datetime, limit: int = 100) -> list[PersonHypothesis]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM continuous_tracking.person_hypotheses "
                "WHERE closed_at IS NOT NULL AND closed_at >= $1 "
                "ORDER BY closed_at DESC LIMIT $2",
                since,
                limit,
            )
        return [_row_to_ph(row) for row in rows]

    async def update_identity(
        self,
        ph_id: str,
        identity_id: str | None,
        committed_at: datetime,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE continuous_tracking.person_hypotheses "
                "SET current_identity_id = $2, current_identity_committed_at = $3 "
                "WHERE ph_id = $1",
                ph_id,
                identity_id,
                committed_at,
            )


class PostgresWorldObservationRepository(WorldObservationRepository):
    """Postgres-backed World Observation repository."""

    def __init__(self, pool: Any) -> None:
        self._pool: Any = pool

    async def save(self, observation: WorldObservation, ph_id: str) -> None:
        import json

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO continuous_tracking.world_observations (
                    observation_id, ph_id, camera_id, frame_index, captured_at,
                    floor_x_m, floor_y_m, detection_confidence, bbox, height_m, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                str(uuid.uuid4()),
                ph_id,
                observation.camera_id,
                observation.frame_index,
                observation.captured_at,
                observation.floor_point.x_mm / 1000.0,
                observation.floor_point.y_mm / 1000.0,
                observation.detection_confidence,
                json.dumps(
                    {
                        "x_min": observation.bbox.x_min,
                        "y_min": observation.bbox.y_min,
                        "x_max": observation.bbox.x_max,
                        "y_max": observation.bbox.y_max,
                    }
                ),
                observation.height_estimate_m,
                {},
            )

    async def list_by_ph(self, ph_id: str, limit: int = 50) -> list[WorldObservation]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM continuous_tracking.world_observations "
                "WHERE ph_id = $1 ORDER BY captured_at DESC LIMIT $2",
                ph_id,
                limit,
            )
        return [_row_to_world_observation(row) for row in rows]


def _row_to_ph(row: Any) -> PersonHypothesis:
    mean_raw: list[float] = [float(v) for v in row["state_mean"]]
    cov_raw: list[float] = [float(v) for v in row["state_cov"]]
    return PersonHypothesis(
        ph_id=str(row["ph_id"]),
        state_mean=(mean_raw[0], mean_raw[1], mean_raw[2], mean_raw[3]),
        state_cov=tuple(cov_raw),
        born_at=row["born_at"],
        last_seen_at=row["last_seen_at"],
        last_seen_camera=str(row["last_seen_camera"] or ""),
        observation_count=int(row["observation_count"]),
        current_identity_id=str(row["current_identity_id"]) if row["current_identity_id"] else None,
        current_identity_committed_at=row.get("current_identity_committed_at"),
        gallery_mean=row["gallery_mean"],
        height_estimate_m=row.get("height_m"),
        active_cameras=frozenset(row["active_cameras"] or []),
        closed_at=row.get("closed_at"),
        last_floor_speed_m_s=0.0,
        last_posture=None,
        metadata=row.get("metadata") or {},
    )


def _row_to_world_observation(row: Any) -> WorldObservation:
    import json

    from ...domain import BoundingBox, FloorPoint

    bbox_raw = row["bbox"]
    if isinstance(bbox_raw, str):
        bbox_raw = json.loads(bbox_raw)

    return WorldObservation(
        camera_id=row["camera_id"],
        frame_index=row["frame_index"],
        captured_at=row["captured_at"],
        floor_point=FloorPoint(
            x_mm=int(row["floor_x_m"] * 1000),
            y_mm=int(row["floor_y_m"] * 1000),
            calibrated=True,
        ),
        bbox=BoundingBox(
            x_min=bbox_raw["x_min"],
            y_min=bbox_raw["y_min"],
            x_max=bbox_raw["x_max"],
            y_max=bbox_raw["y_max"],
        ),
        embedding=[],
        detection_confidence=float(row["detection_confidence"]),
        height_estimate_m=row["height_m"],
        face_anchor=None,
    )

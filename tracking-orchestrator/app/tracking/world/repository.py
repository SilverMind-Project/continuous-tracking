"""Person Hypothesis and World Observation persistence.

Protocol + InMemory + Postgres triplet following the project pattern.
N1: extended with rich query, correction, merge, split, and audit methods.
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Literal

from structlog import get_logger

from ...domain import (
    IdentityEvidence,
    IdentityRevision,
    PersonHypothesis,
    WorldObservation,
)

logger = get_logger(__name__)

RevisionKind = Literal["auto", "manual_correct", "manual_merge", "manual_split"]


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class PHRepositoryProtocol(ABC):
    """Persist Person Hypotheses and their observations."""

    # -- existing WorldTracker methods (kept for backward compat) --

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
        self, ph_id: str, identity_id: str | None, committed_at: datetime
    ) -> None:
        """Update the current identity assignment for a PH."""

    # -- N1 rich query methods --

    @abstractmethod
    async def get_by_id(self, ph_id: str) -> PersonHypothesis | None: ...

    @abstractmethod
    async def list_active(
        self,
        *,
        since: datetime | None = None,
        room_id: str | None = None,
        identity_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[PersonHypothesis], int]: ...

    @abstractmethod
    async def list_history(
        self,
        *,
        since: datetime,
        until: datetime,
        identity_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[PersonHypothesis], int]: ...

    @abstractmethod
    async def list_observations(
        self, ph_id: str, *, limit: int = 200
    ) -> list[WorldObservation]: ...

    @abstractmethod
    async def list_observations_by_ph(
        self, ph_id: str, *, limit: int = 200
    ) -> list[WorldObservation]: ...

    # -- N1 correction methods (return IdentityRevision) --

    @abstractmethod
    async def correct_identity(
        self,
        ph_id: str,
        *,
        new_identity_id: str | None,
        reason: str,
        actor: str,
    ) -> IdentityRevision: ...

    @abstractmethod
    async def merge(
        self,
        *,
        source_ph_id: str,
        target_ph_id: str,
        actor: str,
        reason: str,
    ) -> IdentityRevision: ...

    @abstractmethod
    async def split(
        self,
        ph_id: str,
        *,
        at_observation_id: str,
        actor: str,
        reason: str,
    ) -> tuple[str, str]: ...

    @abstractmethod
    async def list_revisions(
        self,
        *,
        ph_id: str | None = None,
        kind: RevisionKind | None = None,
        limit: int = 50,
        before_id: str | None = None,
    ) -> tuple[list[IdentityRevision], bool]: ...


class WorldObservationRepository(ABC):
    """Persist individual world observations linked to a PH."""

    @abstractmethod
    async def save(self, observation: WorldObservation, ph_id: str) -> None:
        """Store an observation linked to a PH."""

    @abstractmethod
    async def list_by_ph(self, ph_id: str, limit: int = 50) -> list[WorldObservation]:
        """List recent observations for a PH."""


# ---------------------------------------------------------------------------
# In-memory implementations
# ---------------------------------------------------------------------------


class InMemoryPHRepository(PHRepositoryProtocol):
    """In-memory store for Person Hypotheses with full N1 surface."""

    def __init__(self) -> None:
        import asyncio

        self._phs: dict[str, PersonHypothesis] = {}
        self._observations: dict[str, list[WorldObservation]] = {}
        self._revisions: list[IdentityRevision] = []
        self._merges: dict[str, str] = {}  # source_ph_id -> target_ph_id
        self._lock = asyncio.Lock()

    # -- existing WorldTracker methods --

    async def save(self, ph: PersonHypothesis) -> None:
        async with self._lock:
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
        self, ph_id: str, identity_id: str | None, committed_at: datetime
    ) -> None:
        async with self._lock:
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

    # -- N1 rich query methods --

    async def get_by_id(self, ph_id: str) -> PersonHypothesis | None:
        return self._phs.get(ph_id)

    async def list_active(
        self,
        *,
        since: datetime | None = None,
        room_id: str | None = None,
        identity_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[PersonHypothesis], int]:
        results = [
            ph
            for ph in self._phs.values()
            if ph.closed_at is None
            and (since is None or ph.last_seen_at >= since)
            and (identity_id is None or ph.current_identity_id == identity_id)
        ]
        results.sort(key=lambda ph: ph.last_seen_at, reverse=True)
        total = len(results)
        return results[offset : offset + limit], total

    async def list_history(
        self,
        *,
        since: datetime,
        until: datetime,
        identity_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[PersonHypothesis], int]:
        results = [
            ph
            for ph in self._phs.values()
            if since <= ph.last_seen_at <= until
            and (identity_id is None or ph.current_identity_id == identity_id)
        ]
        results.sort(key=lambda ph: ph.last_seen_at, reverse=True)
        total = len(results)
        return results[offset : offset + limit], total

    async def list_observations(self, ph_id: str, *, limit: int = 200) -> list[WorldObservation]:
        obs_list = self._observations.get(ph_id, [])
        return obs_list[-limit:]

    async def list_observations_by_ph(
        self, ph_id: str, *, limit: int = 200
    ) -> list[WorldObservation]:
        return await self.list_observations(ph_id, limit=limit)

    # -- N1 correction methods --

    async def correct_identity(
        self,
        ph_id: str,
        *,
        new_identity_id: str | None,
        reason: str,
        actor: str,
    ) -> IdentityRevision:
        from datetime import UTC

        async with self._lock:
            ph = self._phs.get(ph_id)
            if ph is None:
                raise ValueError(f"PH not found: {ph_id}")
            previous = ph.current_identity_id
            now = datetime.now(UTC)
            self._phs[ph_id] = PersonHypothesis(
                ph_id=ph.ph_id,
                state_mean=ph.state_mean,
                state_cov=ph.state_cov,
                born_at=ph.born_at,
                last_seen_at=ph.last_seen_at,
                last_seen_camera=ph.last_seen_camera,
                observation_count=ph.observation_count,
                current_identity_id=new_identity_id,
                current_identity_committed_at=now,
                gallery_mean=ph.gallery_mean,
                height_estimate_m=ph.height_estimate_m,
                active_cameras=ph.active_cameras,
                closed_at=ph.closed_at,
                last_floor_speed_m_s=ph.last_floor_speed_m_s,
                last_posture=ph.last_posture,
                metadata=ph.metadata,
            )
        revision = IdentityRevision(
            revision_id=str(uuid.uuid4()),
            ph_id=ph_id,
            previous_identity_id=previous,
            new_identity_id=new_identity_id,
            actor=actor,
            reason=reason,
            applied_at=now,
            rewritten_rows=1,
            evidence=None,
        )
        self._revisions.append(revision)
        return revision

    async def merge(
        self,
        *,
        source_ph_id: str,
        target_ph_id: str,
        actor: str,
        reason: str,
    ) -> IdentityRevision:
        from datetime import UTC

        async with self._lock:
            source = self._phs.get(source_ph_id)
            target = self._phs.get(target_ph_id)
            if source is None or target is None:
                raise ValueError("Source or target PH not found")
            # Move observations to target
            src_obs = self._observations.pop(source_ph_id, [])
            self._observations.setdefault(target_ph_id, []).extend(src_obs)
            # Mark source as ended
            now = datetime.now(UTC)
            self._phs[source_ph_id] = PersonHypothesis(
                ph_id=source.ph_id,
                state_mean=source.state_mean,
                state_cov=source.state_cov,
                born_at=source.born_at,
                last_seen_at=source.last_seen_at,
                last_seen_camera=source.last_seen_camera,
                observation_count=source.observation_count,
                current_identity_id=source.current_identity_id,
                current_identity_committed_at=source.current_identity_committed_at,
                gallery_mean=source.gallery_mean,
                height_estimate_m=source.height_estimate_m,
                active_cameras=source.active_cameras,
                closed_at=now,
                last_floor_speed_m_s=source.last_floor_speed_m_s,
                last_posture=source.last_posture,
                metadata={**source.metadata, "merged_into_ph_id": target_ph_id},
            )
            self._merges[source_ph_id] = target_ph_id
        revision = IdentityRevision(
            revision_id=str(uuid.uuid4()),
            ph_id=source_ph_id,
            previous_identity_id=source.current_identity_id,
            new_identity_id=target.current_identity_id,
            actor=actor,
            reason=reason,
            applied_at=now,
            rewritten_rows=len(src_obs),
            evidence=None,
        )
        self._revisions.append(revision)
        return revision

    async def split(
        self,
        ph_id: str,
        *,
        at_observation_id: str,
        actor: str,
        reason: str,
    ) -> tuple[str, str]:
        from datetime import UTC

        async with self._lock:
            ph = self._phs.get(ph_id)
            if ph is None:
                raise ValueError(f"PH not found: {ph_id}")
            obs_list = self._observations.get(ph_id, [])
            split_idx = None
            for i, obs in enumerate(obs_list):
                if getattr(obs, "observation_id", None) == at_observation_id:
                    split_idx = i
                    break
            if split_idx is None:
                raise ValueError(f"Observation {at_observation_id} not found")
            if split_idx == 0:
                raise ValueError("Cannot split at first observation")
            earlier_obs = obs_list[:split_idx]
            later_obs = obs_list[split_idx:]
            new_ph_id = str(uuid.uuid4())
            now = datetime.now(UTC)
            # Close original PH; create new PH for later half
            self._phs[ph_id] = PersonHypothesis(
                ph_id=ph.ph_id,
                state_mean=ph.state_mean,
                state_cov=ph.state_cov,
                born_at=ph.born_at,
                last_seen_at=earlier_obs[-1].captured_at if earlier_obs else ph.last_seen_at,
                last_seen_camera=ph.last_seen_camera,
                observation_count=len(earlier_obs),
                current_identity_id=ph.current_identity_id,
                current_identity_committed_at=ph.current_identity_committed_at,
                gallery_mean=ph.gallery_mean,
                height_estimate_m=ph.height_estimate_m,
                active_cameras=ph.active_cameras,
                closed_at=now,
                last_floor_speed_m_s=ph.last_floor_speed_m_s,
                last_posture=ph.last_posture,
                metadata=ph.metadata,
            )
            later_ph = PersonHypothesis(
                ph_id=new_ph_id,
                state_mean=ph.state_mean,
                state_cov=ph.state_cov,
                born_at=later_obs[0].captured_at,
                last_seen_at=later_obs[-1].captured_at,
                last_seen_camera=ph.last_seen_camera,
                observation_count=len(later_obs),
                current_identity_id=ph.current_identity_id,
                current_identity_committed_at=ph.current_identity_committed_at,
                gallery_mean=ph.gallery_mean,
                height_estimate_m=ph.height_estimate_m,
                active_cameras=ph.active_cameras,
                closed_at=None,
                last_floor_speed_m_s=ph.last_floor_speed_m_s,
                last_posture=ph.last_posture,
                metadata=ph.metadata,
            )
            self._phs[new_ph_id] = later_ph
            self._observations[ph_id] = earlier_obs
            self._observations[new_ph_id] = later_obs
        revision = IdentityRevision(
            revision_id=str(uuid.uuid4()),
            ph_id=ph_id,
            previous_identity_id=ph.current_identity_id,
            new_identity_id=ph.current_identity_id,
            actor=actor,
            reason=reason,
            applied_at=now,
            rewritten_rows=len(later_obs),
            evidence=None,
        )
        self._revisions.append(revision)
        return ph_id, new_ph_id

    async def list_revisions(
        self,
        *,
        ph_id: str | None = None,
        kind: RevisionKind | None = None,
        limit: int = 50,
        before_id: str | None = None,
    ) -> tuple[list[IdentityRevision], bool]:
        results = self._revisions
        if ph_id is not None:
            results = [r for r in results if r.ph_id == ph_id]
        if kind is not None:
            # kind is encoded in reason for in-memory
            pass
        results.sort(key=lambda r: r.applied_at, reverse=True)
        total = len(results)
        page = results[:limit]
        has_more = total > limit
        return page, has_more


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
    """Postgres-backed Person Hypothesis repository with N1 audit support."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    # -- existing WorldTracker methods --

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
                "SELECT * FROM continuous_tracking.person_hypotheses "
                "WHERE closed_at IS NULL ORDER BY last_seen_at DESC"
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
        self, ph_id: str, identity_id: str | None, committed_at: datetime
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

    # -- N1 rich query methods --

    async def get_by_id(self, ph_id: str) -> PersonHypothesis | None:
        return await self.get(ph_id)

    async def list_active(
        self,
        *,
        since: datetime | None = None,
        room_id: str | None = None,
        identity_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[PersonHypothesis], int]:
        clauses = ["closed_at IS NULL"]
        params: list[Any] = []
        idx = 1

        if since is not None:
            clauses.append(f"last_seen_at >= ${idx}")
            params.append(since)
            idx += 1
        if identity_id is not None:
            clauses.append(f"current_identity_id = ${idx}")
            params.append(identity_id)
            idx += 1

        where = " AND ".join(clauses)
        async with self._pool.acquire() as conn:
            count_row = await conn.fetchrow(
                f"SELECT COUNT(*) FROM continuous_tracking.person_hypotheses WHERE {where}",
                *params,
            )
            total = count_row[0] if count_row else 0

            params.extend([limit, offset])
            rows = await conn.fetch(
                f"SELECT * FROM continuous_tracking.person_hypotheses "
                f"WHERE {where} ORDER BY last_seen_at DESC LIMIT ${idx} OFFSET ${idx + 1}",
                *params,
            )
        return [_row_to_ph(row) for row in rows], total

    async def list_history(
        self,
        *,
        since: datetime,
        until: datetime,
        identity_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[PersonHypothesis], int]:
        clauses = ["last_seen_at >= $1", "last_seen_at <= $2"]
        params: list[Any] = [since, until]
        idx = 3

        if identity_id is not None:
            clauses.append(f"current_identity_id = ${idx}")
            params.append(identity_id)
            idx += 1

        where = " AND ".join(clauses)
        async with self._pool.acquire() as conn:
            count_row = await conn.fetchrow(
                f"SELECT COUNT(*) FROM continuous_tracking.person_hypotheses WHERE {where}",
                *params,
            )
            total = count_row[0] if count_row else 0

            params.extend([limit, offset])
            rows = await conn.fetch(
                f"SELECT * FROM continuous_tracking.person_hypotheses "
                f"WHERE {where} ORDER BY last_seen_at DESC LIMIT ${idx} OFFSET ${idx + 1}",
                *params,
            )
        return [_row_to_ph(row) for row in rows], total

    async def list_observations(self, ph_id: str, *, limit: int = 200) -> list[WorldObservation]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM continuous_tracking.world_observations "
                "WHERE ph_id = $1 ORDER BY captured_at DESC LIMIT $2",
                ph_id,
                limit,
            )
        return [_row_to_world_observation(row) for row in rows]

    async def list_observations_by_ph(
        self, ph_id: str, *, limit: int = 200
    ) -> list[WorldObservation]:
        return await self.list_observations(ph_id, limit=limit)

    # -- N1 correction methods --

    async def correct_identity(
        self,
        ph_id: str,
        *,
        new_identity_id: str | None,
        reason: str,
        actor: str,
    ) -> IdentityRevision:
        from datetime import UTC

        revision_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        async with self._pool.acquire() as conn, conn.transaction():
            ph_row = await conn.fetchrow(
                "SELECT current_identity_id FROM continuous_tracking.person_hypotheses "
                "WHERE ph_id = $1 FOR UPDATE",
                ph_id,
            )
            if ph_row is None:
                raise ValueError(f"PH not found: {ph_id}")
            previous = ph_row["current_identity_id"]

            await conn.execute(
                "UPDATE continuous_tracking.person_hypotheses "
                "SET current_identity_id = $2, current_identity_committed_at = $3 "
                "WHERE ph_id = $1",
                ph_id,
                new_identity_id,
                now,
            )

            await conn.execute(
                """
                    INSERT INTO continuous_tracking.ph_revisions (
                        revision_id, ph_id, previous_identity_id, new_identity_id,
                        actor, reason, kind, applied_at, rewritten_rows
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                revision_id,
                ph_id,
                previous,
                new_identity_id,
                actor,
                reason,
                "manual_correct",
                now,
                1,
            )

        return IdentityRevision(
            revision_id=revision_id,
            ph_id=ph_id,
            previous_identity_id=previous,
            new_identity_id=new_identity_id,
            actor=actor,
            reason=reason,
            applied_at=now,
            rewritten_rows=1,
            evidence=None,
        )

    async def merge(
        self,
        *,
        source_ph_id: str,
        target_ph_id: str,
        actor: str,
        reason: str,
    ) -> IdentityRevision:
        from datetime import UTC

        revision_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        async with self._pool.acquire() as conn, conn.transaction():
            src_row = await conn.fetchrow(
                "SELECT * FROM continuous_tracking.person_hypotheses WHERE ph_id = $1 FOR UPDATE",
                source_ph_id,
            )
            if src_row is None:
                raise ValueError(f"Source PH not found: {source_ph_id}")

            obs_count = (
                await conn.fetchval(
                    "SELECT COUNT(*) FROM continuous_tracking.world_observations WHERE ph_id = $1",
                    source_ph_id,
                )
                or 0
            )

            # Re-key observations to target
            await conn.execute(
                "UPDATE continuous_tracking.world_observations SET ph_id = $2 WHERE ph_id = $1",
                source_ph_id,
                target_ph_id,
            )

            # Mark source as ended
            await conn.execute(
                "UPDATE continuous_tracking.person_hypotheses "
                "SET closed_at = $2, metadata = metadata || $3 "
                "WHERE ph_id = $1",
                source_ph_id,
                now,
                json.dumps({"merged_into_ph_id": target_ph_id}),
            )

            # Record merge
            await conn.execute(
                "INSERT INTO continuous_tracking.ph_merges "
                "(merge_id, source_ph_id, target_ph_id, revision_id, applied_at) "
                "VALUES ($1, $2, $3, $4, $5)",
                str(uuid.uuid4()),
                source_ph_id,
                target_ph_id,
                revision_id,
                now,
            )

            # Record revision
            await conn.execute(
                """
                    INSERT INTO continuous_tracking.ph_revisions (
                        revision_id, ph_id, previous_identity_id, new_identity_id,
                        actor, reason, kind, applied_at, rewritten_rows
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                revision_id,
                source_ph_id,
                src_row["current_identity_id"],
                src_row["current_identity_id"],
                actor,
                reason,
                "manual_merge",
                now,
                obs_count,
            )

        return IdentityRevision(
            revision_id=revision_id,
            ph_id=source_ph_id,
            previous_identity_id=src_row["current_identity_id"],
            new_identity_id=src_row["current_identity_id"],
            actor=actor,
            reason=reason,
            applied_at=now,
            rewritten_rows=obs_count,
            evidence=None,
        )

    async def split(
        self,
        ph_id: str,
        *,
        at_observation_id: str,
        actor: str,
        reason: str,
    ) -> tuple[str, str]:
        from datetime import UTC

        revision_id = str(uuid.uuid4())
        new_ph_id = str(uuid.uuid4())
        now = datetime.now(UTC)

        async with self._pool.acquire() as conn, conn.transaction():
            ph_row = await conn.fetchrow(
                "SELECT * FROM continuous_tracking.person_hypotheses WHERE ph_id = $1 FOR UPDATE",
                ph_id,
            )
            if ph_row is None:
                raise ValueError(f"PH not found: {ph_id}")

            # Find the split point
            obs_rows = await conn.fetch(
                "SELECT observation_id, captured_at FROM continuous_tracking.world_observations "
                "WHERE ph_id = $1 ORDER BY captured_at ASC",
                ph_id,
            )
            split_idx = None
            for i, row in enumerate(obs_rows):
                if str(row["observation_id"]) == at_observation_id:
                    split_idx = i
                    break
            if split_idx is None:
                raise ValueError(f"Observation not found: {at_observation_id}")
            if split_idx == 0:
                raise ValueError("Cannot split at first observation")

            later_obs_ids = [str(r["observation_id"]) for r in obs_rows[split_idx:]]

            # Update later observations to new PH
            await conn.execute(
                "UPDATE continuous_tracking.world_observations "
                "SET ph_id = $2 WHERE ph_id = $1 AND observation_id = ANY($3)",
                ph_id,
                new_ph_id,
                later_obs_ids,
            )

            # Close original PH
            await conn.execute(
                "UPDATE continuous_tracking.person_hypotheses "
                "SET closed_at = $2, observation_count = $3 "
                "WHERE ph_id = $1",
                ph_id,
                now,
                split_idx,
            )

            # Create new PH
            await conn.execute(
                """
                    INSERT INTO continuous_tracking.person_hypotheses (
                        ph_id, born_at, closed_at, last_seen_at, last_seen_camera,
                        observation_count, current_identity_id,
                        current_identity_committed_at,
                        state_mean, state_cov, gallery_mean, height_m,
                        active_cameras, metadata
                    ) VALUES ($1, $2, NULL, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    """,
                new_ph_id,
                obs_rows[split_idx]["captured_at"],
                obs_rows[-1]["captured_at"],
                ph_row["last_seen_camera"],
                len(later_obs_ids),
                ph_row["current_identity_id"],
                ph_row["current_identity_committed_at"],
                ph_row["state_mean"],
                ph_row["state_cov"],
                ph_row["gallery_mean"],
                ph_row["height_m"],
                ph_row["active_cameras"],
                ph_row["metadata"],
            )

            # Record revision
            await conn.execute(
                """
                    INSERT INTO continuous_tracking.ph_revisions (
                        revision_id, ph_id, previous_identity_id, new_identity_id,
                        actor, reason, kind, applied_at, rewritten_rows
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                revision_id,
                ph_id,
                ph_row["current_identity_id"],
                ph_row["current_identity_id"],
                actor,
                reason,
                "manual_split",
                now,
                len(later_obs_ids),
            )

        return ph_id, new_ph_id

    async def list_revisions(
        self,
        *,
        ph_id: str | None = None,
        kind: RevisionKind | None = None,
        limit: int = 50,
        before_id: str | None = None,
    ) -> tuple[list[IdentityRevision], bool]:
        clauses: list[str] = []
        params: list[Any] = []
        idx = 1

        if ph_id is not None:
            clauses.append(f"ph_id = ${idx}")
            params.append(ph_id)
            idx += 1
        if kind is not None:
            clauses.append(f"kind = ${idx}")
            params.append(kind)
            idx += 1
        if before_id is not None:
            clauses.append(f"revision_id < ${idx}")
            params.append(before_id)
            idx += 1

        where = " AND ".join(clauses) if clauses else "TRUE"

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM continuous_tracking.ph_revisions "
                f"WHERE {where} ORDER BY applied_at DESC LIMIT ${idx}",
                *params,
                limit,
            )
        revisions = [_row_to_revision(row) for row in rows]
        has_more = len(revisions) == limit
        return revisions, has_more


class PostgresWorldObservationRepository(WorldObservationRepository):
    """Postgres-backed World Observation repository."""

    def __init__(self, pool: Any) -> None:
        self._pool: Any = pool

    async def save(self, observation: WorldObservation, ph_id: str) -> None:
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


# ---------------------------------------------------------------------------
# Row mappers
# ---------------------------------------------------------------------------


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


def _row_to_revision(row: Any) -> IdentityRevision:

    evidence_raw = row.get("evidence_jsonb")
    evidence = None
    if evidence_raw:
        if isinstance(evidence_raw, str):
            evidence_raw = json.loads(evidence_raw)
        evidence = IdentityEvidence(
            top_identity_id=evidence_raw.get("top_identity_id"),
            top_probability=float(evidence_raw.get("top_probability", 0.0)),
            second_probability=float(evidence_raw.get("second_probability", 0.0)),
            posterior_entropy=float(evidence_raw.get("posterior_entropy", 0.0)),
            observation_count=int(evidence_raw.get("observation_count", 0)),
        )

    return IdentityRevision(
        revision_id=str(row["revision_id"]),
        ph_id=str(row["ph_id"]),
        previous_identity_id=row.get("previous_identity_id"),
        new_identity_id=row.get("new_identity_id"),
        actor=str(row["actor"]),
        reason=str(row.get("reason", "")),
        applied_at=row["applied_at"],
        rewritten_rows=int(row.get("rewritten_rows", 0)),
        evidence=evidence,
    )

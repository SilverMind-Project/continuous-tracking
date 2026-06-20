"""Postgres-backed Person Hypothesis repository.

Implements PHRepositoryProtocol using asyncpg against the
``continuous_tracking`` schema. Receives only an ``asyncpg.Pool``.
"""

from __future__ import annotations

import json
import struct
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from structlog import get_logger

from ...domain import (
    BoundingBox,
    FloorPoint,
    IdentityEvidence,
    IdentityRevision,
    Keyframe,
    OrientationBin,
    PersonHypothesis,
    ViewPrototype,
    WorldObservation,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# view_prototypes binary serialization (bytea)
# ---------------------------------------------------------------------------

_PROTOTYPE_HEADER_FMT = ">I"  # uint32: number of prototypes
_PROTOTYPE_ENTRY_FMT = ">BI I"  # orientation (uint8), count (uint32), embedding_dim (uint32)
_EMBEDDING_DIM = 768


def _pack_view_prototypes(prototypes: tuple[ViewPrototype, ...]) -> bytes | None:
    """Serialize view_prototypes to bytea for DB storage.

    Binary layout:
        4 bytes: num_prototypes (uint32 BE)
        For each prototype:
            1 byte:  orientation value (0-4)
            4 bytes: count (uint32 BE)
            4 bytes: embedding_dim (uint32 BE, always 768)
            N*4 bytes: float32 embedding values (BE)
    Returns None when prototypes is empty (stored as SQL NULL).
    """
    if not prototypes:
        return None
    parts: list[bytes] = [struct.pack(_PROTOTYPE_HEADER_FMT, len(prototypes))]
    for p in prototypes:
        parts.append(
            struct.pack(
                _PROTOTYPE_ENTRY_FMT,
                int(p.orientation),
                p.count,
                len(p.embedding),
            )
        )
        parts.append(struct.pack(f">{len(p.embedding)}f", *p.embedding))
    return b"".join(parts)


def _unpack_view_prototypes(data: bytes | None) -> tuple[ViewPrototype, ...]:
    """Deserialize view_prototypes from bytea.

    Returns an empty tuple when *data* is None or empty.
    """
    if not data:
        return ()
    try:
        offset = 0
        num_protos = struct.unpack_from(_PROTOTYPE_HEADER_FMT, data, offset)[0]
        offset += struct.calcsize(_PROTOTYPE_HEADER_FMT)
        prototypes: list[ViewPrototype] = []
        for _ in range(num_protos):
            orientation_val, count, emb_dim = struct.unpack_from(_PROTOTYPE_ENTRY_FMT, data, offset)
            offset += struct.calcsize(_PROTOTYPE_ENTRY_FMT)
            emb_vals = struct.unpack_from(f">{emb_dim}f", data, offset)
            offset += emb_dim * struct.calcsize("f")
            prototypes.append(
                ViewPrototype(
                    orientation=OrientationBin(orientation_val),
                    embedding=emb_vals,
                    count=count,
                )
            )
        return tuple(prototypes)
    except (struct.error, ValueError) as exc:
        logger.warning(
            "view_prototypes_deserialize_failed",
            error=str(exc),
            data_len=len(data) if data else 0,
        )
        return ()


# ---------------------------------------------------------------------------
# PostgresPHRepository
# ---------------------------------------------------------------------------


class PostgresPHRepository:
    """Postgres-backed PH repository — implements PHRepositoryProtocol structurally."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    # -- save / get / list_open / list_closed_since / identity operations --

    async def save(self, ph: PersonHypothesis) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO continuous_tracking.person_hypotheses AS ph (
                    ph_id, born_at, closed_at, last_seen_at, last_seen_camera,
                    observation_count, current_identity_id,
                    current_identity_committed_at,
                    last_independent_identity_evidence_at,
                    state_mean, state_cov, gallery_mean, height_m,
                    active_cameras, metadata, mean_quality, view_prototypes
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                    $14, $15::jsonb, $16, $17
                )
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
                    last_independent_identity_evidence_at = COALESCE(
                        EXCLUDED.last_independent_identity_evidence_at,
                        ph.last_independent_identity_evidence_at
                    ),
                    state_mean = EXCLUDED.state_mean,
                    state_cov = EXCLUDED.state_cov,
                    gallery_mean = EXCLUDED.gallery_mean,
                    height_m = EXCLUDED.height_m,
                    active_cameras = EXCLUDED.active_cameras,
                    metadata = EXCLUDED.metadata,
                    mean_quality = EXCLUDED.mean_quality,
                    view_prototypes = EXCLUDED.view_prototypes
                """,
                ph.ph_id,
                ph.born_at,
                ph.closed_at,
                ph.last_seen_at,
                ph.last_seen_camera,
                ph.observation_count,
                ph.current_identity_id,
                ph.current_identity_committed_at,
                ph.last_independent_identity_evidence_at,
                list(ph.state_mean),
                list(ph.state_cov),
                ph.gallery_mean,
                ph.height_estimate_m,
                list(ph.active_cameras),
                json.dumps(
                    _json_object_from_domain(ph.metadata, column="PersonHypothesis.metadata")
                ),
                ph.mean_quality,
                _pack_view_prototypes(ph.view_prototypes),
            )

    async def get(self, ph_id: str) -> PersonHypothesis | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM continuous_tracking.person_hypotheses WHERE ph_id = $1",
                ph_id,
            )
        return _row_to_ph(row) if row else None

    async def get_by_id(self, ph_id: str) -> PersonHypothesis | None:
        return await self.get(ph_id)

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

    async def evidence_backed_commit(
        self, ph_id: str, identity_id: str, evidence_at: datetime, committed_at: datetime
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE continuous_tracking.person_hypotheses "
                "SET current_identity_id = $2, current_identity_committed_at = $3, "
                "last_independent_identity_evidence_at = $4 "
                "WHERE ph_id = $1",
                ph_id,
                identity_id,
                committed_at,
                evidence_at,
            )

    async def prior_only_update(self, ph_id: str, identity_id: str, committed_at: datetime) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE continuous_tracking.person_hypotheses "
                "SET current_identity_id = $2, current_identity_committed_at = $3 "
                "WHERE ph_id = $1",
                ph_id,
                identity_id,
                committed_at,
            )

    async def clear_to_unknown(self, ph_id: str, committed_at: datetime) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE continuous_tracking.person_hypotheses "
                "SET current_identity_id = NULL, current_identity_committed_at = $2 "
                "WHERE ph_id = $1",
                ph_id,
                committed_at,
            )

    # -- list_active (rich filters) --

    async def list_active(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        room_id: str | None = None,
        identity_id: str | None = None,
        state: str | None = None,
        include_transient: bool = False,
        min_duration_s: float | None = None,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[PersonHypothesis], int]:
        predicates: list[str] = []
        params: list[Any] = []
        idx = 1

        if since is not None:
            predicates.append(f"last_seen_at >= ${idx}")
            params.append(since)
            idx += 1
        if until is not None:
            predicates.append(f"born_at <= ${idx}")
            params.append(until)
            idx += 1
        if room_id is not None:
            predicates.append(f"metadata->>'last_room_id' = ${idx}")
            params.append(room_id)
            idx += 1
        if identity_id is not None:
            predicates.append(f"current_identity_id = ${idx}")
            params.append(identity_id)
            idx += 1
        if state in ("active", "coasting"):
            predicates.append("closed_at IS NULL")
        elif state == "ended":
            predicates.append("closed_at IS NOT NULL")
        if not include_transient:
            predicates.append("EXTRACT(EPOCH FROM (COALESCE(closed_at, NOW()) - born_at)) >= 2.0")
            predicates.append("observation_count >= 2")
        if min_duration_s is not None:
            predicates.append(
                f"EXTRACT(EPOCH FROM (COALESCE(closed_at, NOW()) - born_at)) >= ${idx}"
            )
            params.append(min_duration_s)
            idx += 1
        if search is not None:
            predicates.append(f"COALESCE(current_identity_id, '') ILIKE ${idx}")
            params.append(f"%{search}%")
            idx += 1

        where = ("WHERE " + " AND ".join(predicates)) if predicates else ""

        async with self._pool.acquire() as conn:
            total: int = await conn.fetchval(
                f"SELECT COUNT(*) FROM continuous_tracking.person_hypotheses {where}",
                *params,
            )
            rows = await conn.fetch(
                f"SELECT * FROM continuous_tracking.person_hypotheses {where}"
                f" ORDER BY last_seen_at DESC LIMIT ${idx} OFFSET ${idx + 1}",
                *params,
                limit,
                offset,
            )
        return [_row_to_ph(row) for row in rows], total

    # -- list_history --

    async def list_history(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        identity_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[PersonHypothesis], int]:
        predicates: list[str] = []
        params: list[Any] = []
        idx = 1

        if since is not None:
            predicates.append(f"last_seen_at >= ${idx}")
            params.append(since)
            idx += 1
        if until is not None:
            predicates.append(f"last_seen_at <= ${idx}")
            params.append(until)
            idx += 1
        if identity_id is not None:
            predicates.append(f"current_identity_id = ${idx}")
            params.append(identity_id)
            idx += 1

        where = " AND ".join(predicates) if predicates else "TRUE"

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

    # -- observations --

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

    async def get_observations(
        self, ph_id: str, *, limit: int = 50, offset: int = 0
    ) -> tuple[list[WorldObservation], int]:
        async with self._pool.acquire() as conn:
            total: int = await conn.fetchval(
                "SELECT COUNT(*) FROM continuous_tracking.world_observations WHERE ph_id = $1",
                ph_id,
            )
            rows = await conn.fetch(
                "SELECT * FROM continuous_tracking.world_observations "
                "WHERE ph_id = $1 ORDER BY captured_at DESC LIMIT $2 OFFSET $3",
                ph_id,
                limit,
                offset,
            )
        return [_row_to_world_observation(row) for row in rows], total

    # -- trail --

    async def get_trail(self, ph_id: str, *, since: datetime | None = None) -> list[dict[str, Any]]:
        predicates = ["ph_id = $1"]
        params: list[Any] = [ph_id]
        idx = 2

        if since is not None:
            predicates.append(f"captured_at >= ${idx}")
            params.append(since)
            idx += 1

        where = " AND ".join(predicates)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT captured_at, floor_x_m, floor_y_m, camera_id "
                f"FROM continuous_tracking.world_observations "
                f"WHERE {where} ORDER BY captured_at ASC",
                *params,
            )
        return [
            {
                "captured_at": row["captured_at"].isoformat(),
                "floor_x_m": float(row["floor_x_m"]),
                "floor_y_m": float(row["floor_y_m"]),
                "camera_id": str(row["camera_id"]),
            }
            for row in rows
        ]

    # -- co_present --

    async def get_co_present(
        self, ph_id: str, *, at: datetime | None = None, radius_m: float = 5.0
    ) -> list[PersonHypothesis]:
        async with self._pool.acquire() as conn:
            ph = await conn.fetchrow(
                "SELECT last_seen_at, state_mean "
                "FROM continuous_tracking.person_hypotheses WHERE ph_id = $1",
                ph_id,
            )
            if ph is None:
                return []
            ref_time = at if at is not None else ph["last_seen_at"]
            ref_mean = [float(v) for v in ph["state_mean"]]
            rows = await conn.fetch(
                "SELECT * FROM continuous_tracking.person_hypotheses "
                "WHERE ph_id != $1 AND closed_at IS NULL "
                "AND last_seen_at >= $2::timestamptz - INTERVAL '30 seconds' "
                "AND last_seen_at <= $2::timestamptz + INTERVAL '30 seconds' "
                "AND sqrt(power(state_mean[1]::double precision - $3::double precision, 2) "
                "+ power(state_mean[2]::double precision - $4::double precision, 2)) "
                "<= $5::double precision "
                "ORDER BY last_seen_at DESC",
                ph_id,
                ref_time,
                ref_mean[0],
                ref_mean[1],
                radius_m,
            )
        return [_row_to_ph(row) for row in rows]

    # -- keyframes --

    async def get_keyframes(
        self, ph_id: str, *, limit: int = 20, offset: int = 0
    ) -> tuple[list[Keyframe], int]:
        async with self._pool.acquire() as conn:
            total: int = await conn.fetchval(
                "SELECT COUNT(*) FROM continuous_tracking.tagged_keyframes WHERE ph_id = $1::uuid",
                ph_id,
            )
            rows = await conn.fetch(
                "SELECT id, captured_at, camera_id, minio_key, annotations "
                "FROM continuous_tracking.tagged_keyframes "
                "WHERE ph_id = $1::uuid ORDER BY captured_at DESC LIMIT $2 OFFSET $3",
                ph_id,
                limit,
                offset,
            )
        return [
            Keyframe(
                observation_id=str(row["id"]),
                observed_at=row["captured_at"],
                camera_id=str(row["camera_id"]),
                minio_key=str(row["minio_key"]),
                floor_x_mm=_annotation_float(row["annotations"], "floor_x_mm"),
                floor_y_mm=_annotation_float(row["annotations"], "floor_y_mm"),
                pose_class=_annotation_str(row["annotations"], "pose_class"),
                reid_confidence=_annotation_float(row["annotations"], "reid_confidence"),
            )
            for row in rows
        ], total

    # -- correct_identity --

    async def correct_identity(
        self,
        ph_id: str,
        *,
        new_identity_id: str | None,
        reason: str,
        actor: str,
    ) -> IdentityRevision:
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

    # -- merge --

    async def _merge_with_conn(
        self,
        conn: Any,
        *,
        source_ph_id: str,
        target_ph_id: str,
        actor: str,
        reason: str,
        revision_id: str,
        now: datetime,
    ) -> IdentityRevision:
        src_row = await conn.fetchrow(
            "SELECT * FROM continuous_tracking.person_hypotheses WHERE ph_id = $1 FOR UPDATE",
            source_ph_id,
        )
        if src_row is None:
            raise ValueError(f"Source PH not found: {source_ph_id}")

        target_row = await conn.fetchrow(
            "SELECT * FROM continuous_tracking.person_hypotheses WHERE ph_id = $1 FOR UPDATE",
            target_ph_id,
        )
        if target_row is None:
            raise ValueError(f"Target PH not found: {target_ph_id}")

        obs_count = (
            await conn.fetchval(
                "SELECT COUNT(*) FROM continuous_tracking.world_observations WHERE ph_id = $1",
                source_ph_id,
            )
            or 0
        )

        await conn.execute(
            "UPDATE continuous_tracking.world_observations SET ph_id = $2 WHERE ph_id = $1",
            source_ph_id,
            target_ph_id,
        )

        await conn.execute(
            "UPDATE continuous_tracking.person_hypotheses "
            "SET closed_at = $2, metadata = metadata || $3::jsonb "
            "WHERE ph_id = $1",
            source_ph_id,
            now,
            json.dumps({"merged_into_ph_id": target_ph_id}),
        )

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
            target_row["current_identity_id"],
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
            new_identity_id=target_row["current_identity_id"],
            actor=actor,
            reason=reason,
            applied_at=now,
            rewritten_rows=obs_count,
            evidence=None,
        )

    async def merge(
        self,
        *,
        source_ph_id: str,
        target_ph_id: str,
        actor: str,
        reason: str,
        idempotency_key: str | None = None,
    ) -> IdentityRevision:
        _ = idempotency_key
        revision_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        async with self._pool.acquire() as conn, conn.transaction():
            return await self._merge_with_conn(
                conn,
                source_ph_id=source_ph_id,
                target_ph_id=target_ph_id,
                actor=actor,
                reason=reason,
                revision_id=revision_id,
                now=now,
            )

    async def batch_merge(
        self,
        *,
        source_ph_ids: list[str],
        target_ph_id: str,
        actor: str,
        reason: str,
        idempotency_key: str | None = None,
    ) -> list[IdentityRevision]:
        _ = idempotency_key
        if target_ph_id in source_ph_ids:
            raise ValueError("Target PH cannot also be a merge source")
        if len(set(source_ph_ids)) != len(source_ph_ids):
            raise ValueError("Duplicate source PH IDs are not allowed")

        now = datetime.now(UTC)
        async with self._pool.acquire() as conn, conn.transaction():
            revisions: list[IdentityRevision] = []
            for source_ph_id in source_ph_ids:
                revisions.append(
                    await self._merge_with_conn(
                        conn,
                        source_ph_id=source_ph_id,
                        target_ph_id=target_ph_id,
                        actor=actor,
                        reason=reason,
                        revision_id=str(uuid.uuid4()),
                        now=now,
                    )
                )
        return revisions

    # -- split --

    async def split(
        self,
        ph_id: str,
        *,
        at_observation_id: str,
        actor: str,
        reason: str,
        idempotency_key: str | None = None,
    ) -> tuple[str, str]:
        _ = idempotency_key
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

            obs_rows = await conn.fetch(
                "SELECT observation_id, captured_at "
                "FROM continuous_tracking.world_observations "
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

            await conn.execute(
                "UPDATE continuous_tracking.world_observations "
                "SET ph_id = $2 WHERE ph_id = $1 AND observation_id = ANY($3)",
                ph_id,
                new_ph_id,
                later_obs_ids,
            )

            await conn.execute(
                "UPDATE continuous_tracking.person_hypotheses "
                "SET closed_at = $2, observation_count = $3 "
                "WHERE ph_id = $1",
                ph_id,
                now,
                split_idx,
            )

            await conn.execute(
                """
                INSERT INTO continuous_tracking.person_hypotheses (
                    ph_id, born_at, closed_at, last_seen_at, last_seen_camera,
                    observation_count, current_identity_id,
                    current_identity_committed_at,
                    state_mean, state_cov, gallery_mean, height_m,
                    active_cameras, metadata, mean_quality
                ) VALUES ($1, $2, NULL, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
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
                float(ph_row.get("mean_quality") or 0.0),
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
                ph_row["current_identity_id"],
                ph_row["current_identity_id"],
                actor,
                reason,
                "manual_split",
                now,
                len(later_obs_ids),
            )

        return ph_id, new_ph_id

    # -- batch_correct (single transaction) --

    async def batch_correct(
        self,
        ph_ids: list[str],
        new_identity_ids: list[str | None],
        actor: str,
        reasons: list[str],
        idempotency_key: str | None = None,
    ) -> list[IdentityRevision]:
        """Apply identity corrections atomically.

        Looks up current identity for each PH within the transaction,
        applies updates, and returns the resulting revisions.
        """
        _ = idempotency_key
        now = datetime.now(UTC)
        revisions: list[IdentityRevision] = []

        async with self._pool.acquire() as conn, conn.transaction():
            for ph_id, new_identity_id, reason in zip(
                ph_ids, new_identity_ids, reasons, strict=True
            ):
                revision_id = str(uuid.uuid4())
                ph_row = await conn.fetchrow(
                    "SELECT current_identity_id FROM continuous_tracking.person_hypotheses "
                    "WHERE ph_id = $1 FOR UPDATE",
                    ph_id,
                )
                if ph_row is None:
                    raise ValueError(f"PH not found: {ph_id}")
                previous = ph_row["current_identity_id"]

                await conn.execute(
                    """
                    INSERT INTO continuous_tracking.ph_revisions
                        (revision_id, ph_id, previous_identity_id, new_identity_id,
                         actor, reason, applied_at, kind, rewritten_rows)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (revision_id, applied_at) DO NOTHING
                    """,
                    revision_id,
                    ph_id,
                    previous,
                    new_identity_id,
                    actor,
                    reason,
                    now,
                    "manual_correct",
                    1,
                )
                await conn.execute(
                    """
                    UPDATE continuous_tracking.person_hypotheses
                       SET current_identity_id = $1,
                           current_identity_committed_at = $2
                     WHERE ph_id = $3
                    """,
                    new_identity_id,
                    now,
                    ph_id,
                )
                revisions.append(
                    IdentityRevision(
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
                )
        return revisions

    async def delete_many(self, ph_ids: list[str], *, actor: str, reason: str) -> int:
        if not ph_ids:
            return 0
        async with self._pool.acquire() as conn, conn.transaction():
            rows = await conn.fetch(
                """
                WITH target AS (
                    SELECT ph_id
                    FROM continuous_tracking.person_hypotheses
                    WHERE ph_id = ANY($1::uuid[])
                    FOR UPDATE
                ),
                deleted_revisions AS (
                    DELETE FROM continuous_tracking.ph_revisions r
                    USING target
                    WHERE r.ph_id = target.ph_id
                ),
                deleted_merges AS (
                    DELETE FROM continuous_tracking.ph_merges m
                    USING target
                    WHERE m.source_ph_id = target.ph_id OR m.target_ph_id = target.ph_id
                )
                DELETE FROM continuous_tracking.person_hypotheses ph
                USING target
                WHERE ph.ph_id = target.ph_id
                RETURNING ph.ph_id
                """,
                ph_ids,
            )
        logger.info(
            "ph_deleted_many",
            actor=actor,
            reason=reason,
            requested=len(ph_ids),
            deleted=len(rows),
        )
        return len(rows)

    async def purge_unknown_older_than(self, cutoff: datetime, *, limit: int = 1000) -> int:
        async with self._pool.acquire() as conn, conn.transaction():
            rows = await conn.fetch(
                """
                WITH target AS (
                    SELECT ph_id
                    FROM continuous_tracking.person_hypotheses
                    WHERE current_identity_id IS NULL
                      AND closed_at IS NOT NULL
                      AND last_seen_at < $1::timestamptz
                    ORDER BY last_seen_at ASC
                    LIMIT $2
                    FOR UPDATE
                ),
                deleted_revisions AS (
                    DELETE FROM continuous_tracking.ph_revisions r
                    USING target
                    WHERE r.ph_id = target.ph_id
                ),
                deleted_merges AS (
                    DELETE FROM continuous_tracking.ph_merges m
                    USING target
                    WHERE m.source_ph_id = target.ph_id OR m.target_ph_id = target.ph_id
                )
                DELETE FROM continuous_tracking.person_hypotheses ph
                USING target
                WHERE ph.ph_id = target.ph_id
                RETURNING ph.ph_id
                """,
                cutoff,
                limit,
            )
        if rows:
            logger.info("ph_unknown_purged", cutoff=cutoff.isoformat(), deleted=len(rows))
        return len(rows)

    # -- list_revisions --

    async def list_revisions(
        self,
        *,
        ph_id: str | None = None,
        kind: str | None = None,
        limit: int = 50,
        before_id: str | None = None,
    ) -> tuple[list[IdentityRevision], bool]:
        predicates: list[str] = []
        params: list[Any] = []
        idx = 1

        if ph_id is not None:
            predicates.append(f"ph_id = ${idx}")
            params.append(ph_id)
            idx += 1
        if kind is not None:
            predicates.append(f"kind = ${idx}")
            params.append(kind)
            idx += 1
        if before_id is not None:
            predicates.append(f"revision_id < ${idx}")
            params.append(before_id)
            idx += 1

        where = " AND ".join(predicates) if predicates else "TRUE"

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


# ---------------------------------------------------------------------------
# PostgresWorldObservationRepository
# ---------------------------------------------------------------------------


class PostgresWorldObservationRepository:
    """Postgres-backed World Observation repository."""

    def __init__(self, pool: Any) -> None:
        self._pool: Any = pool

    async def save(self, observation: WorldObservation, ph_id: str) -> str:
        oid = str(uuid.uuid4())
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO continuous_tracking.world_observations (
                    observation_id, ph_id, camera_id, frame_index, captured_at,
                    floor_x_m, floor_y_m, detection_confidence, bbox, height_m, quality, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (observation_id, captured_at) DO NOTHING
                """,
                oid,
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
                observation.quality,
                json.dumps(
                    {
                        "floor_residual_m": observation.floor_residual_m,
                        "floor_cov_random": observation.floor_cov_random,
                        "footpoint_reliable": observation.footpoint_reliable,
                    }
                ),
            )
        return oid

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


def _json_object_from_db(raw: Any, *, column: str, default_empty: bool = False) -> dict[str, Any]:
    if raw is None:
        if default_empty:
            return {}
        raise TypeError(f"{column} is required")
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{column} contains invalid JSON") from exc
        if isinstance(parsed, Mapping):
            return dict(parsed)
        raise TypeError(f"{column} must decode to a JSON object, got {type(parsed).__name__}")
    raise TypeError(f"{column} must be a JSON object, got {type(raw).__name__}")


def _json_object_from_domain(raw: Any, *, column: str) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    raise TypeError(f"{column} must be a mapping, got {type(raw).__name__}")


def _annotation_value(raw: Any, key: str) -> Any:
    annotations = _json_object_from_db(
        raw, column="tagged_keyframes.annotations", default_empty=True
    )
    bbox = annotations.get("bbox")
    if key in annotations:
        return annotations[key]
    if isinstance(bbox, Mapping) and key in bbox:
        return bbox[key]
    return None


def _annotation_float(raw: Any, key: str) -> float | None:
    value = _annotation_value(raw, key)
    if value is None:
        return None
    return float(value)


def _annotation_str(raw: Any, key: str) -> str | None:
    value = _annotation_value(raw, key)
    if value is None:
        return None
    return str(value)


def _cov2x2_tuple_from_metadata(raw: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(raw, list | tuple) or len(raw) != 4:
        return None
    return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))


def _row_to_ph(row: Any) -> PersonHypothesis:
    mean_raw: list[float] = [float(v) for v in row["state_mean"]]
    cov_raw: list[float] = [float(v) for v in row["state_cov"]]
    metadata = _json_object_from_db(
        row.get("metadata"), column="person_hypotheses.metadata", default_empty=True
    )
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
        last_independent_identity_evidence_at=row.get("last_independent_identity_evidence_at"),
        gallery_mean=row["gallery_mean"],
        height_estimate_m=row.get("height_m"),
        active_cameras=frozenset(row["active_cameras"] or []),
        closed_at=row.get("closed_at"),
        last_floor_speed_m_s=0.0,
        last_posture=None,
        metadata=metadata,
        mean_quality=float(row.get("mean_quality") or 0.0),
        view_prototypes=_unpack_view_prototypes(row.get("view_prototypes")),
    )


def _row_to_world_observation(row: Any) -> WorldObservation:
    bbox_raw = _json_object_from_db(row["bbox"], column="world_observations.bbox")
    metadata = _json_object_from_db(
        row.get("metadata"), column="world_observations.metadata", default_empty=True
    )

    return WorldObservation(
        observation_id=str(row["observation_id"]),
        camera_id=str(row["camera_id"]),
        frame_index=int(row["frame_index"]),
        captured_at=row["captured_at"],
        floor_point=FloorPoint(
            x_mm=int(float(row["floor_x_m"]) * 1000),
            y_mm=int(float(row["floor_y_m"]) * 1000),
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
        height_estimate_m=row.get("height_m"),
        face_anchor=None,
        quality=float(row.get("quality") or 0.0),
        floor_residual_m=metadata.get("floor_residual_m"),
        floor_cov_random=_cov2x2_tuple_from_metadata(metadata.get("floor_cov_random")),
        footpoint_reliable=bool(metadata.get("footpoint_reliable", True)),
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

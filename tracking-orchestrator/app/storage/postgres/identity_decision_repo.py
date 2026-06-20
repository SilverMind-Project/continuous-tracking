"""Postgres-backed Identity Decision repository.

Implements IdentityDecisionRepositoryProtocol using asyncpg against the
``continuous_tracking`` schema. Receives only an ``asyncpg.Pool``.
"""

from __future__ import annotations

import json
from datetime import UTC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncpg

from ...domain import (
    IdentityDecision,
    IdentityDecisionGalleryHit,
    IdentityEvidenceItem,
)


class PostgresIdentityDecisionRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def save(self, decision: IdentityDecision) -> None:
        # Use a transaction to ensure atomic writes
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO continuous_tracking.identity_decisions (
                        decision_id, ph_id, observation_id, captured_at,
                        inferred_identity_id, effective_identity_id, authority, decision_source,
                        conflict_kind, top_probability, second_probability, posterior_entropy,
                        last_independent_evidence_at, config_hash, resolver_version, model_set_version,
                        diagnostics_schema_version, diagnostics
                    ) VALUES (
                        $1, $2, $3, $4,
                        $5, $6, $7, $8,
                        $9, $10, $11, $12,
                        $13, $14, $15, $16,
                        $17, $18
                    )
                    ON CONFLICT (ph_id, observation_id, resolver_version) DO NOTHING
                    """,
                    decision.decision_id,
                    decision.ph_id,
                    decision.observation_id,
                    decision.captured_at,
                    decision.inferred_identity_id,
                    decision.effective_identity_id,
                    decision.authority,
                    decision.decision_source,
                    decision.conflict_kind,
                    decision.top_probability,
                    decision.second_probability,
                    decision.posterior_entropy,
                    decision.last_independent_evidence_at,
                    decision.config_hash,
                    decision.resolver_version,
                    decision.model_set_version,
                    decision.diagnostics_schema_version,
                    json.dumps(decision.diagnostics),
                )

                # insert evidence items
                if decision.evidence_items:
                    ev_args = []
                    for ev in decision.evidence_items:
                        ev_args.append((
                            decision.decision_id,
                            ev.source_identity_id,
                            ev.score_type,
                            ev.score_value,
                            ev.quality,
                            ev.camera_id,
                            ev.timestamp,
                            ev.model_version,
                            ev.preprocessing_version,
                            ev.calibration_version,
                            ev.directness,
                            ev.authoritative_eligibility,
                        ))
                    await conn.executemany(
                        """
                        INSERT INTO continuous_tracking.identity_evidence_items (
                            decision_id, source_identity_id, score_type, score_value,
                            quality, camera_id, timestamp, model_version,
                            preprocessing_version, calibration_version, directness,
                            authoritative_eligibility
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                        """,
                        ev_args
                    )

                # insert gallery hits
                if decision.gallery_hits:
                    hit_args = []
                    for hit in decision.gallery_hits:
                        hit_args.append((
                            decision.decision_id,
                            hit.entry_id,
                            hit.identity_id,
                            hit.raw_similarity,
                            hit.trust_multiplier,
                            hit.recency_factor,
                            hit.source_episode_group,
                            hit.orientation,
                            hit.rank,
                            hit.weighted_contribution,
                        ))
                    await conn.executemany(
                        """
                        INSERT INTO continuous_tracking.identity_decision_gallery_hits (
                            decision_id, entry_id, identity_id, raw_similarity,
                            trust_multiplier, recency_factor, source_episode_group,
                            orientation, rank, weighted_contribution
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        """,
                        hit_args
                    )

    async def get_decision(self, decision_id: str) -> IdentityDecision | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM continuous_tracking.identity_decisions WHERE decision_id = $1",
                decision_id
            )
            if not row:
                return None
            return await self._build_decision(conn, row)

    async def get_by_ph_id(self, ph_id: str, limit: int = 50, offset: int = 0) -> tuple[list[IdentityDecision], int]:
        async with self._pool.acquire() as conn:
            total_count = await conn.fetchval(
                "SELECT count(*) FROM continuous_tracking.identity_decisions WHERE ph_id = $1",
                ph_id
            )
            rows = await conn.fetch(
                """
                SELECT * FROM continuous_tracking.identity_decisions
                WHERE ph_id = $1
                ORDER BY captured_at DESC
                LIMIT $2 OFFSET $3
                """,
                ph_id, limit, offset
            )
            decisions = []
            for row in rows:
                decisions.append(await self._build_decision(conn, row))
            return decisions, total_count

    async def get_by_observation_id(self, observation_id: str) -> IdentityDecision | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM continuous_tracking.identity_decisions WHERE observation_id = $1",
                observation_id
            )
            if not row:
                return None
            return await self._build_decision(conn, row)

    async def _build_decision(self, conn: asyncpg.Connection, row: asyncpg.Record) -> IdentityDecision:
        ev_rows = await conn.fetch(
            "SELECT * FROM continuous_tracking.identity_evidence_items WHERE decision_id = $1",
            row["decision_id"]
        )
        hit_rows = await conn.fetch(
            "SELECT * FROM continuous_tracking.identity_decision_gallery_hits WHERE decision_id = $1 ORDER BY rank ASC",
            row["decision_id"]
        )
        
        evidence_items = []
        for ev in ev_rows:
            evidence_items.append(IdentityEvidenceItem(
                score_type=ev["score_type"],
                score_value=ev["score_value"],
                source_identity_id=ev["source_identity_id"],
                quality=ev["quality"],
                camera_id=ev["camera_id"],
                timestamp=ev["timestamp"].replace(tzinfo=UTC) if ev["timestamp"] else None,
                model_version=ev["model_version"],
                preprocessing_version=ev["preprocessing_version"],
                calibration_version=ev["calibration_version"],
                directness=ev["directness"],
                authoritative_eligibility=ev["authoritative_eligibility"],
            ))
            
        gallery_hits = []
        for hit in hit_rows:
            gallery_hits.append(IdentityDecisionGalleryHit(
                entry_id=str(hit["entry_id"]),
                identity_id=hit["identity_id"],
                raw_similarity=hit["raw_similarity"],
                trust_multiplier=hit["trust_multiplier"],
                recency_factor=hit["recency_factor"],
                rank=hit["rank"],
                weighted_contribution=hit["weighted_contribution"],
                source_episode_group=hit["source_episode_group"],
                orientation=hit["orientation"],
            ))

        return IdentityDecision(
            decision_id=str(row["decision_id"]),
            ph_id=str(row["ph_id"]),
            captured_at=row["captured_at"].replace(tzinfo=UTC),
            authority=row["authority"],
            decision_source=row["decision_source"],
            diagnostics=json.loads(row["diagnostics"]) if row["diagnostics"] else {},
            observation_id=str(row["observation_id"]) if row["observation_id"] else None,
            inferred_identity_id=row["inferred_identity_id"],
            effective_identity_id=row["effective_identity_id"],
            conflict_kind=row["conflict_kind"],
            top_probability=row["top_probability"],
            second_probability=row["second_probability"],
            posterior_entropy=row["posterior_entropy"],
            last_independent_evidence_at=row["last_independent_evidence_at"].replace(tzinfo=UTC) if row["last_independent_evidence_at"] else None,
            config_hash=row["config_hash"],
            resolver_version=row["resolver_version"],
            model_set_version=row["model_set_version"],
            diagnostics_schema_version=row["diagnostics_schema_version"],
            evidence_items=evidence_items,
            gallery_hits=gallery_hits,
        )

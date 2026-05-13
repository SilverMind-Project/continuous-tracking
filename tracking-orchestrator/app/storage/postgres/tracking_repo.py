"""Postgres/TimescaleDB implementation of the TrackingRepository protocol.

All SQL is centralized here. Services hold repository references, never
raw connection pools. This is the storage layer boundary.
"""

from __future__ import annotations

import json
from datetime import datetime

import asyncpg  # type: ignore[import-untyped]
from structlog import get_logger

from ...domain import (
    Detection,
    GlobalTrack,
    IdentityCandidate,
    IdentityRevision,
    TrackingEvent,
    Tracklet,
)
from ..base import TrackingRepository

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# SQL statements (parameterized, one per operation)
# ---------------------------------------------------------------------------

_SQL_SAVE_TRACKING_EVENT = """
    INSERT INTO continuous_tracking.tracking_events
        (event_id, event_time, camera_id, frame_index, frame_data)
    VALUES ($1, $2, $3, $4, $5::jsonb)
    ON CONFLICT DO NOTHING
"""

_SQL_GET_TRACKING_EVENT = """
    SELECT te.event_id, te.event_time, te.camera_id, te.frame_index, te.frame_data
    FROM continuous_tracking.tracking_events te
    WHERE te.event_id = $1
"""

_SQL_GET_DETECTIONS_FOR_EVENT = """
    SELECT detection_id, event_id, camera_id, bbox, embedding,
           confidence, tracklet_id, global_track_id,
           floor_point, capture_time, event_time AS det_event_time
    FROM continuous_tracking.detections
    WHERE event_id = $1
    ORDER BY detection_id
"""

_SQL_SAVE_DETECTIONS = """
    INSERT INTO continuous_tracking.detections (
        detection_id, event_id, camera_id, bbox, embedding,
        confidence, tracklet_id, global_track_id,
        floor_point, capture_time, event_time
    )
    VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, $8, $9::jsonb, $10, $11)
    ON CONFLICT (detection_id) DO NOTHING
"""

_SQL_SAVE_TRACKLET = """
    INSERT INTO continuous_tracking.tracklets
        (tracklet_id, camera_id, detection_ids, started_at, ended_at, state)
    VALUES ($1, $2, $3, $4, $5, $6)
    ON CONFLICT (tracklet_id) DO UPDATE SET
        detection_ids = EXCLUDED.detection_ids,
        ended_at = COALESCE(EXCLUDED.ended_at, continuous_tracking.tracklets.ended_at),
        state = CASE
            WHEN EXCLUDED.state = 'terminated' THEN 'terminated'
            ELSE continuous_tracking.tracklets.state
        END,
        updated_at = now()
"""

_SQL_GET_TRACKLET = """
    SELECT tracklet_id, camera_id, detection_ids, started_at, ended_at, state
    FROM continuous_tracking.tracklets
    WHERE tracklet_id = $1
"""

_SQL_SAVE_GLOBAL_TRACK = """
    INSERT INTO continuous_tracking.global_tracks
        (global_track_id, camera_ids, tracklet_ids, started_at, last_seen_at, state)
    VALUES ($1, $2, $3, $4, $5, $6)
    ON CONFLICT (global_track_id) DO UPDATE SET
        camera_ids = (
            SELECT array_agg(DISTINCT v)
            FROM (
                SELECT unnest(
                    EXCLUDED.camera_ids || continuous_tracking.global_tracks.camera_ids
                ) AS v
            ) sub
            WHERE v <> ''
        ),
        tracklet_ids = (
            SELECT array_agg(DISTINCT v)
            FROM (
                SELECT unnest(
                    EXCLUDED.tracklet_ids || continuous_tracking.global_tracks.tracklet_ids
                ) AS v
            ) sub
            WHERE v <> ''
        ),
        last_seen_at = GREATEST(
            EXCLUDED.last_seen_at, continuous_tracking.global_tracks.last_seen_at
        ),
        state = EXCLUDED.state,
        updated_at = now()
"""

_SQL_GET_GLOBAL_TRACK = """
    SELECT global_track_id, camera_ids, tracklet_ids, started_at, last_seen_at, state
    FROM continuous_tracking.global_tracks
    WHERE global_track_id = $1
"""

_SQL_SAVE_IDENTITY_REVISION = """
    INSERT INTO continuous_tracking.identity_revisions (revision_id, revision_time, global_track_id,
                                    tracklet_ids, candidates, map_identity_id,
                                    posterior_entropy, previous_identity_id,
                                    new_identity_id, reason, evidence)
    VALUES ($1, $2, $3, $4::uuid[], $5::jsonb, $6, $7, $8, $9, $10, $11::jsonb)
    ON CONFLICT (revision_id) DO NOTHING
"""

_SQL_LIST_IDENTITY_REVISIONS = """
    SELECT revision_id, revision_time, global_track_id, tracklet_ids, candidates,
           map_identity_id, posterior_entropy, previous_identity_id,
           new_identity_id, reason, evidence
    FROM continuous_tracking.identity_revisions
    WHERE global_track_id = $1
    ORDER BY revision_time DESC
    LIMIT 100
"""


class PostgresTrackingRepository(TrackingRepository):
    """Postgres/TimescaleDB implementation of the TrackingRepository.

    Requires an asyncpg connection pool. The pool is passed in at
    construction time and held for the lifetime of the repository.

    Usage::

        pool = await asyncpg.create_pool(dsn="postgresql://...")
        repo = PostgresTrackingRepository(pool)
        await repo.save_tracklet(tracklet)
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def save_tracking_event(self, event: TrackingEvent) -> str:
        """Store a tracking event. Returns its ID.

        Use save_tracking_event_with_detections() for atomic event+detections.
        """
        frame_data = {
            "minio_key": event.frame_ref.minio_key,
            "width": event.frame_ref.width,
            "height": event.frame_ref.height,
            "frame_index": event.frame_ref.frame_index,
            "capture_time": event.frame_ref.capture_time.isoformat(),
        }
        async with self._pool.acquire() as conn:
            await conn.execute(
                _SQL_SAVE_TRACKING_EVENT,
                event.event_id,
                event.event_time,
                event.camera_id,
                event.frame_index,
                json.dumps(frame_data),
            )
        return event.event_id

    async def save_tracking_event_with_detections(
        self, event: TrackingEvent, detections: list[Detection]
    ) -> str:
        """Store a tracking event and its detections atomically in a transaction."""
        frame_data = {
            "minio_key": event.frame_ref.minio_key,
            "width": event.frame_ref.width,
            "height": event.frame_ref.height,
            "frame_index": event.frame_ref.frame_index,
            "capture_time": event.frame_ref.capture_time.isoformat(),
        }
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                _SQL_SAVE_TRACKING_EVENT,
                event.event_id,
                event.event_time,
                event.camera_id,
                event.frame_index,
                json.dumps(frame_data),
            )
            if detections:
                values = []
                for det in detections:
                    bbox = {
                        "x_min": det.bbox.x_min,
                        "y_min": det.bbox.y_min,
                        "x_max": det.bbox.x_max,
                        "y_max": det.bbox.y_max,
                    }
                    floor_point = {
                        "x_mm": det.floor_point.x_mm,
                        "y_mm": det.floor_point.y_mm,
                        "calibrated": det.floor_point.calibrated,
                    }
                    embedding_json = json.dumps(det.embedding) if det.embedding else None
                    values.append(
                        (
                            det.detection_id,
                            event.event_id,
                            det.camera_id,
                            json.dumps(bbox),
                            embedding_json,
                            det.confidence,
                            det.tracklet_id,
                            det.global_track_id,
                            json.dumps(floor_point),
                            det.capture_time,
                            det.event_time,
                        )
                    )
                await conn.executemany(_SQL_SAVE_DETECTIONS, values)
        return event.event_id

    async def get_tracking_event(self, event_id: str) -> TrackingEvent | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(_SQL_GET_TRACKING_EVENT, event_id)
            det_rows = await conn.fetch(_SQL_GET_DETECTIONS_FOR_EVENT, event_id)
        if row is None:
            return None
        frame_data = json.loads(row["frame_data"])
        from ...domain import FrameRef

        frame_ref = FrameRef(
            minio_key=frame_data["minio_key"],
            width=frame_data["width"],
            height=frame_data["height"],
            frame_index=frame_data["frame_index"],
            capture_time=datetime.fromisoformat(frame_data["capture_time"]),
        )

        from ...domain import BoundingBox, Detection, FloorPoint

        detections: list[Detection] = []
        for dr in det_rows:
            det_bbox = json.loads(dr["bbox"])
            det_fp = json.loads(dr["floor_point"])
            det_emb = json.loads(dr["embedding"]) if dr["embedding"] else []
            detections.append(
                Detection(
                    detection_id=dr["detection_id"],
                    camera_id=dr["camera_id"],
                    bbox=BoundingBox(
                        x_min=det_bbox["x_min"],
                        y_min=det_bbox["y_min"],
                        x_max=det_bbox["x_max"],
                        y_max=det_bbox["y_max"],
                    ),
                    embedding=det_emb,
                    confidence=dr["confidence"],
                    tracklet_id=dr["tracklet_id"] or "",
                    global_track_id=dr["global_track_id"] or "",
                    floor_point=FloorPoint(
                        x_mm=det_fp["x_mm"],
                        y_mm=det_fp["y_mm"],
                        calibrated=det_fp["calibrated"],
                    ),
                    capture_time=dr["capture_time"],
                    event_time=dr["det_event_time"],
                )
            )
        return TrackingEvent(
            event_id=row["event_id"],
            camera_id=row["camera_id"],
            event_time=row["event_time"],
            frame_index=row["frame_index"],
            frame_ref=frame_ref,
            detections=detections,
        )

    async def save_detections(self, event_id: str, detections: list[Detection]) -> None:
        if not detections:
            return

        # Build array of detection tuples for COPY
        values = []
        for det in detections:
            bbox = {
                "x_min": det.bbox.x_min,
                "y_min": det.bbox.y_min,
                "x_max": det.bbox.x_max,
                "y_max": det.bbox.y_max,
            }
            floor_point = {
                "x_mm": det.floor_point.x_mm,
                "y_mm": det.floor_point.y_mm,
                "calibrated": det.floor_point.calibrated,
            }
            embedding_json = json.dumps(det.embedding) if det.embedding else None
            values.append(
                (
                    det.detection_id,
                    event_id,
                    det.camera_id,
                    json.dumps(bbox),
                    embedding_json,
                    det.confidence,
                    det.tracklet_id,
                    det.global_track_id,
                    json.dumps(floor_point),
                    det.capture_time,
                    det.event_time,
                )
            )

        async with self._pool.acquire() as conn:
            await conn.executemany(
                _SQL_SAVE_DETECTIONS,
                values,
            )

    async def save_tracklet(self, tracklet: Tracklet) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                _SQL_SAVE_TRACKLET,
                tracklet.tracklet_id,
                tracklet.camera_id,
                tracklet.detection_ids,
                tracklet.started_at,
                tracklet.ended_at,
                tracklet.state,
            )

    async def get_tracklet(self, tracklet_id: str) -> Tracklet | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(_SQL_GET_TRACKLET, tracklet_id)
        if row is None:
            return None
        return Tracklet(
            tracklet_id=row["tracklet_id"],
            camera_id=row["camera_id"],
            detection_ids=list(row["detection_ids"]),
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            state=row["state"],
        )

    async def save_global_track(self, track: GlobalTrack) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                _SQL_SAVE_GLOBAL_TRACK,
                track.global_track_id,
                track.camera_ids,
                track.tracklet_ids,
                track.started_at,
                track.last_seen_at,
                track.state,
            )

    async def get_global_track(self, global_track_id: str) -> GlobalTrack | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(_SQL_GET_GLOBAL_TRACK, global_track_id)
        if row is None:
            return None
        return GlobalTrack(
            global_track_id=row["global_track_id"],
            camera_ids=list(row["camera_ids"]),
            tracklet_ids=list(row["tracklet_ids"]),
            started_at=row["started_at"],
            last_seen_at=row["last_seen_at"],
            state=row["state"],
        )

    async def save_identity_revision(self, revision: IdentityRevision) -> None:
        candidates_json = json.dumps(
            [
                {
                    "identity_id": c.identity_id,
                    "display_name": c.display_name,
                    "probability": c.probability,
                }
                for c in revision.candidates
            ]
        )
        evidence_json = json.dumps(revision.evidence or {})
        async with self._pool.acquire() as conn:
            await conn.execute(
                _SQL_SAVE_IDENTITY_REVISION,
                revision.revision_id,
                revision.revision_time,
                revision.global_track_id,
                list(revision.tracklet_ids),
                candidates_json,
                revision.map_identity_id,
                revision.posterior_entropy,
                revision.previous_identity_id,
                revision.new_identity_id,
                revision.reason,
                evidence_json,
            )

    async def list_identity_revisions(
        self, global_track_id: str, after: datetime | None = None
    ) -> list[IdentityRevision]:
        sql = _SQL_LIST_IDENTITY_REVISIONS
        params: list[str | datetime] = [global_track_id]

        if after is not None:
            sql = _SQL_LIST_IDENTITY_REVISIONS.replace(
                "WHERE global_track_id = $1",
                "WHERE global_track_id = $1\n    AND revision_time >= $2",
            )
            params = [global_track_id, after]

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        revisions: list[IdentityRevision] = []
        for row in rows:
            candidates_data = json.loads(row["candidates"])
            candidates = [
                IdentityCandidate(
                    identity_id=c["identity_id"],
                    display_name=c["display_name"],
                    probability=c["probability"],
                )
                for c in candidates_data
            ]

            revisions.append(
                IdentityRevision(
                    revision_id=row["revision_id"],
                    revision_time=row["revision_time"],
                    global_track_id=row["global_track_id"],
                    tracklet_ids=list(row["tracklet_ids"]) if row["tracklet_ids"] else [],
                    candidates=candidates,
                    map_identity_id=row["map_identity_id"],
                    posterior_entropy=row["posterior_entropy"],
                    previous_identity_id=row["previous_identity_id"],
                    new_identity_id=row["new_identity_id"],
                    reason=row["reason"] or "",
                    evidence=json.loads(row["evidence"]) if row["evidence"] else {},
                )
            )
        return revisions

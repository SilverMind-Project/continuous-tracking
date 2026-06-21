import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, UTC
import math

from pydantic import BaseModel, ConfigDict
import numpy as np

class ReIDCandidateService:
    def __init__(self, gallery_repo, storage_client, config=None):
        self._gallery_repo = gallery_repo
        self._storage = storage_client
        self._config = config or {}

    async def create_candidate(
        self,
        entity,
        identity_id: str,
        embedding: list[float],
        crop_image_bytes: bytes,
        frame_image_bytes: bytes | None = None,
        model_version: str = "v1",
        preprocessing_version: str = "v1",
        dimensions: tuple[int, int] = (128, 256),
        quality: float = 1.0,
        orientation: int = 4,
        is_truncated: bool = False,
        is_occluded: bool = False,
        candidate_reason: str = "",
        ph_id: str | None = None,
        observation_id: str | None = None,
        keyframe_id: str | None = None,
        camera_id: str | None = None,
        capture_time: datetime | None = None,
        confidence: float = 1.0,
        source_episode_id: str | None = None,
        actor: str = "system",
        arcface_identity: str | None = None,
        effective_identity: str | None = None,
    ):
        # Candidate eligibility validates
        if not math.isfinite(sum(embedding)):
            raise ValueError("invalid_embedding: not finite")
            
        norm = sum(x*x for x in embedding) ** 0.5
        if abs(norm - 1.0) > 1e-4:
            raise ValueError("invalid_embedding: not L2 normalized")
            
        if is_truncated or is_occluded:
            raise ValueError("invalid_quality: truncated or occluded")
            
        # Face-derived candidates require direct authoritative ArcFace identity to equal the candidate identity and resolved effective identity
        if candidate_reason == "face_derived":
            if arcface_identity != identity_id or effective_identity != identity_id:
                raise ValueError("identity_mismatch: face identity does not match candidate/effective identity")

        import uuid
        candidate_id = str(uuid.uuid4())
        
        # Store dedicated immutable crop
        crop_hash = hashlib.sha256(crop_image_bytes).hexdigest()
        crop_key = f"reid-candidates/{model_version}/{candidate_id}.jpg"
        
        await self._storage.put_object(crop_key, crop_image_bytes)
        
        frame_hash = None
        frame_key = None
        if frame_image_bytes:
            frame_hash = hashlib.sha256(frame_image_bytes).hexdigest()
            frame_key = f"reid-candidates-frames/{model_version}/{candidate_id}.jpg"
            await self._storage.put_object(frame_key, frame_image_bytes)
            
        # Insert DB row
        embedding_str = f"[{','.join(f'{v:.8f}' for v in embedding)}]"
        async with self._gallery_repo._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO continuous_tracking.reid_gallery
                (id, identity_id, proposed_identity_id, effective_identity_id, embedding, quality,
                 state, model_version, preprocessing_version, dimension, source_frame_key, crop_key,
                 frame_hash, crop_hash, ph_id, observation_id, keyframe_id, camera_id, capture_time,
                 confidence, is_truncated, is_occluded, candidate_reason, source_episode_id, created_actor, origin_tracklet_id, orientation)
                VALUES ($1, $2, $3, $4, $5::vector, $6, 'pending_review', $7, $8, $9, $10, $11, $12, $13,
                        $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26)
                """,
                candidate_id, identity_id, identity_id, effective_identity, embedding_str, quality,
                model_version, preprocessing_version, dimensions[0] * dimensions[1], frame_key, crop_key,
                frame_hash, crop_hash,
                uuid.UUID(ph_id) if ph_id else None,
                uuid.UUID(observation_id) if observation_id else None,
                uuid.UUID(keyframe_id) if keyframe_id else None,
                camera_id, capture_time, confidence, is_truncated, is_occluded,
                candidate_reason,
                uuid.UUID(source_episode_id) if source_episode_id else None,
                actor,
                uuid.UUID(entity.entity_id) if entity else None,
                orientation
            )
        return candidate_id

    async def _transition_state(self, candidate_id: str, new_state: str, actor: str, reason: str = "", note: str = ""):
        async with self._gallery_repo._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT state, audit_version FROM continuous_tracking.reid_gallery WHERE id = $1", 
                    candidate_id
                )
                if not row:
                    raise ValueError(f"Candidate {candidate_id} not found")
                
                prev_state = row["state"]
                audit_version = row["audit_version"]
                
                if new_state == "rejected":
                    await conn.execute(
                        """
                        UPDATE continuous_tracking.reid_gallery
                        SET state = $1, embedding = NULL, reviewed_actor = $2, reviewed_time = now(),
                            review_reason = $3, review_note = $4, audit_version = audit_version + 1
                        WHERE id = $5 AND audit_version = $6
                        """,
                        new_state, actor, reason, note, candidate_id, audit_version
                    )
                else:
                    await conn.execute(
                        """
                        UPDATE continuous_tracking.reid_gallery
                        SET state = $1, reviewed_actor = $2, reviewed_time = now(),
                            review_reason = $3, review_note = $4, audit_version = audit_version + 1
                        WHERE id = $5 AND audit_version = $6
                        """,
                        new_state, actor, reason, note, candidate_id, audit_version
                    )
                    
                await conn.execute(
                    """
                    INSERT INTO continuous_tracking.gallery_review_events
                    (entry_id, previous_state, new_state, actor, reason, note, audit_version)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    candidate_id, prev_state, new_state, actor, reason, note, audit_version + 1
                )
                
                # Delete crop if rejected
                if new_state == "rejected":
                    row = await conn.fetchrow("SELECT crop_key FROM continuous_tracking.reid_gallery WHERE id = $1", candidate_id)
                    if row and row["crop_key"]:
                        try:
                            await self._storage.delete_object(row["crop_key"])
                        except Exception:
                            pass # We can have a reconciliation job clean this up later

    async def approve_candidate(self, candidate_id: str, actor: str, reason: str = ""):
        await self._transition_state(candidate_id, "operator_verified", actor, reason)

    async def reject_candidate(self, candidate_id: str, actor: str, reason: str = "", note: str = ""):
        # _transition_state already deletes the row's real crop_key inside the
        # transition. The earlier hardcoded "reid-candidates/v1/{id}.jpg" delete
        # was wrong for any model_version != "v1" (it orphaned the real crop and
        # could raise on a missing object), so it is gone.
        await self._transition_state(candidate_id, "rejected", actor, reason, note)

    async def relabel_candidate(self, candidate_id: str, new_identity_id: str, actor: str, reason: str = ""):
        async with self._gallery_repo._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT state, audit_version, identity_id FROM continuous_tracking.reid_gallery WHERE id = $1", 
                    candidate_id
                )
                if not row:
                    raise ValueError(f"Candidate {candidate_id} not found")
                
                prev_state = row["state"]
                audit_version = row["audit_version"]
                
                await conn.execute(
                    """
                    UPDATE continuous_tracking.reid_gallery
                    SET state = 'operator_verified', identity_id = $1,
                        reviewed_actor = $2, reviewed_time = now(),
                        review_reason = $3, audit_version = audit_version + 1
                    WHERE id = $4 AND audit_version = $5
                    """,
                    new_identity_id, actor, reason, candidate_id, audit_version
                )
                
                await conn.execute(
                    """
                    INSERT INTO continuous_tracking.gallery_review_events
                    (entry_id, previous_state, new_state, actor, reason, note, audit_version)
                    VALUES ($1, $2, 'operator_verified', $3, $4, $5, $6)
                    """,
                    candidate_id, prev_state, actor, reason, f"Relabeled from {row['identity_id']}", audit_version + 1
                )

    async def undo_review(self, candidate_id: str, actor: str, reason: str = ""):
        async with self._gallery_repo._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT state, audit_version FROM continuous_tracking.reid_gallery WHERE id = $1", 
                    candidate_id
                )
                if not row:
                    raise ValueError(f"Candidate {candidate_id} not found")
                
                prev_state = row["state"]
                audit_version = row["audit_version"]
                
                # Fetch the state prior to the current state from the events log
                # We want the state before the last transition.
                last_event = await conn.fetchrow(
                    "SELECT previous_state FROM continuous_tracking.gallery_review_events WHERE entry_id = $1 ORDER BY event_time DESC LIMIT 1",
                    candidate_id
                )
                new_state = last_event["previous_state"] if last_event else "pending_review"

                await conn.execute(
                    """
                    UPDATE continuous_tracking.reid_gallery
                    SET state = $1, reviewed_actor = $2, reviewed_time = now(),
                        review_reason = $3, audit_version = audit_version + 1
                    WHERE id = $4 AND audit_version = $5
                    """,
                    new_state, actor, reason, candidate_id, audit_version
                )
                
                await conn.execute(
                    """
                    INSERT INTO continuous_tracking.gallery_review_events
                    (entry_id, previous_state, new_state, actor, reason, note, audit_version)
                    VALUES ($1, $2, $3, $4, $5, 'Undo operation', $6)
                    """,
                    candidate_id, prev_state, new_state, actor, reason, audit_version + 1
                )


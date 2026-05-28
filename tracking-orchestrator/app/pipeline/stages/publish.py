"""Publish stage: emits TrackingEvent to Redis Streams."""

from __future__ import annotations

import json
from typing import cast

from structlog import get_logger

from ...transport.redis_streams import RedisStreamsTransport
from ..frame_context import FrameContext
from .base import FrameStage

logger = get_logger(__name__)


class PublishStage(FrameStage):
    name = "publish"

    def __init__(
        self,
        transport: RedisStreamsTransport,
        camera_room_map: dict[str, str] | None = None,
    ) -> None:
        self._transport = transport
        self._camera_room_map = camera_room_map or {}

    async def run(self, ctx: FrameContext) -> None:
        identities: dict[str, tuple[str, float]] = {}
        evidence_by_gt: dict[str, tuple[float, float, bool]] = {}

        # Primary source: WorldFrameSnapshots from this frame (WTR3).
        for snap in ctx.world_snapshots:
            if snap.identity_id and snap.identity_id != "UNKNOWN":
                identities[snap.ph_id] = (snap.identity_id, snap.identity_confidence)
                evidence_by_gt[snap.ph_id] = (
                    snap.identity_confidence,
                    0.0,
                    snap.direct_face_evidence,
                )

        # Augment with outcome_decisions for the second-probability field.
        if ctx.outcome_decisions:
            for decision in ctx.outcome_decisions:
                top_id, top_prob = decision.posterior.top_identity()
                if top_id == "UNKNOWN" or top_prob <= 0.0:
                    continue
                top_probs = sorted(decision.posterior.distribution.values(), reverse=True)
                top2_prob = top_probs[1] if len(top_probs) > 1 else 0.0
                ph_id = decision.global_track_id  # WTR3: entity_id is the PH id
                identities[ph_id] = (top_id, top_prob)
                evidence_by_gt[ph_id] = (top_prob, top2_prob, False)

        ctx.identities = identities
        ctx.evidence_by_gt = evidence_by_gt

        # Build identity_snapshots from WorldFrameSnapshots (WTR3).
        identity_snapshots: list[dict[str, object]] = []
        seen_ph_ids: set[str] = set()
        for snap in ctx.world_snapshots:
            seen_ph_ids.add(snap.ph_id)
            id_snap: dict[str, object] = {
                "ph_id": snap.ph_id,
                "identity_id": snap.identity_id or "",
                "top_probability": snap.identity_confidence,
                "second_probability": 0.0,
                "posterior_entropy": snap.posterior_entropy,
                "direct_face_evidence": snap.direct_face_evidence,
                "evidence_json": "{}",
            }
            identity_snapshots.append(id_snap)

        # Include outcome decisions for PHs not yet in snapshots.
        if ctx.outcome_decisions:
            for decision in ctx.outcome_decisions:
                ph_id = decision.global_track_id
                if ph_id in seen_ph_ids:
                    continue
                top_id, top_prob = decision.posterior.top_identity()
                top_probs = sorted(decision.posterior.distribution.values(), reverse=True)
                top2_prob = top_probs[1] if len(top_probs) > 1 else 0.0
                identity_snapshots.append(
                    {
                        "ph_id": ph_id,
                        "identity_id": decision.identity_id or "",
                        "top_probability": top_prob,
                        "second_probability": top2_prob,
                        "posterior_entropy": decision.posterior.entropy(),
                        "direct_face_evidence": (
                            cast(float, decision.evidence.get("direct_face_confidence", 0.0)) > 0
                            if decision.evidence
                            else False
                        ),
                        "evidence_json": (
                            json.dumps(decision.evidence) if decision.evidence else "{}"
                        ),
                    }
                )

        await self._transport.publish_event(
            camera_id=ctx.frame.camera_id,
            event_time=ctx.event_time,
            frame_index=ctx.frame.frame_index,
            detections=ctx.domain_detections if ctx.raw_detections else None,
            minio_key=ctx.frame.minio_key,
            room_name=self._camera_room_map.get(ctx.frame.camera_id, ""),
            identities=identities or None,
            frame_width=ctx.effective_width,
            frame_height=ctx.effective_height,
            capture_time_unix_ns=ctx.frame.capture_time_unix_ns,
            pose_results=ctx.det_pose_result if ctx.det_pose_result else None,
            trail_by_tracklet=ctx.trail_by_tracklet_snapshot or None,
            evidence_by_gt=evidence_by_gt or None,
            det_posture=cast("dict[str, str] | None", ctx.det_posture) if ctx.det_posture else None,
            identity_snapshots=identity_snapshots or None,
        )

        if ctx.new_revisions:
            logger.info(
                "Identity revisions emitted",
                camera_id=ctx.frame.camera_id,
                frame_index=ctx.frame.frame_index,
                revision_count=len(ctx.new_revisions),
            )

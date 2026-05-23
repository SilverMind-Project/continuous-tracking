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
        if ctx.active_tracklets and ctx.outcome_decisions:
            for decision in ctx.outcome_decisions:
                top_id, top_prob = decision.posterior.top_identity()
                if top_id == "UNKNOWN" or top_prob <= 0.0:
                    continue
                identities[decision.global_track_id] = (top_id, top_prob)
                top_probs = sorted(decision.posterior.distribution.values(), reverse=True)
                top2_prob = top_probs[1] if len(top_probs) > 1 else 0.0
                evidence_by_gt[decision.global_track_id] = (top_prob, top2_prob, False)

        for gt in ctx.active_global_tracks:
            if gt.global_track_id not in identities and gt.current_identity_id:
                identities[gt.global_track_id] = (gt.current_identity_id, 0.0)

        ctx.identities = identities
        ctx.evidence_by_gt = evidence_by_gt

        # Build identity_snapshots from outcome decisions.
        identity_snapshots: list[dict[str, object]] = []
        if ctx.outcome_decisions:
            for decision in ctx.outcome_decisions:
                top_id, top_prob = decision.posterior.top_identity()
                top_probs = sorted(decision.posterior.distribution.values(), reverse=True)
                top2_prob = top_probs[1] if len(top_probs) > 1 else 0.0
                snap: dict[str, object] = {
                    "global_track_id": decision.global_track_id,
                    "identity_id": decision.identity_id or "",
                    "top_probability": top_prob,
                    "second_probability": top2_prob,
                    "posterior_entropy": decision.posterior.entropy(),
                    "direct_face_evidence": (
                        cast(float, decision.evidence.get("direct_face_confidence", 0.0)) > 0
                        if decision.evidence
                        else False
                    ),
                    "evidence_json": json.dumps(decision.evidence) if decision.evidence else "{}",
                }
                identity_snapshots.append(snap)

        # Also include GTs with committed identities that have no outcome decision.
        for gt in ctx.active_global_tracks:
            if gt.current_identity_id and gt.global_track_id not in {
                d.global_track_id for d in ctx.outcome_decisions
            }:
                identity_snapshots.append(
                    {
                        "global_track_id": gt.global_track_id,
                        "identity_id": gt.current_identity_id,
                        "top_probability": 0.0,
                        "second_probability": 0.0,
                        "posterior_entropy": 0.0,
                        "direct_face_evidence": False,
                        "evidence_json": "{}",
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

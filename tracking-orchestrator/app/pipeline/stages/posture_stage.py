"""PostureStage: per-detection soft posture scoring.

Runs BEFORE TrajectoryStage. Calls PostureStrategy.score() for every
detection and stores PostureScores in ctx.det_posture_scores. This ensures
FusedPostureStrategy (including the depth slow-path) is called for every
detection, including tracked ones.

When the camera room is "bedroom" and the body is partially occluded
(low keypoint_confidence), a conservative lying prior is added to help
detect bed-sheet-occluded lying posture.
"""

from __future__ import annotations

from dataclasses import replace

from structlog import get_logger

from ...trajectory.posture import PostureScores, score_posture
from ...trajectory.posture_strategy import PostureStrategy
from ..frame_context import FrameContext
from ..types import LiveConfigHolder
from ._room_maps import camera_room_name
from .base import FrameStage

logger = get_logger(__name__)


def _apply_bedroom_prior(scores: PostureScores, prior: float) -> PostureScores:
    """Add bedroom lying prior to an existing PostureScores.

    Only applied when keypoint_confidence is below the occlusion threshold — i.e.,
    when the person's body is partially or fully occluded. The prior is capped so
    the total lying score never exceeds 1.0.
    """
    return replace(scores, lying=min(1.0, scores.lying + prior))


class PostureStage(FrameStage):
    name = "posture"

    # Lying prior added when: room == "bedroom" AND body is partially occluded.
    _BEDROOM_LYING_PRIOR = 0.18
    # Keypoint confidence below this threshold → body considered partially occluded.
    _OCCLUSION_CONFIDENCE_THRESHOLD = 0.35

    def __init__(
        self,
        live_config: LiveConfigHolder,
        posture_strategy: PostureStrategy | None = None,
    ) -> None:
        self._posture_strategy = posture_strategy
        self._live_config = live_config
        self._missing_room_binding_warnings: set[str] = set()

    async def run(self, ctx: FrameContext) -> None:
        room = await camera_room_name(self._live_config.camera_room_map, ctx.frame.camera_id)
        if room is None and ctx.frame.camera_id not in self._missing_room_binding_warnings:
            self._missing_room_binding_warnings.add(ctx.frame.camera_id)
            logger.warning("posture_camera_room_binding_missing", camera_id=ctx.frame.camera_id)
        room = room or ""
        is_bedroom = room.lower() in ("bedroom", "bed_room", "bed room")

        for domain_det in ctx.domain_detections:
            pose_result = ctx.det_pose_result.get(domain_det.detection_id)

            if self._posture_strategy is not None:
                image = ctx.require_image()
                scores = await self._posture_strategy.score(image, domain_det, pose_result)
                # Apply bedroom prior if strategy is depth-only (no keypoints).
                if is_bedroom and scores.keypoint_confidence < self._OCCLUSION_CONFIDENCE_THRESHOLD:
                    scores = _apply_bedroom_prior(scores, self._BEDROOM_LYING_PRIOR)
            elif pose_result is not None:
                bedroom_prior = self._BEDROOM_LYING_PRIOR if is_bedroom else 0.0
                scores = score_posture(pose_result, bedroom_prior=bedroom_prior)
            else:
                # No keypoints at all — if bedroom, give a small lying prior.
                if is_bedroom:
                    scores = PostureScores(
                        lying=self._BEDROOM_LYING_PRIOR,
                        sitting=0.0,
                        standing_walking=0.0,
                        keypoint_confidence=0.0,
                    )
                else:
                    scores = PostureScores(lying=0.0, sitting=0.0, standing_walking=0.0)

            # Bbox aspect ratio guard: wide box in bedroom → amplify lying prior.
            bbox = domain_det.bbox
            bbox_height = bbox.y_max - bbox.y_min
            bbox_width = bbox.x_max - bbox.x_min
            aspect_ratio = bbox_height / bbox_width if bbox_width > 0 else 1.0
            if is_bedroom and aspect_ratio < 0.5 and scores.lying > 0.0:
                # Box is wider than tall and we already have some lying evidence — amplify.
                aspect_boost = min(0.15, 0.15 * (0.5 - aspect_ratio) / 0.5)
                scores = replace(scores, lying=min(1.0, scores.lying + aspect_boost))

            ctx.det_posture_scores[domain_det.detection_id] = scores

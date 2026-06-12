"""M1 B5: Assert that refactored stage-runner segments match pre-refactor integer slices.

The pre-refactor code used ``stages[2:]``, ``stages[2:6]``, ``stages[7:]``.
After M1 these become named lists.  This test proves the name-to-slice
mapping is correct by asserting the expected stage name sequences.
"""

from __future__ import annotations

from app.pipeline.stages.base import StageRunner
from app.pipeline.stages.detection_backfill import DetectionBackfillStage
from app.pipeline.stages.face_identity import FaceIdentityStage
from app.pipeline.stages.inference import InferenceStage
from app.pipeline.stages.keyframes import KeyframeStage
from app.pipeline.stages.posture_stage import PostureStage
from app.pipeline.stages.posture_trails import TrailsStage

# Stage classes to construct
from app.pipeline.stages.privacy import PrivacyStage
from app.pipeline.stages.publish import PublishStage
from app.pipeline.stages.revisions import RevisionsStage
from app.pipeline.stages.spatial_projection import SpatialProjectionStage
from app.pipeline.stages.trajectory import ClosePHStage, TrajectoryStage
from app.pipeline.stages.world_tracking import WorldTrackingStage
from app.pipeline.types import LiveConfigHolder
from app.services.camera_room_map import CameraRoomMap, RoomPolygonMap


def _live_config() -> LiveConfigHolder:
    return LiveConfigHolder(CameraRoomMap(), RoomPolygonMap())


def _pre_world_stages() -> list:
    """Replicate the pre-world stage construction from frame_pipeline.py."""
    return [
        PrivacyStage(),
        SpatialProjectionStage(projection_service=None),  # type: ignore[arg-type]
        InferenceStage(reid_embedder=None, pose_estimator=None),  # type: ignore[arg-type]
        FaceIdentityStage(face_id_client=None),  # type: ignore[arg-type]
    ]


def _post_world_stages() -> list:
    """Replicate the post-world stage construction from frame_pipeline.py."""
    return [
        DetectionBackfillStage(),
        ClosePHStage(
            trajectory_writer=None,  # type: ignore[arg-type]
            motion_energy_tracker=None,  # type: ignore[arg-type]
            posture_tracker=None,  # type: ignore[arg-type]
            prev_active_ph_ids={},
        ),
        PostureStage(live_config=_live_config(), posture_strategy=None),  # type: ignore[arg-type]
        TrajectoryStage(
            trajectory_writer=None,  # type: ignore[arg-type]
            floor_projector=None,  # type: ignore[arg-type]
            motion_energy_tracker=None,  # type: ignore[arg-type]
            posture_tracker=None,  # type: ignore[arg-type]
        ),
        KeyframeStage(
            keyframe_sampler=None,  # type: ignore[arg-type]
            scene_publisher=None,  # type: ignore[arg-type]
            min_keyframe_detection_confidence=0.5,
        ),
        RevisionsStage(
            revision_publisher=None,  # type: ignore[arg-type]
            identity_rewriter=None,  # type: ignore[arg-type]
            bbox_repo=None,  # type: ignore[arg-type]
            identity_rewrite_on_face_commit=True,
        ),
        TrailsStage(trail_by_tracklet={}, trail_maxlen=100),
        PublishStage(transport=None, live_config=_live_config()),  # type: ignore[arg-type]
    ]


class TestStageRunnerSegments:
    """Assert that M1 refactored stage-runner segments match pre-refactor slices."""

    def _names(self, runner: StageRunner) -> list[str]:
        return [s.name for s in runner._stages]

    def test_pre_world_runner_stage_names(self) -> None:
        """pre_world_runner = stages[2:6]."""
        pre_world = _pre_world_stages()
        runner = StageRunner(pre_world)
        assert self._names(runner) == [
            "privacy",
            "spatial_projection",
            "inference",
            "face_identity",
        ]

    def test_post_world_runner_stage_names(self) -> None:
        """post_world_runner = stages[7:] starts with detection_backfill, ends with publish."""
        post_world = _post_world_stages()
        runner = StageRunner(post_world)
        names = self._names(runner)
        assert names == [
            "detection_backfill",
            "close_ph",
            "posture",
            "trajectory",
            "keyframes",
            "revisions",
            "trails",
            "publish",
        ]

    def test_post_detect_runner_stage_names(self) -> None:
        """post_detect_runner = stages[2:] = pre_world + [world] + post_world."""
        pre_world = _pre_world_stages()
        post_world = _post_world_stages()
        world_stage = WorldTrackingStage(  # type: ignore[arg-type]
            tracker=None,
            live_config=_live_config(),
            config=None,
        )
        all_post_detect = [*pre_world, world_stage, *post_world]
        runner = StageRunner(all_post_detect)
        names = self._names(runner)
        assert names[:4] == [
            "privacy",
            "spatial_projection",
            "inference",
            "face_identity",
        ]
        assert names[4] == "world_tracking"
        assert names[5:] == [
            "detection_backfill",
            "close_ph",
            "posture",
            "trajectory",
            "keyframes",
            "revisions",
            "trails",
            "publish",
        ]
        assert len(names) == 13  # 4 + 1 + 8

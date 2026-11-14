"""Overlap groups from the CC config sync reach every pipeline consumer."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from app.domain import FloorPoint, OverlapGroup
from app.inference.schemas import DetectionBox
from app.pipeline.frame_pipeline import FrameProcessingPipeline
from app.pipeline.reid_policy import AdaptiveReidConfig, ReidNeedPolicy

_NOW = datetime(2026, 11, 14, 10, 0, 0, tzinfo=UTC)


def _embeds(policy: ReidNeedPolicy, camera_id: str) -> tuple[bool, str]:
    # One detection with no nearby PH embeds as "no_nearby_ph" unless the camera is grouped.
    should, reason, _ = policy.should_embed_frame(
        [DetectionBox(x1=0.1, y1=0.1, x2=0.3, y2=0.6, confidence=0.9)],
        {0: FloorPoint(x_mm=1000, y_mm=1000, calibrated=True)},
        [],
        camera_id,
        _NOW,
    )
    return should, reason


def test_set_overlap_groups_updates_tracker_and_reid_policy() -> None:
    pipeline = FrameProcessingPipeline()
    tracker = MagicMock()
    policy = ReidNeedPolicy(config=AdaptiveReidConfig(enabled=True, shadow=False))
    pipeline._world_tracker = tracker
    pipeline._reid_policy = policy
    groups = [OverlapGroup(group_id="1", camera_ids=("cam-a", "cam-b"))]

    pipeline.set_overlap_groups(groups)

    tracker.set_overlap_groups.assert_called_once_with(groups)
    assert _embeds(policy, "cam-a") == (True, "overlap_group_camera")
    assert _embeds(policy, "cam-c")[1] != "overlap_group_camera"

    pipeline.set_overlap_groups([])

    assert _embeds(policy, "cam-a")[1] != "overlap_group_camera"


def test_set_overlap_groups_before_initialize_is_harmless() -> None:
    FrameProcessingPipeline().set_overlap_groups(
        [OverlapGroup(group_id="1", camera_ids=("cam-a", "cam-b"))]
    )

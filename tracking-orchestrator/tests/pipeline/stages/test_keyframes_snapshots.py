"""tests for snapshot-driven KeyframeStage.

Verifies that KeyframeStage consumes ctx.world_snapshots and samples/triggers
keyframes per PH per frame.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from app.domain import (
    BoundingBox,
    IdentityRevision,
    WorldFrameSnapshot,
)
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.keyframes import KeyframeStage
from app.sampling.keyframe_sampler import KeyframeSampler, SamplerConfig
from app.storage.base import InMemoryKeyframeRepository


def _fake_frame_ready(*, camera_id: str = "cam1", frame_index: int = 1) -> MagicMock:
    return MagicMock(camera_id=camera_id, frame_index=frame_index)


def _make_snapshot(
    *,
    ph_id: str = "ph-1",
    camera_id: str = "cam1",
    frame_index: int = 1,
    identity_id: str | None = None,
    captured_at: datetime | None = None,
) -> WorldFrameSnapshot:
    if captured_at is None:
        captured_at = datetime.now(UTC)
    return WorldFrameSnapshot(
        ph_id=ph_id,
        camera_id=camera_id,
        frame_index=frame_index,
        captured_at=captured_at,
        floor_x_m=0.0,
        floor_y_m=0.0,
        floor_vx_m_s=0.0,
        floor_vy_m_s=0.0,
        position_sigma_m=0.1,
        identity_id=identity_id,
        identity_confidence=0.0,
        detection_confidence=0.9,
    )


def _make_ctx(
    *,
    camera_id: str = "cam1",
    frame_index: int = 1,
    now: datetime | None = None,
    snapshots: list[WorldFrameSnapshot] | None = None,
    revisions: list[IdentityRevision] | None = None,
    minio_key: str = "minio/test.jpg",
    effective_width: int = 1920,
    effective_height: int = 1080,
) -> FrameContext:
    if now is None:
        now = datetime.now(UTC)
    frame = _fake_frame_ready(camera_id=camera_id, frame_index=frame_index)
    frame.minio_key = minio_key
    ctx = FrameContext(
        frame=frame,  # type: ignore[arg-type]
        event_time=now,
        capture_time=now,
    )
    ctx.world_snapshots = snapshots or []
    ctx.new_revisions = revisions or []
    ctx.domain_detections = []
    ctx.effective_width = effective_width
    ctx.effective_height = effective_height
    return ctx


def _make_revision(
    ph_id: str = "ph-1",
    new_identity_id: str = "alice",
    *,
    now: datetime | None = None,
) -> IdentityRevision:
    if now is None:
        now = datetime.now(UTC)
    return IdentityRevision(
        revision_id=f"rev-{ph_id}",
        ph_id=ph_id,
        previous_identity_id=None,
        new_identity_id=new_identity_id,
        actor="resolver",
        reason="test",
        applied_at=now,
        rewritten_rows=0,
    )


class TestKeyframesFromSnapshots:
    async def test_periodic_sample_per_open_ph(self) -> None:
        """Each open PH on current camera with a detection triggers maybe_sample."""
        now = datetime.now(UTC)
        snap_a = _make_snapshot(ph_id="ph-a", camera_id="cam1", captured_at=now)
        snap_b = _make_snapshot(ph_id="ph-b", camera_id="cam1", captured_at=now)

        repo = InMemoryKeyframeRepository()
        cfg = SamplerConfig(keyframe_min_interval_s=0.0)  # no interval gate
        sampler = KeyframeSampler(repo=repo, config=cfg)
        stage = KeyframeStage(keyframe_sampler=sampler)

        ctx = _make_ctx(now=now, snapshots=[snap_a, snap_b])
        # No detections have ph_id set → det_by_ph empty →
        # snapshots are skipped entirely.
        # To test the sampling path, we need detections with ph_id.
        # Using MagicMock to simulate ph-stamped detections.
        from app.domain import Detection as Det

        det_a = MagicMock(spec=Det)
        det_a.ph_id = "ph-a"
        det_a.confidence = 0.9
        det_a.bbox = BoundingBox(0, 0, 100, 200)
        det_b = MagicMock(spec=Det)
        det_b.ph_id = "ph-b"
        det_b.confidence = 0.85
        det_b.bbox = BoundingBox(50, 50, 150, 250)
        ctx.domain_detections = [det_a, det_b]  # type: ignore[list-item]

        await stage.run(ctx)

        keyframes = await repo.list_keyframes()
        ph_ids = {kf.ph_id for kf in keyframes}
        assert ph_ids == {"ph-a", "ph-b"}, f"expected both PHs sampled, got {ph_ids}"

    async def test_trigger_sample_on_revision(self) -> None:
        """Revision for PH-A triggers identity_changed sample."""
        now = datetime.now(UTC)
        snap_a = _make_snapshot(ph_id="ph-a", camera_id="cam1", captured_at=now)
        snap_b = _make_snapshot(ph_id="ph-b", camera_id="cam1", captured_at=now)
        rev_a = _make_revision(ph_id="ph-a", now=now)

        repo = InMemoryKeyframeRepository()
        cfg = SamplerConfig(keyframe_min_interval_s=0.0)
        sampler = KeyframeSampler(repo=repo, config=cfg)
        stage = KeyframeStage(keyframe_sampler=sampler)

        ctx = _make_ctx(now=now, snapshots=[snap_a, snap_b], revisions=[rev_a])

        from app.domain import Detection as Det

        det_a = MagicMock(spec=Det)
        det_a.ph_id = "ph-a"
        det_a.confidence = 0.9
        det_a.bbox = BoundingBox(10, 10, 120, 220)
        det_b = MagicMock(spec=Det)
        det_b.ph_id = "ph-b"
        det_b.confidence = 0.85
        det_b.bbox = BoundingBox(50, 50, 150, 250)
        ctx.domain_detections = [det_a, det_b]  # type: ignore[list-item]

        await stage.run(ctx)

        keyframes = await repo.list_keyframes()
        rev_keyframes = [kf for kf in keyframes if kf.tag_reason == "identity_changed"]
        assert len(rev_keyframes) == 1
        assert rev_keyframes[0].ph_id == "ph-a"

    async def test_skipped_for_low_confidence_detection(self) -> None:
        """Detection below min_keyframe_detection_confidence → dropped."""
        now = datetime.now(UTC)
        snap = _make_snapshot(ph_id="ph-1", camera_id="cam1", captured_at=now)

        repo = InMemoryKeyframeRepository()
        cfg = SamplerConfig(keyframe_min_interval_s=0.0)
        sampler = KeyframeSampler(repo=repo, config=cfg)
        stage = KeyframeStage(keyframe_sampler=sampler, min_keyframe_detection_confidence=0.5)

        ctx = _make_ctx(now=now, snapshots=[snap])

        from app.domain import Detection as Det

        det = MagicMock(spec=Det)
        det.ph_id = "ph-1"
        det.confidence = 0.3  # below threshold
        det.bbox = BoundingBox(0, 0, 100, 200)
        ctx.domain_detections = [det]  # type: ignore[list-item]

        with patch("app.pipeline.stages.keyframes._metrics") as mock_metrics:
            await stage.run(ctx)
            mock_metrics.metrics.keyframe_dropped_low_confidence_total.inc.assert_called_once()

        keyframes = await repo.list_keyframes()
        assert len(keyframes) == 0

    async def test_no_snapshots_no_keyframes(self) -> None:
        """Empty world_snapshots → no keyframe sampling attempted."""
        repo = InMemoryKeyframeRepository()
        sampler = KeyframeSampler(repo=repo)
        stage = KeyframeStage(keyframe_sampler=sampler)

        ctx = _make_ctx(snapshots=[])
        await stage.run(ctx)

        keyframes = await repo.list_keyframes()
        assert len(keyframes) == 0

    async def test_skips_other_camera_snapshots(self) -> None:
        """Only current-frame camera's PHs get keyframe samples."""
        now = datetime.now(UTC)
        snap_a = _make_snapshot(ph_id="ph-a", camera_id="cam-a", captured_at=now)
        snap_b = _make_snapshot(ph_id="ph-b", camera_id="cam-b", captured_at=now)

        repo = InMemoryKeyframeRepository()
        cfg = SamplerConfig(keyframe_min_interval_s=0.0)
        sampler = KeyframeSampler(repo=repo, config=cfg)
        stage = KeyframeStage(keyframe_sampler=sampler)

        ctx = _make_ctx(camera_id="cam-a", now=now, snapshots=[snap_a, snap_b])

        from app.domain import Detection as Det

        det_a = MagicMock(spec=Det)
        det_a.ph_id = "ph-a"
        det_a.confidence = 0.9
        det_a.bbox = BoundingBox(0, 0, 100, 200)
        ctx.domain_detections = [det_a]  # type: ignore[list-item]

        await stage.run(ctx)

        keyframes = await repo.list_keyframes()
        assert len(keyframes) == 1
        assert keyframes[0].ph_id == "ph-a"

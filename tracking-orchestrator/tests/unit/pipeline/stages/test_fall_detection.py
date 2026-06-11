"""Unit tests for FallDetectionStage and stage-list wiring."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from app.domain import BoundingBox, DementiaSignal, Detection, WorldFrameSnapshot
from app.inference.schemas import NUM_KEYPOINTS, Keypoint, PoseResult
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.fall_detection import FallDetectionConfig, FallDetectionStage
from app.trajectory.posture import PostureScores
from app.transport.redis_streams import FrameReady

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 6, 11, 10, 0, 0, tzinfo=UTC)
_IDENTITY = "identity-alice"
_PH = "ph-001"
_CAM = "cam-1"


class _FakePublisher:
    """Captures publish_signal calls without a Redis connection."""

    def __init__(self) -> None:
        self.published: list[DementiaSignal] = []

    async def publish_signal(self, signal: DementiaSignal) -> str:
        self.published.append(signal)
        return "fake-id"

    async def publish_batch(self, signals: list[DementiaSignal]) -> list[str]:
        self.published.extend(signals)
        return ["fake-id"] * len(signals)


class _InMemorySignalRepo:
    """Minimal in-memory signal repo for tests."""

    def __init__(self) -> None:
        self.signals: dict[str, DementiaSignal] = {}

    async def upsert_signal(self, signal: DementiaSignal) -> None:
        self.signals[signal.signal_id] = signal

    async def list_signals(self, **kwargs: Any) -> list[DementiaSignal]:
        return list(self.signals.values())


def _kps(y: float = 0.9) -> tuple[Keypoint, ...]:
    return tuple(Keypoint(x=0.5, y=y, score=0.9) for _ in range(NUM_KEYPOINTS))


def _make_snap(
    ph_id: str = _PH,
    identity_id: str | None = _IDENTITY,
    room_name: str = "living_room",
    floor_vx_m_s: float = 0.0,
    floor_vy_m_s: float = 0.0,
) -> WorldFrameSnapshot:
    return WorldFrameSnapshot(
        ph_id=ph_id,
        camera_id=_CAM,
        frame_index=1,
        captured_at=_T0,
        floor_x_m=2.0,
        floor_y_m=3.0,
        floor_vx_m_s=floor_vx_m_s,
        floor_vy_m_s=floor_vy_m_s,
        position_sigma_m=0.05,
        identity_id=identity_id,
        identity_confidence=0.9 if identity_id else 0.0,
        posterior_entropy=0.1,
        direct_face_evidence=bool(identity_id),
        bbox=BoundingBox(x_min=100, y_min=50, x_max=220, y_max=350),
        detection_confidence=0.95,
        height_m=1.7,
        room_id="r1",
        room_name=room_name,
    )


def _make_det(
    det_id: str = "d1",
    ph_id: str = _PH,
    bbox_y_min: float = 50.0,
    bbox_height: float = 300.0,
) -> Detection:
    return Detection(
        detection_id=det_id,
        camera_id=_CAM,
        bbox=BoundingBox(
            x_min=100,
            y_min=round(bbox_y_min),
            x_max=220,
            y_max=round(bbox_y_min + bbox_height),
        ),
        embedding=[],
        capture_time=_T0,
        event_time=_T0,
        confidence=0.95,
        tracklet_id="",
        ph_id=ph_id,
    )


def _make_pose(y: float = 0.9) -> PoseResult:
    return PoseResult(keypoints=_kps(y))


def _make_ctx(
    detections: list[Detection],
    snapshots: list[WorldFrameSnapshot],
    pose_results: dict[str, PoseResult] | None = None,
    posture_scores: dict[str, PostureScores] | None = None,
    event_time: datetime = _T0,
) -> FrameContext:
    frame = FrameReady(
        camera_id=_CAM,
        minio_key="k",
        width=640,
        height=480,
        frame_index=1,
        capture_time_unix_ns=int(event_time.timestamp() * 1e9),
    )
    ctx = FrameContext(
        frame=frame,
        event_time=event_time,
        capture_time=event_time,
    )
    ctx.domain_detections = detections
    ctx.world_snapshots = snapshots
    ctx.det_pose_result = pose_results or {}
    ctx.det_posture_scores = posture_scores or {}
    return ctx


def _default_config(**kwargs: Any) -> FallDetectionConfig:
    return FallDetectionConfig(
        enabled=True,
        min_samples=3,
        max_descent_rate_hps_threshold=0.8,
        height_ratio_threshold=0.55,
        lying_score_threshold=0.4,
        floor_speed_max_m_s=2.0,
        escalation_window_s=60.0,
        stillness_motion_floor=0.05,
        standing_clear_height_ratio=0.75,
        standing_clear_lying_score=0.2,
        cooldown_minutes=10,
        resting_rooms=("bed", "bedroom"),
        **kwargs,
    )


def _make_stage(
    config: FallDetectionConfig | None = None,
) -> tuple[FallDetectionStage, _InMemorySignalRepo, _FakePublisher]:
    repo = _InMemorySignalRepo()
    publisher = _FakePublisher()
    cfg = config or _default_config()
    stage = FallDetectionStage(
        config=cfg,
        signal_repo=repo,  # type: ignore[arg-type]
        signal_publisher=publisher,  # type: ignore[arg-type]
    )
    return stage, repo, publisher


# ---------------------------------------------------------------------------
# Scripted fall sequence: 10 frames that cross the detection threshold
# ---------------------------------------------------------------------------


def _scripted_fall_frames(
    *,
    start_time: datetime = _T0,
    n_standing: int = 4,
    n_falling: int = 3,
    lying_score: float = 0.7,
    pose_available: bool = True,
) -> list[tuple[FrameContext, datetime]]:
    """Return a sequence of (ctx, event_time) pairs scripted to produce a fall.

    Standing frames: high y_min (high position), full height.
    Falling frames: rapidly increasing y_min (descent), lower height.
    """
    det_id = "d1"
    ph_id = _PH
    snap = _make_snap()
    frames_out: list[tuple[FrameContext, datetime]] = []
    fps = 5.0
    dt = 1.0 / fps

    # Standing phase: bbox_y_min=50, height=300, hip at crop-y=0.5 → hip_abs ≈ 200.
    for i in range(n_standing):
        t = start_time + timedelta(seconds=i * dt)
        det = _make_det(det_id=det_id, ph_id=ph_id, bbox_y_min=50.0, bbox_height=300.0)
        pose = _make_pose(y=0.5) if pose_available else None
        posture = PostureScores(lying=0.0, sitting=0.0, standing_walking=0.9)
        ctx = _make_ctx(
            [det],
            [snap],
            {det_id: pose} if pose else {},
            {det_id: posture},
            event_time=t,
        )
        frames_out.append((ctx, t))

    # Fall phase: 50 px/frame hip descent at 5 fps → 0.833 hps (threshold 0.8).
    # bbox collapses toward lying: height drops from 300 → 100 over n_falling frames.
    # Rate calc: Δhip = 50 px, h_est ≈ 300 (p90 stable), dt = 0.2 s → 50/300/0.2 = 0.833.
    for j in range(n_falling):
        frac = (j + 1) / n_falling
        t = start_time + timedelta(seconds=(n_standing + j) * dt)
        target_hip = 200.0 + frac * 150.0  # 50 px per frame
        new_height = max(300.0 - frac * 200.0, 50.0)  # collapses: 300 → ~100
        new_y_min = target_hip - 0.5 * new_height
        det = _make_det(det_id=det_id, ph_id=ph_id, bbox_y_min=new_y_min, bbox_height=new_height)
        pose = _make_pose(y=0.5) if pose_available else None
        posture = PostureScores(lying=lying_score, sitting=0.0, standing_walking=0.0)
        ctx = _make_ctx(
            [det],
            [snap],
            {det_id: pose} if pose else {},
            {det_id: posture},
            event_time=t,
        )
        frames_out.append((ctx, t))

    return frames_out


# ---------------------------------------------------------------------------
# Stage integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stage_emits_exactly_one_signal_on_scripted_fall() -> None:
    stage, _repo, _publisher = _make_stage()
    frames = _scripted_fall_frames()

    for ctx, _ in frames:
        await stage.run(ctx)

    signals = list(_repo.signals.values())
    assert len(signals) == 1
    sig = signals[0]
    assert sig.signal_kind == "fall_suspected"
    assert sig.severity == "warning"
    assert sig.identity_id == _IDENTITY


@pytest.mark.asyncio
async def test_stage_published_signal_has_fall_suspected_kind() -> None:
    stage, _, publisher = _make_stage()
    frames = _scripted_fall_frames()

    for ctx, _ in frames:
        await stage.run(ctx)

    assert len(publisher.published) == 1
    assert publisher.published[0].signal_kind == "fall_suspected"


@pytest.mark.asyncio
async def test_escalation_to_emergency_on_stillness() -> None:
    """After warning, if post-event stillness confirmed → escalate to emergency."""
    stage, _repo, _publisher = _make_stage()

    # Run scripted fall to get warning.
    for ctx, _ in _scripted_fall_frames():
        await stage.run(ctx)

    warning_signals = [s for s in _repo.signals.values() if s.severity == "warning"]
    assert len(warning_signals) >= 1

    # Now run more frames at floor level with low motion → escalation.
    snap = _make_snap()
    det_id = "d1"
    t_base = _T0 + timedelta(seconds=2.0)
    for k in range(5):
        t = t_base + timedelta(seconds=k * 0.2)
        det = _make_det(det_id=det_id, bbox_y_min=230.0, bbox_height=120.0)
        posture = PostureScores(lying=0.8, sitting=0.0, standing_walking=0.0)
        ctx = _make_ctx(
            [det],
            [snap],
            {det_id: _make_pose(y=0.5)},
            {det_id: posture},
            event_time=t,
        )
        await stage.run(ctx)

    # Should now have an emergency signal (same signal_id, severity escalated).
    severities = {s.severity for s in _repo.signals.values()}
    assert "emergency" in severities


@pytest.mark.asyncio
async def test_movement_resumption_closes_episode() -> None:
    """After warning, standing 2+ frames clears the episode; no re-escalation."""
    stage, _repo, publisher = _make_stage()
    initial_count = len(publisher.published)

    # Run scripted fall.
    for ctx, _ in _scripted_fall_frames():
        await stage.run(ctx)

    # Confirm warning was emitted.
    assert len(publisher.published) > initial_count

    # Run 3 standing-clear frames.
    snap = _make_snap()
    det_id = "d1"
    t_base = _T0 + timedelta(seconds=3.0)
    for k in range(3):
        t = t_base + timedelta(seconds=k * 0.2)
        det = _make_det(det_id=det_id, bbox_y_min=50.0, bbox_height=300.0)
        posture = PostureScores(lying=0.05, sitting=0.0, standing_walking=0.9)
        ctx = _make_ctx(
            [det],
            [snap],
            {det_id: _make_pose(y=0.5)},
            {det_id: posture},
            event_time=t,
        )
        await stage.run(ctx)

    # Episode cleared; PH should no longer be in _emission_at.
    assert _PH not in stage._emission_at


@pytest.mark.asyncio
async def test_unidentified_ph_suppressed_not_emitted() -> None:
    """PH with no identity → signal suppressed, unidentified counter incremented."""
    stage, _repo, publisher = _make_stage()
    snap = _make_snap(identity_id=None)

    # Run fall frames with unidentified PH.
    for ctx, _ in _scripted_fall_frames():
        # Patch the snapshot to have no identity.
        ctx.world_snapshots = [snap]
        await stage.run(ctx)

    assert len(publisher.published) == 0
    assert len(_repo.signals) == 0


@pytest.mark.asyncio
async def test_intra_episode_deduplication() -> None:
    """After a warning fires, continued fall-detection frames within the same
    episode do not produce additional warning signals."""
    stage, _repo, publisher = _make_stage()

    # First fall.
    for ctx, _ in _scripted_fall_frames():
        await stage.run(ctx)

    initial_count = len(publisher.published)
    assert initial_count == 1  # exactly one warning

    # Person stays on the floor: high lying score, same low bbox height.
    # These frames keep triggering check_impact but should NOT re-emit.
    snap = _make_snap()
    det_id = "d1"
    # Use the same episode_key window (within 10 s of T0).
    for k in range(5):
        t = _T0 + timedelta(seconds=1.5 + k * 0.2)
        det = _make_det(det_id=det_id, bbox_y_min=230.0, bbox_height=120.0)
        posture = PostureScores(lying=0.8, sitting=0.0, standing_walking=0.0)
        ctx = _make_ctx(
            [det],
            [snap],
            {det_id: _make_pose(y=0.5)},
            {det_id: posture},
            event_time=t,
        )
        await stage.run(ctx)

    # Still only the initial warning; no duplicate signals.
    warning_count = sum(1 for s in publisher.published if s.severity == "warning")
    assert warning_count == 1


@pytest.mark.asyncio
async def test_resting_room_fall_not_emitted() -> None:
    """Falls in resting rooms are not emitted."""
    stage, _repo, publisher = _make_stage()
    bedroom_snap = _make_snap(room_name="bedroom")

    for ctx, _ in _scripted_fall_frames():
        ctx.world_snapshots = [bedroom_snap]
        await stage.run(ctx)

    assert len(publisher.published) == 0


# ---------------------------------------------------------------------------
# Pipeline stage wiring tests
# ---------------------------------------------------------------------------


def test_fall_detection_stage_absent_when_disabled() -> None:
    """When fall_detection.enabled=False, stage is not in post_world_stages."""
    from app.pipeline.frame_pipeline import FrameProcessingPipeline, PipelineConfig
    from app.pipeline.stages.fall_detection import FallDetectionConfig

    config = PipelineConfig(
        fall_detection=FallDetectionConfig(enabled=False),
    )
    pipeline = FrameProcessingPipeline(config)
    assert pipeline._fall_detection_stage is None


def test_fall_detection_config_enabled_flag_defaults_false() -> None:
    """FallDetectionConfig defaults to enabled=False (ships dark)."""
    cfg = FallDetectionConfig()
    assert cfg.enabled is False

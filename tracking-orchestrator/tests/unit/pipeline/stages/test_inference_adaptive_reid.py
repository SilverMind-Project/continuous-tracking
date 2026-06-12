"""Pipeline-level tests for InferenceStage adaptive ReID cadence (M5.1).

Tests:
  1. Two-person frame always embeds (multi_detect trigger).
  2. Single-resolved-PH steady state skips after first refresh interval.
  3. Face-anchor frame embeds (recognized anchor overrides steady state).
  4. Empty-embedding tolerance: PH update path does not error on embedding=[].
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from app.domain import FaceAnchor, FloorPoint, PersonHypothesis
from app.inference.schemas import DetectionBox
from app.pipeline.frame_context import FrameContext
from app.pipeline.reid_policy import AdaptiveReidConfig, ReidNeedPolicy
from app.pipeline.stages.inference import InferenceStage
from app.transport.redis_streams import FrameReady

_T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
_REFRESH_S = 10.0
_PROXIMITY_M = 2.0

_FRAME = FrameReady(
    camera_id="cam-1",
    minio_key="k",
    width=640,
    height=480,
    frame_index=0,
    capture_time_unix_ns=int(_T0.timestamp() * 1e9),
)


def _ctx(
    dets: list[DetectionBox] | None = None,
    floor_points: dict[int, FloorPoint] | None = None,
    face_anchors: list[FaceAnchor] | None = None,
) -> FrameContext:
    ctx = FrameContext(frame=_FRAME, event_time=_T0, capture_time=_T0)
    ctx.image = np.zeros((480, 640, 3), dtype=np.uint8)
    ctx.effective_width = 640
    ctx.effective_height = 480
    ctx.raw_detections = dets or []
    ctx._detection_ids = {i: f"det-{i}" for i in range(len(ctx.raw_detections))}
    ctx._floor_points_by_index = floor_points or {}
    ctx.face_anchors = face_anchors or []
    return ctx


def _det(x_off: float = 0.0) -> DetectionBox:
    return DetectionBox(x1=0.1 + x_off, y1=0.1, x2=0.3 + x_off, y2=0.6, confidence=0.85)


def _fp(x_mm: float = 1000.0) -> FloorPoint:
    return FloorPoint(x_mm=x_mm, y_mm=1000.0, calibrated=True)


def _ph(
    ph_id: str = "ph-1",
    x_m: float = 1.0,
    committed_seconds_ago: float = 5.0,
) -> PersonHypothesis:
    return PersonHypothesis(
        ph_id=ph_id,
        state_mean=(x_m, 1.0, 0.0, 0.0),
        state_cov=(1.0,) * 16,
        born_at=_T0 - timedelta(seconds=60),
        last_seen_at=_T0 - timedelta(seconds=1),
        last_seen_camera="cam-1",
        observation_count=10,
        current_identity_id="alice",
        current_identity_committed_at=_T0 - timedelta(seconds=committed_seconds_ago),
    )


def _fake_reid(embedding_value: float = 0.5) -> AsyncMock:
    embedding = np.full(768, embedding_value, dtype=np.float32)
    mock = AsyncMock()
    mock.embed_batch = AsyncMock(return_value=[embedding])
    return mock


def _fake_tracker(phs: list[PersonHypothesis]) -> MagicMock:
    tracker = MagicMock()
    tracker.last_open_phs = phs
    return tracker


def _live_policy() -> AdaptiveReidConfig:
    return AdaptiveReidConfig(
        enabled=True,
        shadow=False,
        refresh_interval_s=_REFRESH_S,
        proximity_gate_m=_PROXIMITY_M,
    )


# ---------------------------------------------------------------------------
# Test 1: two-person frame always embeds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_person_frame_always_embeds() -> None:
    """Multi-detect trigger: embed_batch is called even in steady state."""
    reid = _fake_reid()
    ph1 = _ph("ph-1", x_m=1.0)
    ph2 = _ph("ph-2", x_m=3.0)
    tracker = _fake_tracker([ph1, ph2])

    policy = ReidNeedPolicy(config=_live_policy(), prior_maintenance_max_age_s=120.0)
    # Seed both PHs as recently embedded.
    policy._last_embed_at_by_ph = {
        "ph-1": _T0 - timedelta(seconds=2),
        "ph-2": _T0 - timedelta(seconds=2),
    }

    # Two detections per the two-person frame.
    reid.embed_batch = AsyncMock(
        return_value=[
            np.full(768, 0.5, dtype=np.float32),
            np.full(768, 0.6, dtype=np.float32),
        ]
    )

    stage = InferenceStage(
        reid_embedder=reid,
        reid_policy=policy,
        world_tracker=tracker,
    )
    ctx = _ctx(
        dets=[_det(0.0), _det(0.4)],
        floor_points={0: _fp(1000), 1: _fp(3000)},
    )
    await stage.run(ctx)

    reid.embed_batch.assert_called_once()
    assert len(ctx.embeddings) == 2


# ---------------------------------------------------------------------------
# Test 2: single-resolved-PH steady state skips after first embed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_resident_steady_state_skips() -> None:
    """After first embed, the next call within refresh_interval skips."""
    reid = _fake_reid()
    ph = _ph("ph-1", x_m=1.0)
    tracker = _fake_tracker([ph])

    policy = ReidNeedPolicy(config=_live_policy(), prior_maintenance_max_age_s=120.0)

    stage = InferenceStage(
        reid_embedder=reid,
        reid_policy=policy,
        world_tracker=tracker,
    )

    # Frame 1: refresh interval not yet elapsed → embed.
    ctx1 = _ctx(dets=[_det()], floor_points={0: _fp(1000)})
    await stage.run(ctx1)
    reid.embed_batch.assert_called_once()
    assert len(ctx1.embeddings) == 1

    # Simulate the policy recording the embed (done inside InferenceStage).
    # Now advance time within refresh interval.
    reid.embed_batch.reset_mock()

    # Frame 2: within refresh_interval → skip.
    stage_inner = InferenceStage(
        reid_embedder=reid,
        reid_policy=policy,
        world_tracker=tracker,
    )
    ctx2 = FrameContext(
        frame=_FRAME,
        event_time=_T0 + timedelta(seconds=_REFRESH_S - 1),
        capture_time=_T0,
    )
    ctx2.image = np.zeros((480, 640, 3), dtype=np.uint8)
    ctx2.effective_width = 640
    ctx2.effective_height = 480
    ctx2.raw_detections = [_det()]
    ctx2._detection_ids = {0: "det-0"}
    ctx2._floor_points_by_index = {0: _fp(1000)}
    ctx2.face_anchors = []

    await stage_inner.run(ctx2)
    reid.embed_batch.assert_not_called()
    assert len(ctx2.embeddings) == 0


# ---------------------------------------------------------------------------
# Test 3: recognized face anchor always embeds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_face_anchor_frame_embeds() -> None:
    """Recognized face anchor overrides steady-state skip."""
    reid = _fake_reid()
    ph = _ph("ph-1", x_m=1.0)
    tracker = _fake_tracker([ph])

    policy = ReidNeedPolicy(config=_live_policy(), prior_maintenance_max_age_s=120.0)
    policy._last_embed_at_by_ph = {"ph-1": _T0 - timedelta(seconds=2)}

    stage = InferenceStage(
        reid_embedder=reid,
        reid_policy=policy,
        world_tracker=tracker,
    )
    anchor = FaceAnchor(
        person_id="alice",
        camera_id="cam-1",
        detection_id="det-0",
        recognition_state="recognized",
        confidence=0.92,
        captured_at=_T0,
    )
    ctx = _ctx(dets=[_det()], floor_points={0: _fp(1000)}, face_anchors=[anchor])
    await stage.run(ctx)

    reid.embed_batch.assert_called_once()
    assert len(ctx.embeddings) == 1


# ---------------------------------------------------------------------------
# Test 4: empty-embedding tolerance — PH update path handles embedding=[]
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_embedding_inference_no_error() -> None:
    """When embedding is skipped, domain detections carry embedding=[] without error."""
    reid = _fake_reid()
    ph = _ph("ph-1", x_m=1.0)
    tracker = _fake_tracker([ph])

    policy = ReidNeedPolicy(config=_live_policy(), prior_maintenance_max_age_s=120.0)
    # Force steady-state skip by seeding a recent embed.
    policy._last_embed_at_by_ph = {"ph-1": _T0 - timedelta(seconds=2)}

    stage = InferenceStage(
        reid_embedder=reid,
        reid_policy=policy,
        world_tracker=tracker,
    )
    ctx = _ctx(dets=[_det()], floor_points={0: _fp(1000)})
    await stage.run(ctx)

    reid.embed_batch.assert_not_called()
    # Domain detection must exist with empty embedding.
    assert len(ctx.domain_detections) == 1
    assert ctx.domain_detections[0].embedding == []

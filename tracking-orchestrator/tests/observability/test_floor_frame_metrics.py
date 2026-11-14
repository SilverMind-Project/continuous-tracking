"""Metrics that make floor-frame calibration state visible (IR-S04 task 1).

Each test swaps in a fresh registry so sample values are exact.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from prometheus_client import CollectorRegistry

from app.domain import BoundingBox, Detection, FloorPoint, OverlapGroup, WorldObservation
from app.observability import metrics as _metrics
from app.observability.metrics import build_metrics
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.world_tracking import WorldTrackingStage
from app.pipeline.types import LiveConfigHolder
from app.services.camera_room_map import CameraRoomMap, RoomPolygonMap
from app.storage.base import InMemoryPHRepository, InMemoryWorldObservationRepository
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.tracker import WorldTracker, WorldTrackerResult
from app.transport.redis_streams import FrameReady

_NOW = datetime(2026, 11, 14, 10, 0, 0, tzinfo=UTC)
_ROOM_POLYGONS = {"room": [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]}


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> CollectorRegistry:
    reg = CollectorRegistry()
    monkeypatch.setattr(_metrics, "metrics", build_metrics(registry=reg))
    return reg


def _obs(
    camera_id: str,
    x_m: float,
    detection_id: str,
    *,
    calibrated: bool = True,
    embedding: list[float] | None = None,
) -> WorldObservation:
    return WorldObservation(
        camera_id=camera_id,
        frame_index=1,
        captured_at=_NOW,
        floor_point=FloorPoint(x_mm=int(x_m * 1000), y_mm=5000, calibrated=calibrated),
        bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=200),
        embedding=embedding if embedding is not None else [0.9, 0.1, 0.0],
        detection_confidence=0.92,
        detection_id=detection_id,
        quality=0.6,
    )


def _tracker(cfg: WorldTrackerConfig) -> WorldTracker:
    return WorldTracker(
        ph_repo=InMemoryPHRepository(),
        obs_repo=InMemoryWorldObservationRepository(),
        config=cfg,
    )


@pytest.mark.asyncio
async def test_pair_distance_histogram_observes_cross_camera_merge(
    registry: CollectorRegistry,
) -> None:
    cfg = WorldTrackerConfig(
        dedup_enabled=True, dedup_max_distance_m=0.9, min_observations_to_publish=1
    )
    observations = [_obs("cam-a", 5.0, "d1"), _obs("cam-b", 5.4, "d2")]

    await _tracker(cfg).step(observations, now=_NOW, room_polygons=_ROOM_POLYGONS)

    name = "cts_worldtracker_dedup_pair_distance_m"
    assert registry.get_sample_value(f"{name}_count") == 1.0
    assert registry.get_sample_value(f"{name}_sum") == pytest.approx(0.4)
    assert registry.get_sample_value(f"{name}_bucket", {"le": "0.3"}) == 0.0
    assert registry.get_sample_value(f"{name}_bucket", {"le": "0.5"}) == 1.0


@pytest.mark.asyncio
async def test_pair_distance_histogram_ignores_pairs_outside_the_gate(
    registry: CollectorRegistry,
) -> None:
    cfg = WorldTrackerConfig(
        dedup_enabled=True,
        dedup_max_distance_m=0.3,
        dedup_max_distance_ceiling_m=0.3,
        min_observations_to_publish=1,
    )
    observations = [_obs("cam-a", 5.0, "d1"), _obs("cam-b", 5.4, "d2")]

    await _tracker(cfg).step(observations, now=_NOW, room_polygons=_ROOM_POLYGONS)

    assert registry.get_sample_value("cts_worldtracker_dedup_pair_distance_m_count") == 0.0


@pytest.mark.asyncio
async def test_group_appearance_counter_counts_merged_pairs(registry: CollectorRegistry) -> None:
    cfg = WorldTrackerConfig(
        dedup_enabled=True,
        enable_group_appearance_dedup=True,
        dedup_group_appearance_min_sim=0.75,
        min_observations_to_publish=1,
    )
    tracker = WorldTracker(
        ph_repo=InMemoryPHRepository(),
        obs_repo=InMemoryWorldObservationRepository(),
        config=cfg,
        overlap_groups=[OverlapGroup(group_id="1", camera_ids=("cam-a", "cam-b"))],
    )
    same_look = [0.9, 0.1, 0.0]
    observations = [
        _obs("cam-a", 1.0, "d1", calibrated=False, embedding=same_look),
        _obs("cam-b", 9.0, "d2", calibrated=False, embedding=same_look),
    ]

    await tracker.step(observations, now=_NOW, room_polygons=_ROOM_POLYGONS)

    assert registry.get_sample_value("cts_worldtracker_group_appearance_dedup_total") == 1.0
    assert registry.get_sample_value("cts_worldtracker_dedup_clusters_total") == 1.0


def _context(camera_id: str, detections: list[Detection]) -> FrameContext:
    frame = FrameReady(
        camera_id=camera_id,
        minio_key="",
        frame_index=1,
        capture_time_unix_ns=int(_NOW.timestamp() * 1e9),
        received_time_unix_ns=int(_NOW.timestamp() * 1e9),
        width=640,
        height=480,
    )
    ctx = FrameContext(
        frame=frame,
        event_time=_NOW,
        capture_time=_NOW,
        effective_width=640,
        effective_height=480,
    )
    ctx.domain_detections = detections
    return ctx


def _detection(detection_id: str, camera_id: str, *, calibrated: bool) -> Detection:
    return Detection(
        detection_id=detection_id,
        camera_id=camera_id,
        bbox=BoundingBox(x_min=100, y_min=50, x_max=250, y_max=400),
        confidence=0.9,
        floor_point=FloorPoint(3000, 4000, calibrated=calibrated),
        embedding=[],
        capture_time=_NOW,
        event_time=_NOW,
    )


@pytest.mark.asyncio
async def test_observations_counted_by_calibration_per_camera(
    registry: CollectorRegistry,
) -> None:
    tracker = MagicMock()
    tracker.step = AsyncMock(
        return_value=WorldTrackerResult(updated_phs=[], snapshots=[], continuations=[])
    )
    stage = WorldTrackingStage(
        tracker=tracker,
        live_config=LiveConfigHolder(CameraRoomMap(), RoomPolygonMap()),
    )
    contexts = [
        _context(
            "cam-cal",
            [
                _detection("d1", "cam-cal", calibrated=True),
                _detection("d2", "cam-cal", calibrated=True),
            ],
        ),
        _context("cam-virtual", [_detection("d3", "cam-virtual", calibrated=False)]),
    ]

    await stage.run_many(contexts)

    name = "cts_worldtracker_observations_by_calibration_total"
    assert registry.get_sample_value(name, {"camera_id": "cam-cal", "calibrated": "true"}) == 2.0
    assert registry.get_sample_value(name, {"camera_id": "cam-virtual", "calibrated": "false"}) == (
        1.0
    )
    assert registry.get_sample_value(name, {"camera_id": "cam-cal", "calibrated": "false"}) is None

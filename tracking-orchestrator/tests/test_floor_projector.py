"""Tests for FloorProjector and cross-camera geometric scoring."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from app.calibration.state import CalibrationState
from app.domain import BoundingBox, FloorPoint, Tracklet
from app.storage.base import InMemoryGalleryRepository, InMemoryGlobalTrackRepository
from app.tracking.camera_adjacency import AdjacencyEdge, CameraAdjacency
from app.tracking.cross_camera import CrossCamConfig, CrossCameraAssociator
from app.tracking.floor_projector import FloorProjector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity_h() -> list[list[float]]:
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _scale_h(sx: float, sy: float) -> list[list[float]]:
    return [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]]


def _state_with(*camera_ids: str) -> CalibrationState:
    """Return a CalibrationState with identity homographies pre-set."""
    state = CalibrationState()
    for cam in camera_ids:
        state.homographies[cam] = _identity_h()
    return state


def _make_adjacency() -> CameraAdjacency:
    adj = CameraAdjacency()
    adj.add_edge(AdjacencyEdge("cam_a", "cam_b", max_transition_seconds=120))
    return adj


def _make_tracklet(
    tracklet_id: str,
    camera_id: str,
    bbox: BoundingBox | None = None,
    floor_point: FloorPoint | None = None,
) -> Tracklet:
    now = datetime.now(UTC)
    return Tracklet(
        tracklet_id=tracklet_id,
        camera_id=camera_id,
        detection_ids=[f"det-{tracklet_id}"],
        started_at=now,
        last_bbox=bbox,
        last_floor_point=floor_point,
    )


def _assoc(state: CalibrationState, config: CrossCamConfig | None = None) -> CrossCameraAssociator:
    return CrossCameraAssociator(
        gallery=InMemoryGalleryRepository(),
        adjacency=_make_adjacency(),
        global_track_repo=InMemoryGlobalTrackRepository(),
        config=config or CrossCamConfig(floor_sigma_m=1.5, max_floor_distance_m=8.0),
        floor_projector=FloorProjector(state),
    )


# ---------------------------------------------------------------------------
# FloorProjector unit tests
# ---------------------------------------------------------------------------


class TestFloorProjector:
    def test_no_homography_returns_uncalibrated(self) -> None:
        proj = FloorProjector(CalibrationState())
        fp = proj.project("cam_a", BoundingBox(100, 200, 200, 400))
        assert fp.calibrated is False
        assert fp.x_mm == 0
        assert fp.y_mm == 0

    def test_identity_homography_footpoint(self) -> None:
        state = _state_with("cam_a")
        proj = FloorProjector(state)
        # footpoint = bottom-centre: x=(100+200)/2=150, y=400
        fp = proj.project("cam_a", BoundingBox(100, 200, 200, 400))
        assert fp.calibrated is True
        assert fp.x_mm == 150
        assert fp.y_mm == 400

    def test_scale_homography(self) -> None:
        state = CalibrationState()
        state.homographies["cam_a"] = _scale_h(10.0, 10.0)
        proj = FloorProjector(state)
        # footpoint px: (20, 50) → floor mm: (200, 500)
        fp = proj.project("cam_a", BoundingBox(10, 20, 30, 50))
        assert fp.calibrated is True
        assert fp.x_mm == 200
        assert fp.y_mm == 500

    def test_unknown_camera_returns_uncalibrated(self) -> None:
        proj = FloorProjector(_state_with("cam_a"))
        fp = proj.project("cam_unknown", BoundingBox(0, 0, 100, 100))
        assert fp.calibrated is False

    def test_distance_m_zero(self) -> None:
        fp = FloorPoint(x_mm=1000, y_mm=2000, calibrated=True)
        assert FloorProjector.distance_m(fp, fp) == pytest.approx(0.0)

    def test_distance_m_pythagorean(self) -> None:
        # 3 m, 4 m → 5 m
        a = FloorPoint(x_mm=0, y_mm=0, calibrated=True)
        b = FloorPoint(x_mm=3000, y_mm=4000, calibrated=True)
        assert FloorProjector.distance_m(a, b) == pytest.approx(5.0)

    def test_hot_reload_picks_up_new_homography(self) -> None:
        state = CalibrationState()
        proj = FloorProjector(state)
        bbox = BoundingBox(0, 0, 100, 200)
        assert proj.project("cam_a", bbox).calibrated is False
        state.homographies["cam_a"] = _identity_h()
        assert proj.project("cam_a", bbox).calibrated is True


# ---------------------------------------------------------------------------
# CrossCameraAssociator geo-score integration tests
# ---------------------------------------------------------------------------


class TestGeoScore:
    def test_no_projector_returns_1(self) -> None:
        """Without a FloorProjector, geo_score is 1.0 (binary gate)."""
        assoc = CrossCameraAssociator(
            gallery=InMemoryGalleryRepository(),
            adjacency=_make_adjacency(),
            global_track_repo=InMemoryGlobalTrackRepository(),
            floor_projector=None,
        )
        ta = _make_tracklet("t1", "cam_a")
        tb = _make_tracklet("t2", "cam_b")
        assert assoc._geo_score(ta, tb) == pytest.approx(1.0)

    def test_no_bbox_falls_back_to_1(self) -> None:
        state = _state_with("cam_a", "cam_b")
        a = _assoc(state)
        ta = _make_tracklet("t1", "cam_a", bbox=None)
        tb = _make_tracklet("t2", "cam_b", bbox=None)
        assert a._geo_score(ta, tb) == pytest.approx(1.0)

    def test_uncalibrated_camera_falls_back_to_1(self) -> None:
        """Camera without a homography yields uncalibrated FloorPoint → fallback."""
        a = _assoc(CalibrationState())  # no homographies
        bbox = BoundingBox(0, 0, 100, 100)
        ta = _make_tracklet("t1", "cam_a", bbox=bbox)
        tb = _make_tracklet("t2", "cam_b", bbox=bbox)
        assert a._geo_score(ta, tb) == pytest.approx(1.0)

    def test_close_pair_high_geo_score(self) -> None:
        """Nearby floor points produce a geo_score close to 1."""
        state = _state_with("cam_a", "cam_b")
        a = _assoc(state)
        # foot(bbox_a) = (500, 500) mm; foot(bbox_b) = (1000, 800) mm
        bbox_a = BoundingBox(400, 300, 600, 500)
        bbox_b = BoundingBox(900, 300, 1100, 800)
        ta = _make_tracklet("t1", "cam_a", bbox=bbox_a)
        tb = _make_tracklet("t2", "cam_b", bbox=bbox_b)
        score = a._geo_score(ta, tb)
        assert score is not None
        dist_m = math.sqrt((1000 - 500) ** 2 + (800 - 500) ** 2) / 1000.0
        expected = math.exp(-((dist_m / 1.5) ** 2))
        assert score == pytest.approx(expected, rel=1e-4)
        assert score > 0.8  # ~0.58 m separation with sigma=1.5 → score ≈ 0.86

    def test_far_pair_pruned(self) -> None:
        """Pairs beyond max_floor_distance_m return None."""
        state = _state_with("cam_a", "cam_b")
        a = _assoc(state)
        bbox_a = BoundingBox(0, 0, 0, 0)
        bbox_b = BoundingBox(9000, 9000, 9100, 9100)
        ta = _make_tracklet("t1", "cam_a", bbox=bbox_a)
        tb = _make_tracklet("t2", "cam_b", bbox=bbox_b)
        assert a._geo_score(ta, tb) is None

    def test_pre_attached_floor_point_used_directly(self) -> None:
        """last_floor_point is used directly without projection."""
        a = _assoc(CalibrationState())  # no homographies — projection would fail
        fp_a = FloorPoint(x_mm=0, y_mm=0, calibrated=True)
        fp_b = FloorPoint(x_mm=1000, y_mm=0, calibrated=True)  # 1 m
        ta = _make_tracklet("t1", "cam_a", floor_point=fp_a)
        tb = _make_tracklet("t2", "cam_b", floor_point=fp_b)
        score = a._geo_score(ta, tb)
        assert score is not None
        expected = math.exp(-((1.0 / 1.5) ** 2))
        assert score == pytest.approx(expected, rel=1e-4)

    def test_one_calibrated_one_not_falls_back(self) -> None:
        """Only one calibrated floor point → fallback to 1.0."""
        state = _state_with("cam_a")  # cam_b has no homography
        a = _assoc(state)
        bbox = BoundingBox(100, 100, 200, 200)
        ta = _make_tracklet("t1", "cam_a", bbox=bbox)
        tb = _make_tracklet("t2", "cam_b", bbox=bbox)
        assert a._geo_score(ta, tb) == pytest.approx(1.0)

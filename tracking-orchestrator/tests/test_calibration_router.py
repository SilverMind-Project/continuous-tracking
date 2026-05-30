"""Tests for the internal calibration endpoints."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.routers.calibration as _cal_router_mod
from app.calibration.state import CalibrationState
from app.routers.calibration import router as calibration_router


@pytest.fixture
def client(monkeypatch):
    """Minimal app with fresh calibration state — no lifespan, no Redis."""
    fresh = CalibrationState()
    monkeypatch.setattr(_cal_router_mod, "_default_state", fresh)
    app = FastAPI()
    app.include_router(calibration_router)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def state():
    return CalibrationState()


# ---------------------------------------------------------------------------
# Homography
# ---------------------------------------------------------------------------


def test_post_homography_stores_matrix(client: TestClient):
    matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    resp = client.post(
        "/internal/calibration/homography",
        json={"camera_id": "cam-1", "matrix": matrix},
    )
    assert resp.status_code == 204

    status_resp = client.get("/internal/calibration/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["cameras_with_homography"] == 1


def test_post_homography_invalid_matrix_shape(client: TestClient):
    resp = client.post(
        "/internal/calibration/homography",
        json={"camera_id": "cam-1", "matrix": [[1.0, 0.0], [0.0, 1.0]]},
    )
    assert resp.status_code == 422


def test_auto_calibrate_returns_draft_without_storing_homography(client: TestClient, monkeypatch):
    class FakeFetcher:
        async def fetch_rgb(self, key: str):
            assert key == "frames/cam-1/1.jpg"
            return np.zeros((480, 640, 3), dtype=np.uint8)

    class FakeAutoCalibrator:
        async def calibrate(self, image, fov_deg: float):
            assert image.shape == (480, 640, 3)
            return SimpleNamespace(
                draft_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                suggested_points=[
                    {"pixel": [100.0, 300.0], "local_floor_m": [-0.5, 0.2]},
                    {"pixel": [300.0, 420.0], "local_floor_m": [0.4, 1.1]},
                ],
                confidence=0.8,
                inlier_count=200,
                sample_count=400,
                fov_deg=fov_deg,
                method="depth_auto_draft",
            )

    monkeypatch.setattr(_cal_router_mod, "_auto_calibrator", FakeAutoCalibrator())
    monkeypatch.setattr(_cal_router_mod, "_frame_fetcher", FakeFetcher())

    resp = client.post(
        "/internal/calibration/auto/cam-1",
        json={"minio_key": "frames/cam-1/1.jpg", "fov_deg": 70.0},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["method"] == "depth_auto_draft"
    assert body["draft_matrix"][0][0] == 1.0
    assert len(body["suggested_points"]) == 2
    assert body["image_width"] == 640
    assert body["image_height"] == 480

    status_resp = client.get("/internal/calibration/status")
    assert status_resp.json()["cameras_with_homography"] == 0


# ---------------------------------------------------------------------------
# Privacy zones
# ---------------------------------------------------------------------------

VALID_ZONE = {
    "zone_id": "z1",
    "name": "Bathroom",
    "polygon": [[0.0, 0.3], [1.0, 0.3], [1.0, 1.0], [0.0, 1.0]],
    "policy": "mask_region",
    "enabled": True,
}


def test_post_privacy_zones_accepted(client: TestClient):
    resp = client.post(
        "/internal/calibration/privacy_zones",
        json={"camera_id": "cam-2", "zones": [VALID_ZONE]},
    )
    assert resp.status_code == 204

    status_resp = client.get("/internal/calibration/status")
    assert status_resp.json()["cameras_with_privacy_zones"] == 1


def test_post_privacy_zones_invalid_policy_rejected(client: TestClient):
    zone = {**VALID_ZONE, "policy": "delete_everything"}
    resp = client.post(
        "/internal/calibration/privacy_zones",
        json={"camera_id": "cam-2", "zones": [zone]},
    )
    assert resp.status_code == 422


def test_post_privacy_zones_coordinates_out_of_range_rejected(client: TestClient):
    zone = {**VALID_ZONE, "polygon": [[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]]}
    resp = client.post(
        "/internal/calibration/privacy_zones",
        json={"camera_id": "cam-2", "zones": [zone]},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Adjacency
# ---------------------------------------------------------------------------


def test_post_adjacency_accepted(client: TestClient):
    resp = client.post(
        "/internal/calibration/camera_adjacency",
        json={
            "edges": [
                {"from": "hallway", "to": "kitchen", "min_transit_s": 1.0, "max_transit_s": 20.0}
            ]
        },
    )
    assert resp.status_code == 204

    status_resp = client.get("/internal/calibration/status")
    assert status_resp.json()["adjacency_edge_count"] == 1


def test_post_adjacency_inverted_transit_rejected(client: TestClient):
    resp = client.post(
        "/internal/calibration/camera_adjacency",
        json={"edges": [{"from": "a", "to": "b", "min_transit_s": 30.0, "max_transit_s": 1.0}]},
    )
    assert resp.status_code == 422


def test_post_adjacency_defaults_applied(client: TestClient):
    resp = client.post(
        "/internal/calibration/camera_adjacency",
        json={"edges": [{"from": "a", "to": "b"}]},
    )
    assert resp.status_code == 204


# ---------------------------------------------------------------------------
# Reload
# ---------------------------------------------------------------------------


def test_reload_updates_timestamp(client: TestClient):
    client.post("/internal/calibration/reload")
    status_resp = client.get("/internal/calibration/status")
    assert status_resp.json()["last_reload_at"] is not None


# ---------------------------------------------------------------------------
# CalibrationState unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_state_set_homography():
    s = CalibrationState()
    matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    await s.set_homography("cam-x", matrix)
    assert "cam-x" in s.homographies
    assert s.homographies["cam-x"] == matrix


@pytest.mark.asyncio
async def test_state_set_privacy_zones():
    from app.calibration.state import PrivacyZoneConfig

    s = CalibrationState()
    zones = [PrivacyZoneConfig(zone_id="z1", polygon=[[0, 0], [1, 0], [1, 1]], policy="blur_faces")]
    await s.set_privacy_zones("cam-y", zones)
    assert len(s.privacy_zones["cam-y"]) == 1


@pytest.mark.asyncio
async def test_state_set_adjacency():
    from app.calibration.state import AdjacencyEdge

    s = CalibrationState()
    edges = [AdjacencyEdge(from_camera="a", to_camera="b")]
    await s.set_adjacency(edges)
    assert len(s.adjacency_edges) == 1

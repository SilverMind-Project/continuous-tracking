"""CTS reads each camera's face-id settings from cognitive-companion's camera list.

The per-camera raw-similarity floor arrives as ``face_id_min_raw_similarity``; it is usually unset,
which means no per-camera floor. CTS reads no other name for it.
"""

from __future__ import annotations

from typing import Any

from app.main import _fetch_cc_camera_configs
from app.pipeline.types import FaceIdCameraConfig
from app.services.cc_client import CognitiveCompanionClient


class _CameraListClient(CognitiveCompanionClient):
    def __init__(self, cameras: list[dict[str, Any]]) -> None:
        super().__init__("http://cc.test")
        self._cameras = cameras

    async def get(self, path: str) -> Any:
        assert path == "/api/v1/cts/cameras"
        return self._cameras


async def test_camera_floor_is_read_from_the_raw_similarity_field() -> None:
    client = _CameraListClient(
        [
            {"id": "cam-floor", "face_id_enabled": True, "face_id_min_raw_similarity": 0.55},
            {"id": "cam-none", "face_id_enabled": True, "face_id_min_raw_similarity": None},
            {"id": "cam-off", "face_id_enabled": False},
        ]
    )

    configs = await _fetch_cc_camera_configs(client)

    assert configs == {
        "cam-floor": FaceIdCameraConfig(enabled=True, min_raw_similarity=0.55),
        "cam-none": FaceIdCameraConfig(enabled=True, min_raw_similarity=None),
        "cam-off": FaceIdCameraConfig(enabled=False, min_raw_similarity=None),
    }


async def test_the_old_confidence_field_name_is_not_read() -> None:
    # A CC that predates the rename sends the old name; CTS treats that as "no per-camera floor".
    client = _CameraListClient(
        [{"id": "cam-old", "face_id_enabled": True, "face_id_min_confidence": 0.7}]
    )

    configs = await _fetch_cc_camera_configs(client)

    assert configs["cam-old"].min_raw_similarity is None

"""Thread-safe calibration state store for the orchestrator.

The CC backend pushes homography matrices, privacy zones, and the adjacency
graph here via /internal/calibration/* endpoints.  The pipeline reads from
this store on every frame so updates take effect without restart.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class PrivacyZoneConfig:
    zone_id: str
    polygon: list[list[float]]
    policy: str  # drop_detections | blur_faces | mask_region
    name: str = ""
    enabled: bool = True


@dataclass
class AdjacencyEdge:
    from_camera: str
    to_camera: str
    min_transit_s: float = 0.5
    max_transit_s: float = 30.0
    overlap: bool = False


@dataclass
class CalibrationState:
    """In-memory calibration store.  All mutations are protected by an asyncio lock."""

    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    # camera_id -> 3x3 homography matrix (row-major)
    homographies: dict[str, list[list[float]]] = field(default_factory=dict)

    # camera_id -> list of privacy zones
    privacy_zones: dict[str, list[PrivacyZoneConfig]] = field(default_factory=dict)

    # list of directed adjacency edges
    adjacency_edges: list[AdjacencyEdge] = field(default_factory=list)

    # ISO-8601 wall-clock of the last successful reload
    last_reload_at: str | None = None

    # arbitrary metadata per camera (e.g. reprojection error)
    camera_meta: dict[str, dict[str, Any]] = field(default_factory=dict)

    async def set_homography(
        self,
        camera_id: str,
        matrix: list[list[float]],
        meta: dict[str, Any] | None = None,
    ) -> None:
        async with self._lock:
            self.homographies[camera_id] = matrix
            if meta:
                self.camera_meta.setdefault(camera_id, {}).update(meta)

    async def set_privacy_zones(
        self,
        camera_id: str,
        zones: list[PrivacyZoneConfig],
    ) -> None:
        async with self._lock:
            self.privacy_zones[camera_id] = zones

    async def set_adjacency(self, edges: list[AdjacencyEdge]) -> None:
        async with self._lock:
            self.adjacency_edges = edges

    async def reload(self) -> None:
        """Mark the last reload time.  Callers may extend this for filesystem re-reads."""
        async with self._lock:
            self.last_reload_at = datetime.now(UTC).isoformat()

    def snapshot(self) -> dict[str, Any]:
        return {
            "homographies": dict(self.homographies),
            "privacy_zones": {
                k: [{"zone_id": z.zone_id, "policy": z.policy, "enabled": z.enabled} for z in v]
                for k, v in self.privacy_zones.items()
            },
            "adjacency_edge_count": len(self.adjacency_edges),
            "last_reload_at": self.last_reload_at,
        }


# Module-level singleton shared by the FastAPI app.
calibration_state = CalibrationState()

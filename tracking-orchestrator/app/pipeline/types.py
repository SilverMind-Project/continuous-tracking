"""Shared pipeline types imported by both ``frame_pipeline`` and ``stages/``.

These live in their own module to avoid a circular import between the
pipeline orchestrator and the individual stage classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import numpy as np
import numpy.typing as npt

from ..domain import RoomTransitionEvent, TransitZone
from ..inference.schemas import Embedding
from ..services.camera_room_map import CameraRoomMap, RoomPolygonMap
from ..services.transit_zone_map import TransitZoneMap


class FrameImageFetcher(Protocol):
    """Loads an RGB frame image from object storage."""

    async def fetch_rgb(self, minio_key: str) -> npt.NDArray[np.uint8]:
        """Return an RGB uint8 image for the object key."""


class ReidEmbedderProtocol(Protocol):
    """Appearance embedding boundary used by the pipeline."""

    async def embed_batch(
        self,
        crops: list[npt.NDArray[np.uint8]],
    ) -> list[Embedding]:
        """Return one ReID embedding per crop."""


class TransitDetectorProtocol(Protocol):
    """Detects room-transition events from person-hypothesis movement."""

    def check(
        self,
        ph_id: str,
        floor_x_m: float,
        floor_y_m: float,
        zones: list[TransitZone],
        now: datetime,
    ) -> list[RoomTransitionEvent]:
        """Return transit events produced by the latest position."""
        ...

    def remove_ph(self, ph_id: str) -> None:
        """Discard state for a closed person hypothesis."""
        ...


class RoomTransitionPublisherProtocol(Protocol):
    """Publishes room-transition events."""

    async def publish(
        self,
        event: RoomTransitionEvent,
        identity_id: str | None = None,
    ) -> str | None:
        """Publish one room-transition event."""
        ...


@dataclass
class LiveConfigHolder:
    """CC-synced configuration shared by pipeline stages.

    The pipeline replaces fields as dependencies become available; stages keep
    this holder and read its current fields for every frame.
    """

    camera_room_map: CameraRoomMap
    room_polygon_map: RoomPolygonMap
    transit_detector: TransitDetectorProtocol | None = None
    transit_zone_map: TransitZoneMap | None = None
    room_transition_publisher: RoomTransitionPublisherProtocol | None = None


@dataclass(frozen=True)
class FaceIdCameraConfig:
    """Per-camera face identification configuration.

    If *enabled* is False, face identification is skipped entirely for
    this camera (e.g. top-down surveillance cameras where faces are
    never visible).  If *min_confidence* is not None it overrides the
    global ``face_id_min_confidence`` for this camera.
    """

    enabled: bool = True
    min_confidence: float | None = None

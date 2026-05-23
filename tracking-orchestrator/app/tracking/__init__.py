"""Per-camera tracking, tracklet lifecycle, and identity resolution."""

from __future__ import annotations

from .camera_adjacency import CameraAdjacency
from .tracker import PerCameraTracker
from .tracklet_manager import TrackletManager

__all__ = [
    "CameraAdjacency",
    "PerCameraTracker",
    "TrackletManager",
]

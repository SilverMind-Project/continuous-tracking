"""Per-camera tracking, tracklet lifecycle, and identity resolution."""

from __future__ import annotations

from .camera_adjacency import CameraAdjacency
from .cross_camera import CrossCamConfig, CrossCameraAssociator
from .identity_resolver import IdentityResolver, ResolverConfig
from .tracker import PerCameraTracker
from .tracklet_manager import TrackletManager

__all__ = [
    "CameraAdjacency",
    "CrossCamConfig",
    "CrossCameraAssociator",
    "IdentityResolver",
    "PerCameraTracker",
    "ResolverConfig",
    "TrackletManager",
]

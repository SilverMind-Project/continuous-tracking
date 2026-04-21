"""Per-camera tracking and tracklet lifecycle management."""

from __future__ import annotations

from .tracker import PerCameraTracker
from .tracklet_manager import TrackletManager

__all__ = ["PerCameraTracker", "TrackletManager"]

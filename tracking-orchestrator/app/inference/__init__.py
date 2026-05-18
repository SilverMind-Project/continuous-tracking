"""Triton-backed inference clients for detection, ReID, pose, and depth estimation."""

from __future__ import annotations

from app.inference.depth import DepthEstimator
from app.inference.detector import PersonDetector
from app.inference.pose import PoseEstimator
from app.inference.reid_embedder import ReidEmbedder
from app.inference.schemas import DetectionBox, Embedding, Keypoint, PoseResult
from app.inference.triton_client import TritonClientProtocol, TritonGrpcClient

__all__ = [
    "DepthEstimator",
    "DetectionBox",
    "Embedding",
    "Keypoint",
    "PersonDetector",
    "PoseEstimator",
    "PoseResult",
    "ReidEmbedder",
    "TritonClientProtocol",
    "TritonGrpcClient",
]

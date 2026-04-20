"""Triton-backed inference clients for detection, ReID, and pose estimation."""

from __future__ import annotations

from app.inference.detector import PersonDetector
from app.inference.pose import PoseEstimator
from app.inference.reid_embedder import ReidEmbedder
from app.inference.schemas import DetectionBox, Embedding, Keypoint, PoseResult
from app.inference.triton_client import TritonClientProtocol, TritonGrpcClient

__all__ = [
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

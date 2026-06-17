"""World-coordinate person tracker.

Replaces per-camera BoT-SORT + cross-camera Hungarian with a single
floor-plane Kalman tracker keyed by Person Hypothesis (PH).
"""

from __future__ import annotations

from .association import Assignment as Assignment
from .association import associate as associate
from .config import WorldTrackerConfig as WorldTrackerConfig
from .cost_matrix import GATE_INF as GATE_INF
from .cost_matrix import pair_cost as pair_cost
from .kalman import KalmanState as KalmanState
from .kalman import initialize as initialize
from .kalman import isotropic_cov as isotropic_cov
from .kalman import mahalanobis2_position as mahalanobis2_position
from .kalman import predict as predict
from .kalman import update as update
from .tracker import ContinuationPublisher as ContinuationPublisher
from .tracker import WorldTracker as WorldTracker

__all__ = [
    "GATE_INF",
    "Assignment",
    "ContinuationPublisher",
    "KalmanState",
    "WorldTracker",
    "WorldTrackerConfig",
    "associate",
    "initialize",
    "isotropic_cov",
    "mahalanobis2_position",
    "pair_cost",
    "predict",
    "update",
]

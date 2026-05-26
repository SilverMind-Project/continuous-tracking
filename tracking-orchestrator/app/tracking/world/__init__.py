"""World-coordinate person tracker (M1).

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
from .kalman import mahalanobis2_position as mahalanobis2_position
from .kalman import predict as predict
from .kalman import update as update
from .repository import (
    InMemoryPHRepository as InMemoryPHRepository,
)
from .repository import (
    InMemoryWorldObservationRepository as InMemoryWorldObservationRepository,
)
from .repository import (
    PHObservationRepository as PHObservationRepository,
)
from .repository import (
    PHRepositoryProtocol as PHRepositoryProtocol,
)
from .repository import (
    WorldObservationRepository as WorldObservationRepository,
)
from .tracker import ContinuationPublisher as ContinuationPublisher
from .tracker import WorldTracker as WorldTracker

__all__ = [
    "GATE_INF",
    "Assignment",
    "ContinuationPublisher",
    "InMemoryPHRepository",
    "InMemoryWorldObservationRepository",
    "KalmanState",
    "PHObservationRepository",
    "PHRepositoryProtocol",
    "WorldObservationRepository",
    "WorldTracker",
    "WorldTrackerConfig",
    "associate",
    "initialize",
    "mahalanobis2_position",
    "pair_cost",
    "predict",
    "update",
]

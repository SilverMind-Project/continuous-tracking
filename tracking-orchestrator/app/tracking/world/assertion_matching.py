"""Match CC identity assertions to observations, producing face-anchor evidence.

Used by WorldTrackingStage to convert external identity assertions from
cognitive-companion (recamera VLM, caregiver corrections, etc.) into
FaceAnchors that the Bayesian identity resolver can consume.

Assertions must pass spatial, temporal, and confidence gates before they
become evidence. This prevents a far-away or stale assertion from
contaminating the identity posterior.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ...domain import FaceAnchor, WorldObservation


def match_assertions_to_face_anchors(
    assertions: list[dict[str, Any]],
    observations: list[WorldObservation],
    now: datetime,
    anchor_match_window_s: float = 30.0,
    anchor_match_distance_m: float = 5.0,
    anchor_min_confidence: float = 0.5,
) -> list[FaceAnchor]:
    """Convert matching cached assertions into FaceAnchors.

    An assertion matches an observation when all of these hold:
    - Absolute time delta <= anchor_match_window_s.
    - Floor distance <= anchor_match_distance_m.
    - Camera id matches when the assertion carries one.
    - Confidence >= anchor_min_confidence.

    Each matched (assertion, observation) pair creates one FaceAnchor.
    Only the strongest assertion per observation is kept.
    """
    if not assertions or not observations:
        return []

    anchors: list[FaceAnchor] = []
    for obs in observations:
        obs_x = obs.floor_point.x_mm / 1000.0
        obs_y = obs.floor_point.y_mm / 1000.0

        best: FaceAnchor | None = None
        best_conf = 0.0

        for a in assertions:
            confidence = float(a.get("confidence", 0.0))
            if confidence < anchor_min_confidence:
                continue
            if confidence <= best_conf:
                continue

            a_camera = str(a.get("camera_id", ""))
            if a_camera and a_camera != obs.camera_id:
                continue

            a_time = a.get("captured_at")
            if a_time is None:
                continue
            if isinstance(a_time, str):
                try:
                    a_time = datetime.fromisoformat(a_time)
                except (ValueError, TypeError):
                    continue
            time_delta = abs((now - a_time).total_seconds())
            if time_delta > anchor_match_window_s:
                continue

            # Floor distance: only when assertion carries coordinates.
            fx = a.get("floor_x_m")
            fy = a.get("floor_y_m")
            if fx is not None and fy is not None:
                dist = ((obs_x - float(fx)) ** 2 + (obs_y - float(fy)) ** 2) ** 0.5
                if dist > anchor_match_distance_m:
                    continue

            person_id = str(a.get("person_id", ""))
            if not person_id:
                continue

            best = FaceAnchor(
                person_id=person_id,
                confidence=confidence,
                camera_id=obs.camera_id,
                detection_id=obs.detection_id,
                captured_at=now,
            )
            best_conf = confidence

        if best is not None:
            anchors.append(best)

    return anchors

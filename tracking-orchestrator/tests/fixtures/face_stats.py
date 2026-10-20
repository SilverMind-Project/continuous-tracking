"""Builders for FaceObservationStat rows shared by unit and integration tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.domain import FaceGateOutcome, FaceObservationStat

T0 = datetime(2026, 10, 20, 9, 0, 0, tzinfo=UTC)


def face_stat(
    offset_s: float,
    camera_id: str = "cam-1",
    outcome: FaceGateOutcome = FaceGateOutcome.EMITTED,
) -> FaceObservationStat:
    return FaceObservationStat(
        observed_at=T0 + timedelta(seconds=offset_s),
        camera_id=camera_id,
        detection_id=f"det-{camera_id}-{offset_s}",
        crop_width_px=120,
        crop_height_px=300,
        face_width_px=34.0,
        face_height_px=41.0,
        interocular_px=17.5,
        det_score=0.82,
        yaw_deg=-12.0,
        pitch_deg=-25.0,
        raw_similarity=0.47,
        recognition_state="recognized",
        best_candidate_id="resident-alpha",
        gate_outcome=outcome,
    )

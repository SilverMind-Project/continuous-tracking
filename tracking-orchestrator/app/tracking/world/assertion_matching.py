"""Match CC identity assertions to observations, producing face-anchor evidence.

Used by WorldTrackingStage to convert external identity assertions from
cognitive-companion (reCamera face sightings, caregiver corrections, etc.)
into FaceAnchors that the Bayesian identity resolver can consume.

Assertions must pass spatial, temporal, and confidence gates before they
become evidence. This prevents a far-away, room-mismatched, stale, or
uncalibrated assertion from contaminating the identity posterior.

Spatial gates fail closed (identity-continuity M09): evidence without a
floor point may match at room granularity with a confidence haircut, and
evidence with neither floor point nor room matches nothing. The camera-id
equality gate only applies inside the floor branch, where same-fleet camera
ids are actually meaningful; a reCamera assertion carrying a CC sensor id
would otherwise never pass a CTS camera-id comparison in the room branch.

This module is pure and synchronous: no I/O, no Prometheus calls. Callers
that want match/rejection metrics pass a ``diagnostics`` dict and increment
counters from its contents themselves.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from ...domain import FaceAnchor, WorldObservation

# Rejection reason labels (mirrors cc_assertions_rejected_total{reason=...}).
_REASON_NO_SPATIAL_EVIDENCE = "no_spatial_evidence"
_REASON_ROOM_MISMATCH = "room_mismatch"
_REASON_STALE = "stale"
_REASON_UNCALIBRATED = "uncalibrated"
_REASON_LOW_CONFIDENCE = "low_confidence"

_GATE_FLOOR = "floor"
_GATE_ROOM = "room"


def _bump(diagnostics: dict[str, int] | None, key: str) -> None:
    if diagnostics is None:
        return
    diagnostics[key] = diagnostics.get(key, 0) + 1


def _parse_captured_at(raw: object) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw)
        except (ValueError, TypeError):
            return None
    if isinstance(raw, datetime):
        return raw
    return None


def _evaluate_gate(
    assertion: Mapping[str, Any],
    obs: WorldObservation,
    now: datetime,
    *,
    anchor_match_window_s: float,
    anchor_match_distance_m: float,
    anchor_min_confidence: float,
    camera_room_lookup: Mapping[str, str | None],
    room_match_confidence_scale: float,
    diagnostics: dict[str, int] | None,
) -> tuple[str, float] | None:
    """Evaluate one (assertion, observation) pair.

    Returns ``(gate, effective_confidence)`` on match, or ``None`` on
    rejection (a diagnostics reason is bumped before returning).

    Gate order: confidence, then time, then space. Space is evaluated
    strictest-first: a floor point (with the camera-id equality gate) beats
    room agreement; an assertion with neither position evidence matches
    nothing, never everything.
    """
    raw_confidence = assertion.get("confidence")
    if raw_confidence is None:
        _bump(diagnostics, _REASON_UNCALIBRATED)
        return None
    confidence = float(raw_confidence)
    if confidence < anchor_min_confidence:
        _bump(diagnostics, _REASON_LOW_CONFIDENCE)
        return None

    a_time = _parse_captured_at(assertion.get("captured_at"))
    if a_time is None:
        _bump(diagnostics, _REASON_STALE)
        return None
    if abs((now - a_time).total_seconds()) > anchor_match_window_s:
        _bump(diagnostics, _REASON_STALE)
        return None

    fx = assertion.get("floor_x_m")
    fy = assertion.get("floor_y_m")
    if fx is not None and fy is not None:
        a_camera = str(assertion.get("camera_id") or "")
        if a_camera and a_camera != obs.camera_id:
            _bump(diagnostics, _REASON_NO_SPATIAL_EVIDENCE)
            return None
        obs_x = obs.floor_point.x_mm / 1000.0
        obs_y = obs.floor_point.y_mm / 1000.0
        dist = ((obs_x - float(fx)) ** 2 + (obs_y - float(fy)) ** 2) ** 0.5
        if dist > anchor_match_distance_m:
            _bump(diagnostics, _REASON_NO_SPATIAL_EVIDENCE)
            return None
        return _GATE_FLOOR, confidence

    room_name = assertion.get("room_name")
    if not room_name:
        _bump(diagnostics, _REASON_NO_SPATIAL_EVIDENCE)
        return None
    obs_room = camera_room_lookup.get(obs.camera_id)
    if not obs_room:
        _bump(diagnostics, _REASON_NO_SPATIAL_EVIDENCE)
        return None
    if str(room_name).strip().casefold() != str(obs_room).strip().casefold():
        _bump(diagnostics, _REASON_ROOM_MISMATCH)
        return None
    return _GATE_ROOM, confidence * room_match_confidence_scale


def match_assertions_to_face_anchors(
    assertions: list[dict[str, Any]],
    observations: list[WorldObservation],
    now: datetime,
    anchor_match_window_s: float = 30.0,
    anchor_match_distance_m: float = 5.0,
    anchor_min_confidence: float = 0.5,
    camera_room_lookup: Mapping[str, str | None] | None = None,
    room_match_confidence_scale: float = 0.8,
    default_quality: float = 0.5,
    default_yaw_deg: float = 60.0,
    diagnostics: dict[str, int] | None = None,
) -> list[FaceAnchor]:
    """Convert matching cached assertions into FaceAnchors.

    An assertion matches an observation when all of these hold:
    - Calibrated confidence is present (never raw similarity) and
      >= anchor_min_confidence.
    - Absolute time delta <= anchor_match_window_s.
    - Spatial agreement, strictest evidence first: a floor point within
      anchor_match_distance_m (gated by camera-id equality when the
      assertion carries one; same-fleet ids are only meaningful here) beats
      room-name agreement (case-normalized) against the observation
      camera's room from ``camera_room_lookup`` when no floor point exists.
      An assertion with neither floor point nor room name matches nothing.

    Each matched (assertion, observation) pair creates one FaceAnchor, built
    with the assertion's own captured_at, wire yaw/quality (or the
    configured conservative defaults when absent), and
    ``origin="cc_assertion"``. ``calibrated_confidence`` stays None by
    design so ArcFace authority keeps failing closed structurally on
    external evidence. A room-matched anchor's confidence is scaled by
    room_match_confidence_scale. Only the strongest assertion per
    observation is kept.
    """
    if not assertions or not observations:
        return []

    room_lookup: Mapping[str, str | None] = camera_room_lookup or {}
    anchors: list[FaceAnchor] = []
    for obs in observations:
        best: FaceAnchor | None = None
        best_conf = -1.0
        best_gate: str | None = None

        for a in assertions:
            outcome = _evaluate_gate(
                a,
                obs,
                now,
                anchor_match_window_s=anchor_match_window_s,
                anchor_match_distance_m=anchor_match_distance_m,
                anchor_min_confidence=anchor_min_confidence,
                camera_room_lookup=room_lookup,
                room_match_confidence_scale=room_match_confidence_scale,
                diagnostics=diagnostics,
            )
            if outcome is None:
                continue
            gate, effective_confidence = outcome
            if effective_confidence <= best_conf:
                continue

            person_id = str(a.get("person_id", ""))
            if not person_id:
                continue

            yaw_deg = a.get("yaw_deg")
            quality = a.get("quality")
            captured_at = _parse_captured_at(a.get("captured_at")) or now

            best = FaceAnchor(
                person_id=person_id,
                confidence=effective_confidence,
                quality=float(quality) if quality is not None else default_quality,
                detection_id=obs.detection_id,
                camera_id=obs.camera_id,
                captured_at=captured_at,
                recognition_state="recognized",
                yaw_deg=float(yaw_deg) if yaw_deg is not None else default_yaw_deg,
                calibrated_confidence=None,
                origin="cc_assertion",
            )
            best_conf = effective_confidence
            best_gate = gate

        if best is not None:
            anchors.append(best)
            _bump(diagnostics, f"matched_{best_gate}")

    return anchors

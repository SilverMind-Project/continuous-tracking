"""Unit tests for ReidNeedPolicy (M5.1 adaptive ReID cadence).

Truth-table coverage of all trigger conditions plus boundary cases.
Tests run without any I/O or DB.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import NamedTuple

import pytest

from app.domain import FaceAnchor, FloorPoint, PersonHypothesis
from app.inference.schemas import DetectionBox
from app.pipeline.reid_policy import AdaptiveReidConfig, ReidNeedPolicy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
_REFRESH_S = 10.0
_PROXIMITY_M = 2.0
_PRIOR_MAINTENANCE_S = 120.0

_CFG = AdaptiveReidConfig(
    enabled=True,
    shadow=False,
    refresh_interval_s=_REFRESH_S,
    proximity_gate_m=_PROXIMITY_M,
)


def _policy(**kwargs: object) -> ReidNeedPolicy:
    cfg = AdaptiveReidConfig(
        enabled=True,
        shadow=False,
        refresh_interval_s=_REFRESH_S,
        proximity_gate_m=_PROXIMITY_M,
    )
    return ReidNeedPolicy(
        config=cfg,
        prior_maintenance_max_age_s=_PRIOR_MAINTENANCE_S,
        **kwargs,  # type: ignore[arg-type]
    )


def _det(x1: float = 0.1, y1: float = 0.1, x2: float = 0.3, y2: float = 0.6) -> DetectionBox:
    return DetectionBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.85)


def _fp(x_mm: float = 1000.0, y_mm: float = 1000.0, calibrated: bool = True) -> FloorPoint:
    return FloorPoint(x_mm=x_mm, y_mm=y_mm, calibrated=calibrated)


def _ph(
    ph_id: str = "ph-1",
    x_m: float = 1.0,
    y_m: float = 1.0,
    identity_id: str | None = "alice",
    committed_at: datetime | None = _T0 - timedelta(seconds=5),
) -> PersonHypothesis:
    state_cov = (1.0,) * 16
    return PersonHypothesis(
        ph_id=ph_id,
        state_mean=(x_m, y_m, 0.0, 0.0),
        state_cov=state_cov,
        born_at=_T0 - timedelta(seconds=60),
        last_seen_at=_T0 - timedelta(seconds=1),
        last_seen_camera="cam-1",
        observation_count=10,
        current_identity_id=identity_id,
        current_identity_committed_at=committed_at,
    )


def _face_anchor(recognized: bool = True) -> FaceAnchor:
    return FaceAnchor(
        person_id="alice" if recognized else "",
        camera_id="cam-1",
        detection_id="det-1",
        recognition_state="recognized" if recognized else "unrecognized",
        confidence=0.9 if recognized else 0.1,
        captured_at=_T0,
    )


# ---------------------------------------------------------------------------
# Parametrized truth table
# ---------------------------------------------------------------------------


class _Case(NamedTuple):
    label: str
    detections: list[DetectionBox]
    floor_points: dict[int, FloorPoint]
    open_phs: list[PersonHypothesis]
    face_anchors: list[FaceAnchor]
    camera_id: str
    now: datetime
    last_embed_at: dict[str, datetime]
    expected_embed: bool
    expected_reason_contains: str


_CASES: list[_Case] = [
    # 1. Multiple detections → always embed.
    _Case(
        label="multi_detect",
        detections=[_det(), _det(x1=0.6)],
        floor_points={0: _fp(1000), 1: _fp(3000)},
        open_phs=[_ph("ph-1", 1.0, 1.0), _ph("ph-2", 3.0, 1.0)],
        face_anchors=[],
        camera_id="cam-1",
        now=_T0,
        last_embed_at={},
        expected_embed=True,
        expected_reason_contains="multi_detect",
    ),
    # 2. Single detection, recognized face anchor → embed for gallery seeding.
    _Case(
        label="face_anchor",
        detections=[_det()],
        floor_points={0: _fp(1000)},
        open_phs=[_ph("ph-1", 1.0, 1.0)],
        face_anchors=[_face_anchor(recognized=True)],
        camera_id="cam-1",
        now=_T0,
        last_embed_at={"ph-1": _T0 - timedelta(seconds=2)},  # fresh, but face anchor overrides
        expected_embed=True,
        expected_reason_contains="face_anchor",
    ),
    # 3. Single detection, unrecognized face anchor → not a trigger.
    _Case(
        label="unrecognized_anchor_does_not_trigger",
        detections=[_det()],
        floor_points={0: _fp(1000)},
        open_phs=[_ph("ph-1", 1.0, 1.0)],
        face_anchors=[_face_anchor(recognized=False)],
        camera_id="cam-1",
        now=_T0,
        last_embed_at={"ph-1": _T0 - timedelta(seconds=2)},
        expected_embed=False,
        expected_reason_contains="steady_state",
    ),
    # 4. Single detection, no floor point → embed (can't check proximity).
    _Case(
        label="no_floor_point",
        detections=[_det()],
        floor_points={},  # no floor point
        open_phs=[_ph("ph-1", 1.0, 1.0)],
        face_anchors=[],
        camera_id="cam-1",
        now=_T0,
        last_embed_at={},
        expected_embed=True,
        expected_reason_contains="no_floor_point",
    ),
    # 5. Single detection, 0 nearby PHs → embed (potential spawn).
    _Case(
        label="no_nearby_ph",
        detections=[_det()],
        floor_points={0: _fp(1000)},  # det at (1m,1m)
        open_phs=[_ph("ph-1", 5.0, 5.0)],  # PH far away
        face_anchors=[],
        camera_id="cam-1",
        now=_T0,
        last_embed_at={},
        expected_embed=True,
        expected_reason_contains="no_nearby_ph",
    ),
    # 6. Single detection, 2 nearby PHs → embed (ambiguous).
    _Case(
        label="multi_nearby_ph",
        detections=[_det()],
        floor_points={0: _fp(1000)},  # det at (1m,1m)
        open_phs=[_ph("ph-1", 1.0, 1.0), _ph("ph-2", 1.2, 1.2)],  # both near
        face_anchors=[],
        camera_id="cam-1",
        now=_T0,
        last_embed_at={},
        expected_embed=True,
        expected_reason_contains="multi_nearby_ph",
    ),
    # 7. Single detection, 1 nearby PH but PH is unresolved (no identity).
    _Case(
        label="ph_unresolved_no_identity",
        detections=[_det()],
        floor_points={0: _fp(1000)},
        open_phs=[_ph("ph-1", 1.0, 1.0, identity_id=None, committed_at=None)],
        face_anchors=[],
        camera_id="cam-1",
        now=_T0,
        last_embed_at={},
        expected_embed=True,
        expected_reason_contains="ph_unresolved",
    ),
    # 8. Single detection, 1 nearby PH with identity but committed_at=None.
    _Case(
        label="ph_resolved_no_committed_at",
        detections=[_det()],
        floor_points={0: _fp(1000)},
        open_phs=[_ph("ph-1", 1.0, 1.0, identity_id="alice", committed_at=None)],
        face_anchors=[],
        camera_id="cam-1",
        now=_T0,
        last_embed_at={},
        expected_embed=True,
        expected_reason_contains="ph_unresolved",
    ),
    # 9. Single detection, 1 nearby PH, identity approaching expiry.
    # committed_at is old: now - committed_at > prior_maintenance_max_age_s - refresh_s
    _Case(
        label="identity_expiring",
        detections=[_det()],
        floor_points={0: _fp(1000)},
        open_phs=[
            _ph(
                "ph-1",
                1.0,
                1.0,
                committed_at=_T0 - timedelta(seconds=_PRIOR_MAINTENANCE_S - _REFRESH_S + 1),
            )
        ],
        face_anchors=[],
        camera_id="cam-1",
        now=_T0,
        last_embed_at={"ph-1": _T0 - timedelta(seconds=2)},
        expected_embed=True,
        expected_reason_contains="identity_expiring",
    ),
    # 10. Single detection, 1 nearby resolved PH, no previous embed → refresh elapsed.
    _Case(
        label="refresh_interval_elapsed_first_time",
        detections=[_det()],
        floor_points={0: _fp(1000)},
        open_phs=[_ph("ph-1", 1.0, 1.0)],
        face_anchors=[],
        camera_id="cam-1",
        now=_T0,
        last_embed_at={},  # never embedded
        expected_embed=True,
        expected_reason_contains="refresh_interval_elapsed",
    ),
    # 11. Single detection, 1 nearby resolved PH, refresh interval just elapsed.
    _Case(
        label="refresh_interval_elapsed_boundary",
        detections=[_det()],
        floor_points={0: _fp(1000)},
        open_phs=[_ph("ph-1", 1.0, 1.0)],
        face_anchors=[],
        camera_id="cam-1",
        now=_T0,
        last_embed_at={"ph-1": _T0 - timedelta(seconds=_REFRESH_S)},
        expected_embed=True,
        expected_reason_contains="refresh_interval_elapsed",
    ),
    # 12. Steady-state skip: 1 detection, 1 resolved PH, fresh embedding.
    _Case(
        label="steady_state_skip",
        detections=[_det()],
        floor_points={0: _fp(1000)},
        open_phs=[_ph("ph-1", 1.0, 1.0)],
        face_anchors=[],
        camera_id="cam-1",
        now=_T0,
        last_embed_at={"ph-1": _T0 - timedelta(seconds=_REFRESH_S - 1)},
        expected_embed=False,
        expected_reason_contains="steady_state",
    ),
    # 13. Detection at proximity gate boundary: exactly at proximity_gate_m → included.
    _Case(
        label="detection_at_gate_boundary",
        detections=[_det()],
        floor_points={0: _fp(1000)},  # det at (1m, 1m)
        open_phs=[_ph("ph-1", 1.0 + _PROXIMITY_M, 1.0)],  # exactly at gate
        face_anchors=[],
        camera_id="cam-1",
        now=_T0,
        last_embed_at={"ph-1": _T0 - timedelta(seconds=_REFRESH_S - 1)},
        expected_embed=False,
        expected_reason_contains="steady_state",
    ),
    # 14. Detection just outside proximity gate → no nearby PH.
    _Case(
        label="detection_outside_gate",
        detections=[_det()],
        floor_points={0: _fp(1000)},  # det at (1m, 1m)
        open_phs=[_ph("ph-1", 1.0 + _PROXIMITY_M + 0.01, 1.0)],
        face_anchors=[],
        camera_id="cam-1",
        now=_T0,
        last_embed_at={},
        expected_embed=True,
        expected_reason_contains="no_nearby_ph",
    ),
    # 15. Overlap group camera → always embed.
    _Case(
        label="overlap_group_camera",
        detections=[_det()],
        floor_points={0: _fp(1000)},
        open_phs=[_ph("ph-1", 1.0, 1.0)],
        face_anchors=[],
        camera_id="cam-overlap",
        now=_T0,
        last_embed_at={"ph-1": _T0 - timedelta(seconds=_REFRESH_S - 1)},
        expected_embed=True,
        expected_reason_contains="overlap_group_camera",
    ),
]


@pytest.mark.parametrize("case", _CASES, ids=[c.label for c in _CASES])
def test_policy_truth_table(case: _Case) -> None:
    overlap_cams: frozenset[str] = (
        frozenset({"cam-overlap"}) if case.label == "overlap_group_camera" else frozenset()
    )
    p = ReidNeedPolicy(
        config=_CFG,
        prior_maintenance_max_age_s=_PRIOR_MAINTENANCE_S,
        overlap_group_cameras=overlap_cams,
    )
    p._last_embed_at_by_ph = dict(case.last_embed_at)

    should_embed, reason, _ = p.should_embed_frame(
        detections=case.detections,
        floor_points=case.floor_points,
        open_phs=case.open_phs,
        face_anchors=case.face_anchors,
        camera_id=case.camera_id,
        now=case.now,
    )

    assert should_embed == case.expected_embed, (
        f"expected should_embed={case.expected_embed}, got {should_embed} (reason={reason!r})"
    )
    assert case.expected_reason_contains in reason, (
        f"expected reason containing {case.expected_reason_contains!r}, got {reason!r}"
    )


# ---------------------------------------------------------------------------
# record_embed and prune
# ---------------------------------------------------------------------------


def test_record_embed_updates_last_embed_at() -> None:
    p = _policy()
    assert "ph-1" not in p._last_embed_at_by_ph
    p.record_embed("ph-1", _T0)
    assert p._last_embed_at_by_ph["ph-1"] == _T0


def test_steady_state_after_record_embed() -> None:
    """After record_embed, the next call within refresh_interval should skip."""
    p = _policy()
    ph = _ph("ph-1", 1.0, 1.0)

    # First call — refresh interval not elapsed (never embedded).
    should_embed, _reason, matched_id = p.should_embed_frame(
        detections=[_det()],
        floor_points={0: _fp(1000)},
        open_phs=[ph],
        face_anchors=[],
        camera_id="cam-1",
        now=_T0,
    )
    assert should_embed
    assert matched_id == "ph-1"
    p.record_embed("ph-1", _T0)

    # Second call — within refresh interval → skip.
    should_embed2, reason2, _ = p.should_embed_frame(
        detections=[_det()],
        floor_points={0: _fp(1000)},
        open_phs=[ph],
        face_anchors=[],
        camera_id="cam-1",
        now=_T0 + timedelta(seconds=_REFRESH_S - 1),
    )
    assert not should_embed2
    assert "steady_state" in reason2


def test_prune_closed_phs() -> None:
    """Closed PHs are removed from the last-embed-at cache."""
    p = _policy()
    p._last_embed_at_by_ph = {"ph-1": _T0, "ph-2": _T0}

    # Only ph-1 is in open_phs; ph-2 should be pruned.
    p.should_embed_frame(
        detections=[_det()],
        floor_points={0: _fp(1000)},
        open_phs=[_ph("ph-1", 1.0, 1.0)],
        face_anchors=[],
        camera_id="cam-1",
        now=_T0 + timedelta(seconds=2),
    )
    assert "ph-1" in p._last_embed_at_by_ph
    assert "ph-2" not in p._last_embed_at_by_ph

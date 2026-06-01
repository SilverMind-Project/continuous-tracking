"""Unit tests for the cross-camera dedup pass (geometric + group-appearance)."""

from __future__ import annotations

from datetime import UTC, datetime

from app.domain import BoundingBox, FaceAnchor, FloorPoint, OverlapGroup, WorldObservation
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.dedup import dedup_observations

_NOW = datetime(2026, 5, 29, 10, 0, 0, tzinfo=UTC)
_CFG = WorldTrackerConfig(dedup_enabled=True, dedup_max_distance_m=0.6)


def _obs(
    camera_id: str,
    x_m: float,
    y_m: float,
    detection_id: str = "",
    face_person_id: str | None = None,
    quality: float = 0.5,
    floor_residual_m: float | None = None,
    calibrated: bool = True,
    embedding: list[float] | None = None,
) -> WorldObservation:
    face: FaceAnchor | None = None
    if face_person_id is not None:
        face = FaceAnchor(
            person_id=face_person_id,
            confidence=0.9,
            quality=1.0,
            tracklet_id="",
            camera_id=camera_id,
        )
    return WorldObservation(
        camera_id=camera_id,
        frame_index=1,
        captured_at=_NOW,
        floor_point=FloorPoint(x_mm=int(x_m * 1000), y_mm=int(y_m * 1000), calibrated=calibrated),
        bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=200),
        embedding=embedding if embedding is not None else [0.9, 0.1, 0.0],
        detection_confidence=0.92,
        detection_id=detection_id,
        face_anchor=face,
        quality=quality,
        floor_residual_m=floor_residual_m,
    )


# two different-camera observations within gate collapse to one representative.
def test_different_cameras_within_gate_collapsed():
    obs1 = _obs("cam-1", 5.0, 5.0, detection_id="d1", quality=0.7)
    obs2 = _obs("cam-2", 5.1, 5.05, detection_id="d2", quality=0.5)
    deduped, cluster_map = dedup_observations([obs1, obs2], _CFG)

    assert len(deduped) == 1, "two overlapping observations should collapse to one"
    rep = deduped[0]
    assert rep.camera_id == "cam-1", "cam-1 has higher quality so it wins"
    assert set(cluster_map[rep.detection_id]) == {"d1", "d2"}


# two SAME-camera observations at the same point are NOT merged.
def test_same_camera_not_merged():
    obs1 = _obs("cam-1", 5.0, 5.0, detection_id="d1")
    obs2 = _obs("cam-1", 5.1, 5.0, detection_id="d2")
    deduped, cluster_map = dedup_observations([obs1, obs2], _CFG)

    assert len(deduped) == 2, "same-camera observations must not be merged"
    assert cluster_map["d1"] == ("d1",)
    assert cluster_map["d2"] == ("d2",)


# two different-camera observations with CONFLICTING committed face ids are NOT merged.
def test_face_conflict_prevents_merge():
    obs1 = _obs("cam-1", 5.0, 5.0, detection_id="d1", face_person_id="alice")
    obs2 = _obs("cam-2", 5.05, 5.0, detection_id="d2", face_person_id="bob")
    deduped, _ = dedup_observations([obs1, obs2], _CFG)

    assert len(deduped) == 2, "conflicting face ids must not be merged"


# two different-camera observations BEYOND dedup_max_distance_m are NOT merged.
def test_beyond_distance_not_merged():
    obs1 = _obs("cam-1", 5.0, 5.0, detection_id="d1")
    obs2 = _obs("cam-2", 6.0, 5.0, detection_id="d2")  # 1.0 m apart, > 0.6 m threshold
    deduped, _ = dedup_observations([obs1, obs2], _CFG)

    assert len(deduped) == 2, "observations beyond threshold must not be merged"


def test_dedup_same_person_merged_with_residual():
    cfg = WorldTrackerConfig(
        dedup_enabled=True,
        dedup_max_distance_m=0.6,
        dedup_residual_coeff_k=1.0,
        dedup_max_distance_ceiling_m=1.5,
    )
    obs1 = _obs("cam-1", 5.0, 5.0, detection_id="d1", quality=0.8, floor_residual_m=0.2)
    obs2 = _obs("cam-2", 5.9, 5.0, detection_id="d2", quality=0.6, floor_residual_m=0.2)

    deduped, cluster_map = dedup_observations([obs1, obs2], cfg)

    assert len(deduped) == 1
    rep = deduped[0]
    assert rep.detection_id == "d1"
    assert set(cluster_map[rep.detection_id]) == {"d1", "d2"}


def test_dedup_different_people_not_merged_with_residual():
    cfg = WorldTrackerConfig(
        dedup_enabled=True,
        dedup_max_distance_m=0.6,
        dedup_residual_coeff_k=1.0,
        dedup_max_distance_ceiling_m=1.5,
    )
    obs1 = _obs(
        "cam-1",
        5.0,
        5.0,
        detection_id="d1",
        face_person_id="alice",
        floor_residual_m=0.2,
    )
    obs2 = _obs(
        "cam-2",
        5.7,
        5.0,
        detection_id="d2",
        face_person_id="bob",
        floor_residual_m=0.2,
    )

    deduped, _ = dedup_observations([obs1, obs2], cfg)

    assert len(deduped) == 2


def test_dedup_ceiling_bounds_widening():
    cfg = WorldTrackerConfig(
        dedup_enabled=True,
        dedup_max_distance_m=0.6,
        dedup_residual_coeff_k=1.0,
        dedup_max_distance_ceiling_m=1.5,
    )
    obs1 = _obs("cam-1", 5.0, 5.0, detection_id="d1", floor_residual_m=10.0)
    obs2 = _obs("cam-2", 6.6, 5.0, detection_id="d2", floor_residual_m=10.0)

    deduped, _ = dedup_observations([obs1, obs2], cfg)

    assert len(deduped) == 2


def test_dedup_skips_uncalibrated():
    obs1 = _obs("cam-1", 5.0, 5.0, detection_id="d1", calibrated=False)
    obs2 = _obs("cam-2", 5.1, 5.0, detection_id="d2", calibrated=False)

    deduped, _ = dedup_observations([obs1, obs2], _CFG)

    assert len(deduped) == 2


# representative selection is deterministic (highest quality wins, ties by camera+det id).
def test_representative_selection_is_deterministic():
    obs_hi = _obs("cam-1", 5.0, 5.0, detection_id="d1", quality=0.8)
    obs_lo = _obs("cam-2", 5.05, 5.0, detection_id="d2", quality=0.3)
    for _ in range(3):
        deduped, _ = dedup_observations([obs_hi, obs_lo], _CFG)
        assert len(deduped) == 1
        assert deduped[0].camera_id == "cam-1"
        assert deduped[0].detection_id == "d1"


def test_representative_selection_tie_broken_by_camera_then_detection():
    obs_a = _obs("cam-a", 5.0, 5.0, detection_id="d2", quality=0.5)
    obs_b = _obs("cam-b", 5.05, 5.0, detection_id="d1", quality=0.5)
    deduped, _ = dedup_observations([obs_a, obs_b], _CFG)
    assert len(deduped) == 1
    assert deduped[0].camera_id == "cam-a"  # "cam-a" < "cam-b" alphabetically


# Dedup disabled: no merging occurs.
def test_dedup_disabled_no_merge():
    cfg = WorldTrackerConfig(dedup_enabled=False)
    obs1 = _obs("cam-1", 5.0, 5.0, detection_id="d1")
    obs2 = _obs("cam-2", 5.05, 5.0, detection_id="d2")
    deduped, _ = dedup_observations([obs1, obs2], cfg)
    assert len(deduped) == 2


# Quality-weighted floor point: representative position is weighted toward the higher-quality obs.
def test_representative_floor_point_is_quality_weighted():
    obs_hi = _obs("cam-1", 5.0, 5.0, detection_id="d1", quality=1.0)
    obs_lo = _obs("cam-2", 5.6, 5.0, detection_id="d2", quality=0.0)
    deduped, _ = dedup_observations([obs_hi, obs_lo], _CFG)
    assert len(deduped) == 1
    # With quality 1.0 vs 1e-6, the weighted mean is very close to obs_hi's position.
    rep = deduped[0]
    assert abs(rep.floor_point.x_mm / 1000.0 - 5.0) < 0.01


# ---------------------------------------------------------------------------
# Group-appearance dedup tests
# ---------------------------------------------------------------------------


_GROUP = OverlapGroup(group_id="g1", name="test", camera_ids=("cam-1", "cam-2"))


def test_group_appearance_dedup_merges_same_perspective() -> None:
    """Within a group, two uncalibrated observations with high sim merge."""
    cfg = WorldTrackerConfig(
        dedup_enabled=True,
        enable_group_appearance_dedup=True,
        dedup_group_appearance_min_sim=0.75,
    )
    obs1 = _obs(
        "cam-1",
        5.0,
        5.0,
        detection_id="d1",
        calibrated=False,
        embedding=[0.9, 0.1, 0.0],
        quality=0.7,
    )
    obs2 = _obs(
        "cam-2",
        5.0,
        5.0,
        detection_id="d2",
        calibrated=False,
        embedding=[0.85, 0.15, 0.0],
        quality=0.5,
    )
    deduped, _cluster_map = dedup_observations([obs1, obs2], cfg, overlap_groups=[_GROUP])
    assert len(deduped) == 1, "same-perspective uncalibrated obs in group should merge"
    rep = deduped[0]
    assert rep.camera_id == "cam-1"  # higher quality wins


def test_group_appearance_dedup_low_sim_does_not_merge() -> None:
    """Within a group, low-appearance uncalibrated observations are not merged."""
    cfg = WorldTrackerConfig(
        dedup_enabled=True,
        enable_group_appearance_dedup=True,
        dedup_group_appearance_min_sim=0.75,
    )
    obs1 = _obs(
        "cam-1",
        5.0,
        5.0,
        detection_id="d1",
        calibrated=False,
        embedding=[1.0, 0.0, 0.0],
        quality=0.7,
    )
    obs2 = _obs(
        "cam-2",
        5.0,
        5.0,
        detection_id="d2",
        calibrated=False,
        embedding=[0.0, 1.0, 0.0],
        quality=0.5,
    )
    deduped, _ = dedup_observations([obs1, obs2], cfg, overlap_groups=[_GROUP])
    assert len(deduped) == 2, "opposite-perspective should not merge at observation level"


def test_group_appearance_dedup_no_group_no_merge() -> None:
    """Uncalibrated observations not in any group never merge."""
    cfg = WorldTrackerConfig(
        dedup_enabled=True,
        enable_group_appearance_dedup=True,
        dedup_group_appearance_min_sim=0.75,
    )
    obs1 = _obs("cam-1", 5.0, 5.0, detection_id="d1", calibrated=False, embedding=[0.9, 0.1, 0.0])
    obs2 = _obs("cam-2", 5.0, 5.0, detection_id="d2", calibrated=False, embedding=[0.9, 0.1, 0.0])
    deduped, _ = dedup_observations([obs1, obs2], cfg, overlap_groups=[])
    assert len(deduped) == 2, "no overlap group -> no merge"


def test_group_appearance_dedup_face_conflict_blocked() -> None:
    """Group appearance dedup respects face-conflict gate."""
    cfg = WorldTrackerConfig(
        dedup_enabled=True,
        enable_group_appearance_dedup=True,
        dedup_group_appearance_min_sim=0.75,
        dedup_require_no_face_conflict=True,
    )
    obs1 = _obs(
        "cam-1",
        5.0,
        5.0,
        detection_id="d1",
        calibrated=False,
        embedding=[0.9, 0.1, 0.0],
        face_person_id="alice",
    )
    obs2 = _obs(
        "cam-2",
        5.0,
        5.0,
        detection_id="d2",
        calibrated=False,
        embedding=[0.85, 0.15, 0.0],
        face_person_id="bob",
    )
    deduped, _ = dedup_observations([obs1, obs2], cfg, overlap_groups=[_GROUP])
    assert len(deduped) == 2, "face conflict must prevent group-appearance merge"


def test_group_appearance_dedup_disabled_skips() -> None:
    """When enable_group_appearance_dedup is False, uncalibrated obs never merge."""
    cfg = WorldTrackerConfig(
        dedup_enabled=True,
        enable_group_appearance_dedup=False,
    )
    obs1 = _obs(
        "cam-1",
        5.0,
        5.0,
        detection_id="d1",
        calibrated=False,
        embedding=[0.9, 0.1, 0.0],
        quality=0.7,
    )
    obs2 = _obs(
        "cam-2",
        5.0,
        5.0,
        detection_id="d2",
        calibrated=False,
        embedding=[0.85, 0.15, 0.0],
        quality=0.5,
    )
    deduped, _ = dedup_observations([obs1, obs2], cfg, overlap_groups=[_GROUP])
    assert len(deduped) == 2, "flag off -> no group-appearance dedup"

"""Unit tests for the cross-camera dedup pass (geometric + group-appearance)."""

from __future__ import annotations

import dataclasses
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


# ---------------------------------------------------------------------------
# M04: Information-form fusion tests
# ---------------------------------------------------------------------------


def _obs_with_cov(
    camera_id: str,
    x_m: float,
    y_m: float,
    *,
    detection_id: str = "",
    quality: float = 0.5,
    floor_cov_random: tuple[float, float, float, float] | None = None,
    floor_residual_m: float | None = None,
) -> WorldObservation:
    base = _obs(
        camera_id=camera_id,
        x_m=x_m,
        y_m=y_m,
        detection_id=detection_id,
        quality=quality,
        floor_residual_m=floor_residual_m,
    )
    return dataclasses.replace(base, floor_cov_random=floor_cov_random)


def test_fusion_weights_toward_low_covariance_camera() -> None:
    """Fused position sits near the low-R camera, not the midpoint a mean would give."""
    r_low = (0.0001, 0.0, 0.0, 0.0001)  # 1 cm sigma — very precise
    r_high = (1.0, 0.0, 0.0, 1.0)  # 1 m sigma — very imprecise
    # Place observations within the 0.6 m dedup gate.
    obs_low_r = _obs_with_cov(
        "cam-1", 5.0, 5.0, detection_id="d1", quality=0.5, floor_cov_random=r_low
    )
    obs_high_r = _obs_with_cov(
        "cam-2", 5.5, 5.0, detection_id="d2", quality=0.7, floor_cov_random=r_high
    )
    deduped, _ = dedup_observations([obs_low_r, obs_high_r], _CFG)
    assert len(deduped) == 1
    rep = deduped[0]
    fused_x_m = rep.floor_point.x_mm / 1000.0
    # Fused position must be much closer to 5.0 (low-R cam) than to 5.5 (high-R cam).
    # A quality-weighted mean with quality 0.5 and 0.7 would give ~5.3; information-form
    # fusion with r_high >> r_low gives nearly 5.0.
    assert fused_x_m < 5.1, (
        f"fused x={fused_x_m:.3f} should be near 5.0 (low-R camera), not near 5.5"
    )


def test_fusion_does_not_shrink_below_bias_floor() -> None:
    """Anti-jump guarantee: fused R* diagonal never falls below the bias floor."""
    residual_m = 0.1
    bias_floor_diag = (1.0 * residual_m) ** 2  # k_cal=1.0
    r_rand = (0.04, 0.0, 0.0, 0.04)
    # Build N identical cameras all at the same point.
    for n_cameras in [2, 3, 5]:
        cameras = [
            _obs_with_cov(
                f"cam-{i}",
                5.0,
                5.0,
                detection_id=f"d{i}",
                quality=0.5,
                floor_cov_random=r_rand,
                floor_residual_m=residual_m,
            )
            for i in range(n_cameras)
        ]
        # Build clusters manually for n>2 since the dedup gate only merges pairs.
        # Use a very wide gate to ensure all collapse.
        cfg_wide = WorldTrackerConfig(dedup_enabled=True, dedup_max_distance_m=100.0)
        deduped, _ = dedup_observations(cameras, cfg_wide)
        assert len(deduped) == 1
        rep = deduped[0]
        assert rep.floor_cov_random is not None
        fused_diag = rep.floor_cov_random[0]  # R*[0,0]
        assert fused_diag >= bias_floor_diag - 1e-9, (
            f"R* diagonal {fused_diag:.6f} fell below bias_floor {bias_floor_diag:.6f} "
            f"at N={n_cameras} cameras"
        )


def test_bias_floor_uses_worst_residual() -> None:
    """Bias floor comes from the cluster member with the largest residual."""
    r_rand = (0.04, 0.0, 0.0, 0.04)
    obs_good = _obs_with_cov(
        "cam-1",
        5.0,
        5.0,
        detection_id="d1",
        quality=0.5,
        floor_cov_random=r_rand,
        floor_residual_m=0.05,
    )
    obs_bad = _obs_with_cov(
        "cam-2",
        5.1,
        5.0,
        detection_id="d2",
        quality=0.5,
        floor_cov_random=r_rand,
        floor_residual_m=0.5,
    )
    deduped, _ = dedup_observations([obs_good, obs_bad], _CFG)
    assert len(deduped) == 1
    rep = deduped[0]
    assert rep.floor_cov_random is not None
    expected_bias_floor = (1.0 * 0.5) ** 2  # k_cal * worst_residual
    fused_diag = rep.floor_cov_random[0]
    assert fused_diag >= expected_bias_floor - 1e-9, (
        f"bias floor {fused_diag:.6f} < expected {expected_bias_floor:.6f}"
    )


def test_mixed_calibrated_uncalibrated_cluster() -> None:
    """Only calibrated members with floor_cov_random contribute to fusion."""
    r_rand = (0.04, 0.0, 0.0, 0.04)
    obs_calib = _obs_with_cov(
        "cam-1", 5.0, 5.0, detection_id="d1", quality=0.5, floor_cov_random=r_rand
    )
    obs_uncalib = _obs_with_cov(
        "cam-2", 5.05, 5.0, detection_id="d2", quality=0.5, floor_cov_random=None
    )
    obs_uncalib = dataclasses.replace(
        obs_uncalib,
        floor_point=dataclasses.replace(obs_uncalib.floor_point, calibrated=False),
    )
    # The uncalibrated obs cannot form a geometric dedup cluster with the calibrated one.
    # But we can test _build_representative directly.
    from app.tracking.world.dedup import _build_representative, _select_representative

    cluster = [obs_calib, obs_uncalib]
    best = _select_representative(cluster)
    rep = _build_representative(best, cluster)
    # Position must come from the calibrated member only.
    assert rep.floor_point.x_mm == 5000
    assert rep.floor_point.y_mm == 5000


def test_single_camera_cluster_equals_single_obs_cov() -> None:
    """With one calibrated member, the representative carries the same covariance."""
    r_rand = (0.04, 0.0, 0.0, 0.04)
    obs = _obs_with_cov(
        "cam-1",
        5.0,
        5.0,
        detection_id="d1",
        quality=0.5,
        floor_cov_random=r_rand,
        floor_residual_m=0.0,
    )
    # Singleton cluster (only one member): dedup returns it unchanged.
    deduped, _ = dedup_observations([obs], _CFG)
    assert len(deduped) == 1
    # The singleton is returned as-is (no _build_representative called).
    assert deduped[0].floor_cov_random == r_rand


def test_representative_floor_point_is_quality_weighted_fallback() -> None:
    """Without floor_cov_random, the fallback quality-weighted mean still works."""
    obs_hi = _obs("cam-1", 5.0, 5.0, detection_id="d1", quality=1.0)
    obs_lo = _obs("cam-2", 5.6, 5.0, detection_id="d2", quality=0.0)
    deduped, _ = dedup_observations([obs_hi, obs_lo], _CFG)
    assert len(deduped) == 1
    rep = deduped[0]
    # floor_cov_random=None on both → fallback path → quality-weighted mean.
    assert abs(rep.floor_point.x_mm / 1000.0 - 5.0) < 0.01

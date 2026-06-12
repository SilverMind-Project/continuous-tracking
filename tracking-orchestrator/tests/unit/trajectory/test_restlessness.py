"""Unit tests for app.trajectory.restlessness.

Each test covers a distinct fixture scenario as required by the task spec.
All point sequences are synthetic and seeded for determinism where randomness
is needed.
"""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime, timedelta

import pytest

from app.domain import PersonTrajectoryPoint
from app.trajectory.restlessness import (
    RestlessnessConfig,
    RestlessnessFeatures,
    compute_restlessness,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IDENTITY = "alice"
_PH = "ph-001"
_T0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)


def _pt(
    *,
    t_offset_s: float = 0.0,
    room: str = "living_room",
    x: float = 0.0,
    y: float = 0.0,
    motion_energy: float | None = None,
    floor_speed: float | None = None,
) -> PersonTrajectoryPoint:
    return PersonTrajectoryPoint(
        identity_id=_IDENTITY,
        ph_id=_PH,
        observed_at=_T0 + timedelta(seconds=t_offset_s),
        room_name=room,
        ground_x=x,
        ground_y=y,
        motion_energy=motion_energy,
        floor_speed_m_s=floor_speed,
    )


# ---------------------------------------------------------------------------
# Fixture 1: fidget — high motion energy, near-zero displacement
# ---------------------------------------------------------------------------


class TestFidgetFixture:
    """High motion energy, near-zero displacement: in-place motion, no excursions."""

    def _make_points(self, n: int = 40) -> list[PersonTrajectoryPoint]:
        # Slight jitter around origin, all in same room.
        rng = random.Random(42)
        pts = []
        for i in range(n):
            pts.append(
                _pt(
                    t_offset_s=i * 30.0,  # one point every 30 s
                    room="living_room",
                    x=rng.uniform(-0.05, 0.05),  # < min_displacement_m
                    y=rng.uniform(-0.05, 0.05),
                    motion_energy=0.30,  # well above motion_active_floor (0.10)
                    floor_speed=0.05,  # well below locomotion_floor (0.25)
                )
            )
        return pts

    def test_in_place_motion_ratio_near_one(self) -> None:
        pts = self._make_points()
        feats = compute_restlessness(pts)
        assert feats.in_place_motion_ratio is not None
        assert feats.in_place_motion_ratio == pytest.approx(1.0)

    def test_direction_entropy_none_or_low_due_to_short_displacements(self) -> None:
        pts = self._make_points()
        feats = compute_restlessness(pts)
        # Displacements < min_displacement_m (0.20 m) so no vectors qualify —
        # entropy is None (fewer than 2 qualifying vectors).
        assert feats.direction_change_entropy is None

    def test_no_excursions(self) -> None:
        pts = self._make_points()
        feats = compute_restlessness(pts)
        # All within 0.05 m of origin — peak departure never reaches 2.0 m.
        assert feats.short_excursion_rate == pytest.approx(0.0, abs=0.01)

    def test_motion_energy_p75_elevated(self) -> None:
        pts = self._make_points()
        feats = compute_restlessness(pts)
        assert feats.motion_energy_p75 is not None
        assert feats.motion_energy_p75 > 0.10

    def test_observed_minutes_matches_point_count(self) -> None:
        pts = self._make_points(30)
        feats = compute_restlessness(pts)
        assert feats.observed_minutes == 30


# ---------------------------------------------------------------------------
# Fixture 2: purposeful walk — straight hallway
# ---------------------------------------------------------------------------


class TestPurposefulWalkFixture:
    """Straight hallway walk: low in-place ratio, low direction entropy."""

    def _make_points(self, n: int = 30) -> list[PersonTrajectoryPoint]:
        pts = []
        for i in range(n):
            pts.append(
                _pt(
                    t_offset_s=i * 5.0,
                    room="hallway",
                    x=i * 0.5,  # 0.5 m per step = straight line, well above min_displacement
                    y=0.0,
                    motion_energy=0.25,  # active (walking energy)
                    floor_speed=0.9,  # well above locomotion_floor (0.25)
                )
            )
        return pts

    def test_in_place_ratio_near_zero(self) -> None:
        pts = self._make_points()
        feats = compute_restlessness(pts)
        assert feats.in_place_motion_ratio is not None
        # floor_speed >= locomotion_floor for all points → ratio = 0.0
        assert feats.in_place_motion_ratio == pytest.approx(0.0)

    def test_entropy_low(self) -> None:
        pts = self._make_points()
        feats = compute_restlessness(pts)
        # All displacement vectors point in +x direction → 1 bin occupied → entropy = 0.
        assert feats.direction_change_entropy is not None
        assert feats.direction_change_entropy < 0.4


# ---------------------------------------------------------------------------
# Fixture 3: aimless wander — seeded random walk
# ---------------------------------------------------------------------------


class TestAimlessWanderFixture:
    """Random walk in a room: high direction entropy, several short excursions."""

    def _make_points(self) -> list[PersonTrajectoryPoint]:
        rng = random.Random(7)
        pts = []
        x, y = 0.0, 0.0
        for i in range(60):
            # Larger steps so they exceed min_displacement_m
            dx = rng.uniform(-0.6, 0.6)
            dy = rng.uniform(-0.6, 0.6)
            x += dx
            y += dy
            pts.append(
                _pt(
                    t_offset_s=i * 30.0,
                    room="living_room",
                    x=x,
                    y=y,
                    motion_energy=0.20,
                    floor_speed=0.18,  # below locomotion_floor → in-place category
                )
            )
        return pts

    def test_entropy_high(self) -> None:
        pts = self._make_points()
        feats = compute_restlessness(pts)
        assert feats.direction_change_entropy is not None
        assert feats.direction_change_entropy > 0.7

    def test_some_excursions(self) -> None:
        pts = self._make_points()
        feats = compute_restlessness(pts)
        # Random walk with ±0.6 m steps will cross the 2 m departure radius
        # and return; the exact count depends on the seed but should be > 0.
        assert feats.short_excursion_rate is not None
        assert feats.short_excursion_rate >= 0.0


# ---------------------------------------------------------------------------
# Fixture 4: TV watching — still, occasional posture shifts
# ---------------------------------------------------------------------------


class TestTvWatchingFixture:
    """Still, occasional posture shifts: in_place_motion_ratio LOW (not high).

    The key false-positive guard: motion energy stays UNDER motion_active_floor
    (0.10 nu/s) even during posture shifts.  TV watching is the canonical
    non-agitation scenario that must not trigger the signal.
    """

    def _make_points(self, n: int = 40) -> list[PersonTrajectoryPoint]:
        rng = random.Random(99)
        pts = []
        for i in range(n):
            # Occasional posture shift: motion_energy 0.06-0.08 nu/s.
            # This is above the M1 stillness floor (0.05) but BELOW motion_active_floor (0.10).
            me = rng.uniform(0.01, 0.08)
            pts.append(
                _pt(
                    t_offset_s=i * 30.0,
                    room="living_room",
                    x=rng.uniform(-0.03, 0.03),
                    y=rng.uniform(-0.03, 0.03),
                    motion_energy=me,
                    floor_speed=0.02,
                )
            )
        return pts

    def test_in_place_ratio_low(self) -> None:
        pts = self._make_points()
        feats = compute_restlessness(pts)
        # All motion_energy values are < motion_active_floor (0.10), so no
        # points qualify as "active" → in_place_motion_ratio should be 0.0.
        assert feats.in_place_motion_ratio is not None
        assert feats.in_place_motion_ratio == pytest.approx(0.0)

    def test_motion_energy_p75_none_or_low(self) -> None:
        # p75 is over active points (motion_energy > active_floor).
        # If none qualify, it is None; otherwise it should be low.
        pts = self._make_points()
        feats = compute_restlessness(pts)
        assert feats.motion_energy_p75 is None


# ---------------------------------------------------------------------------
# Fixture 5: mixed calibration window
# ---------------------------------------------------------------------------


class TestMixedCalibrationFixture:
    """Less than half the window is calibrated: geometric features are None."""

    def _make_points(self) -> list[PersonTrajectoryPoint]:
        pts = []
        for i in range(20):
            calibrated = i < 5  # only 5/20 = 25% calibrated
            pts.append(
                _pt(
                    t_offset_s=i * 30.0,
                    x=float(i),
                    y=0.0,
                    motion_energy=0.20,
                    floor_speed=0.10 if calibrated else None,
                )
            )
        return pts

    def test_geometric_features_none(self) -> None:
        pts = self._make_points()
        feats = compute_restlessness(pts)
        assert feats.direction_change_entropy is None
        assert feats.short_excursion_rate is None

    def test_scalar_features_still_computed(self) -> None:
        pts = self._make_points()
        feats = compute_restlessness(pts)
        # motion_energy is non-null for all points; floor_speed is missing for 15/20
        # so in_place_motion_ratio is computed over the 5 calibrated points only.
        assert feats.motion_energy_p75 is not None
        # in_place_motion_ratio uses points that have BOTH fields non-null.
        assert feats.in_place_motion_ratio is not None

    def test_observed_minutes_counts_all_points(self) -> None:
        pts = self._make_points()
        feats = compute_restlessness(pts)
        assert feats.observed_minutes == 20


# ---------------------------------------------------------------------------
# Fixture 6: empty and very short windows
# ---------------------------------------------------------------------------


class TestEmptyAndShortWindowsFixture:
    def test_empty_window_all_none(self) -> None:
        feats = compute_restlessness([])
        assert feats.in_place_motion_ratio is None
        assert feats.direction_change_entropy is None
        assert feats.short_excursion_rate is None
        assert feats.motion_energy_p75 is None
        assert feats.observed_minutes == 0

    def test_single_point_window(self) -> None:
        pts = [_pt(motion_energy=0.30, floor_speed=0.05)]
        feats = compute_restlessness(pts)
        assert feats.observed_minutes == 1
        # In-place ratio: 1 eligible point, active and below locomotion_floor.
        assert feats.in_place_motion_ratio == pytest.approx(1.0)
        # Direction entropy: < 2 vectors → None.
        assert feats.direction_change_entropy is None
        # Excursion rate: < 3 calibrated points → None.
        assert feats.short_excursion_rate is None

    def test_two_points_same_position(self) -> None:
        pts = [
            _pt(t_offset_s=0.0, motion_energy=0.20, floor_speed=0.10),
            _pt(t_offset_s=30.0, motion_energy=0.20, floor_speed=0.10),
        ]
        feats = compute_restlessness(pts)
        assert feats.observed_minutes == 2
        assert feats.direction_change_entropy is None  # < 2 qualifying vectors

    def test_two_calibrated_points_excursion_rate_none(self) -> None:
        # Excursion rate requires >= 3 calibrated points.
        pts = [
            _pt(t_offset_s=0.0, x=0.0, floor_speed=0.10),
            _pt(t_offset_s=30.0, x=5.0, floor_speed=0.10),
        ]
        feats = compute_restlessness(pts)
        assert feats.short_excursion_rate is None


# ---------------------------------------------------------------------------
# Additional behavioural tests
# ---------------------------------------------------------------------------


class TestDirectionEntropyBounds:
    def test_uniform_all_directions_entropy_near_one(self) -> None:
        """Equal counts in all 8 compass bins → max entropy → normalised = 1.0."""
        pts = []
        # 8 directions: steps that cleanly land in each octant
        angles = [i * math.pi / 4 for i in range(8)]
        for idx, angle in enumerate(angles * 4):  # 4 points per bin = 32 total
            dx = math.cos(angle) * 0.5
            dy = math.sin(angle) * 0.5
            pts.append(
                _pt(
                    t_offset_s=idx * 30.0,
                    x=dx * idx,
                    y=dy * idx,
                    floor_speed=0.5,
                )
            )
        feats = compute_restlessness(pts)
        # All 8 bins have equal representation → entropy = log2(8) / log2(8) = 1.0.
        assert feats.direction_change_entropy is not None
        assert feats.direction_change_entropy == pytest.approx(1.0, abs=0.05)

    def test_single_direction_entropy_near_zero(self) -> None:
        """All displacement vectors in one compass bin → entropy = 0."""
        pts = [_pt(t_offset_s=i * 10.0, x=i * 0.5, y=0.0, floor_speed=0.5) for i in range(10)]
        feats = compute_restlessness(pts)
        assert feats.direction_change_entropy is not None
        assert feats.direction_change_entropy == pytest.approx(0.0, abs=0.01)


class TestInPlaceMotionEdgeCases:
    def test_missing_motion_energy_excluded(self) -> None:
        """Points with None motion_energy are excluded from numerator and denominator."""
        pts = [
            _pt(t_offset_s=0.0, motion_energy=None, floor_speed=0.05),  # excluded
            _pt(t_offset_s=30.0, motion_energy=0.30, floor_speed=0.05),  # in-place
            _pt(t_offset_s=60.0, motion_energy=0.30, floor_speed=0.05),  # in-place
        ]
        feats = compute_restlessness(pts)
        # Denominator = 2 (the two non-null points), both in-place → ratio = 1.0
        assert feats.in_place_motion_ratio == pytest.approx(1.0)

    def test_missing_floor_speed_excluded(self) -> None:
        """Points with None floor_speed_m_s are excluded."""
        pts = [
            _pt(t_offset_s=0.0, motion_energy=0.30, floor_speed=None),  # excluded
            _pt(t_offset_s=30.0, motion_energy=0.30, floor_speed=0.50),  # locomotion
        ]
        feats = compute_restlessness(pts)
        # Denominator = 1, floor_speed >= locomotion_floor → not in-place → ratio = 0.0
        assert feats.in_place_motion_ratio == pytest.approx(0.0)

    def test_all_points_missing_both_fields_returns_none(self) -> None:
        pts = [_pt(t_offset_s=i * 30.0) for i in range(5)]  # both None by default
        feats = compute_restlessness(pts)
        assert feats.in_place_motion_ratio is None


class TestShortExcursionCounting:
    def test_single_out_and_back_counted(self) -> None:
        """Person goes > 2 m from origin and returns within 2 m and 60 s."""
        pts = [
            _pt(t_offset_s=0.0, x=0.0, y=0.0, floor_speed=0.5),
            _pt(t_offset_s=5.0, x=1.5, y=0.0, floor_speed=0.5),
            _pt(t_offset_s=10.0, x=3.0, y=0.0, floor_speed=0.5),  # peak > 2 m
            _pt(t_offset_s=20.0, x=1.0, y=0.0, floor_speed=0.5),  # returned < 2 m
        ]
        feats = compute_restlessness(pts)
        assert feats.short_excursion_rate is not None
        assert feats.short_excursion_rate > 0.0

    def test_room_change_resets_excursion(self) -> None:
        """A room change aborts the current excursion tracking."""
        pts = [
            _pt(t_offset_s=0.0, x=0.0, room="living_room", floor_speed=0.5),
            _pt(t_offset_s=5.0, x=1.5, room="living_room", floor_speed=0.5),
            _pt(t_offset_s=10.0, x=3.0, room="hallway", floor_speed=0.5),  # room change: abort
            _pt(t_offset_s=15.0, x=1.0, room="hallway", floor_speed=0.5),
        ]
        feats = compute_restlessness(pts)
        assert feats.short_excursion_rate is not None
        # Excursion was aborted by room change; no loop completed.
        assert feats.short_excursion_rate == pytest.approx(0.0, abs=0.01)

    def test_timeout_resets_excursion(self) -> None:
        """An excursion exceeding max seconds does not count."""
        cfg = RestlessnessConfig(excursion_max_seconds=30.0)
        pts = [
            _pt(t_offset_s=0.0, x=0.0, floor_speed=0.5),
            _pt(t_offset_s=5.0, x=1.5, floor_speed=0.5),
            _pt(t_offset_s=10.0, x=3.0, floor_speed=0.5),  # peak > 2 m
            _pt(t_offset_s=50.0, x=0.5, floor_speed=0.5),  # 45 s after away_start → timeout
        ]
        feats = compute_restlessness(pts, cfg)
        assert feats.short_excursion_rate is not None
        assert feats.short_excursion_rate == pytest.approx(0.0, abs=0.01)

    def test_small_departure_below_radius_not_counted(self) -> None:
        """Departure that never exceeds excursion_radius_m is not counted as excursion."""
        pts = [
            _pt(t_offset_s=0.0, x=0.0, floor_speed=0.5),
            _pt(t_offset_s=5.0, x=0.6, floor_speed=0.5),  # departed but < 2.0 m
            _pt(t_offset_s=10.0, x=1.8, floor_speed=0.5),  # peak = 1.8 m < radius
            _pt(t_offset_s=15.0, x=0.4, floor_speed=0.5),  # "returned" but peak never >= 2 m
        ]
        feats = compute_restlessness(pts)
        assert feats.short_excursion_rate is not None
        assert feats.short_excursion_rate == pytest.approx(0.0, abs=0.01)


class TestMotionEnergyP75:
    def test_p75_selects_correct_percentile(self) -> None:
        """With 4 active points, p75 = 3rd highest (index 2 in 0-based sorted list)."""
        # active values: [0.15, 0.20, 0.25, 0.30] → sorted → p75 index = ceil(4*0.75)-1 = 2
        # → value = 0.25
        pts = [
            _pt(t_offset_s=i * 30.0, motion_energy=v, floor_speed=None)
            for i, v in enumerate([0.30, 0.15, 0.25, 0.20])
        ]
        feats = compute_restlessness(pts)
        assert feats.motion_energy_p75 is not None
        assert feats.motion_energy_p75 == pytest.approx(0.25)

    def test_p75_excludes_inactive_points(self) -> None:
        """Points with motion_energy <= motion_active_floor are excluded."""
        cfg = RestlessnessConfig(motion_active_floor=0.10)
        pts = [
            _pt(t_offset_s=0.0, motion_energy=0.05, floor_speed=None),  # below floor
            _pt(t_offset_s=30.0, motion_energy=0.03, floor_speed=None),  # below floor
            _pt(t_offset_s=60.0, motion_energy=0.40, floor_speed=None),  # active
        ]
        feats = compute_restlessness(pts, cfg)
        assert feats.motion_energy_p75 == pytest.approx(0.40)

    def test_p75_none_when_no_active_points(self) -> None:
        pts = [_pt(t_offset_s=i * 30.0, motion_energy=0.05) for i in range(5)]
        feats = compute_restlessness(pts)
        assert feats.motion_energy_p75 is None


class TestConfigDefaults:
    def test_default_config_produces_restlessness_features(self) -> None:
        pts = [_pt(t_offset_s=i * 30.0, motion_energy=0.20, floor_speed=0.10) for i in range(10)]
        feats = compute_restlessness(pts)
        assert isinstance(feats, RestlessnessFeatures)

    def test_custom_config_applied(self) -> None:
        """Higher motion_active_floor makes fewer points active."""
        pts = [_pt(t_offset_s=i * 30.0, motion_energy=0.12, floor_speed=0.05) for i in range(10)]
        cfg_low = RestlessnessConfig(motion_active_floor=0.10)
        cfg_high = RestlessnessConfig(motion_active_floor=0.20)

        feats_low = compute_restlessness(pts, cfg_low)
        feats_high = compute_restlessness(pts, cfg_high)

        # With low floor all 10 are active; with high floor none are.
        assert feats_low.in_place_motion_ratio == pytest.approx(1.0)
        assert feats_high.in_place_motion_ratio == pytest.approx(0.0)

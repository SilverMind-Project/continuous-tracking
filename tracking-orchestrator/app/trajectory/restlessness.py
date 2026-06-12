"""Per-window restlessness features for the agitation-index signal (M4).

Mirrors the window-handling conventions of DementiaSignalWorker._compute_pacing
(sorted ascending, density gating) and must not duplicate room-transition logic.

Unit: motion_energy in normalised keypoint velocity (nu/s); floor positions in
metres (floor-plan frame).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ..domain import PersonTrajectoryPoint

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RestlessnessConfig:
    """Thresholds for restlessness feature extraction.

    motion_active_floor:
        Motion energy above which a point counts as "active" (nu/s).
        Set above the M1 stillness floor (0.05 nu/s, p95 of still segments)
        so that passive frames — TV-watching, breathing micro-adjustments —
        are excluded from in_place_motion_ratio.  Walking is ~0.15 nu/s;
        this floor sits between still p95 (0.05) and walking p5 (0.15).

    locomotion_floor:
        Kalman floor speed below which an active point is classified as
        in-place motion (fidgeting, rocking) rather than locomotion (m/s).
        At 0.25 m/s the person is not traversing the room.

    min_displacement_m:
        Minimum consecutive calibrated-point displacement to include a
        direction vector in the entropy histogram.  Skips camera jitter
        (< 0.2 m) that would otherwise inflate entropy artificially.

    excursion_radius_m:
        Boundary radius defining "home" (m).  An excursion is completed
        when the person returns to within this distance of the excursion
        origin after having reached beyond it.

    excursion_departure_m:
        Minimum departure from the current origin before the excursion
        state machine enters "away".  Prevents postural sway from
        triggering the tracker.

    excursion_max_seconds:
        Maximum duration of a valid short excursion loop.  Loops that
        take longer are locomotion, not sub-room restlessness.

    min_calibrated_fraction:
        If fewer than this fraction of window points have a non-null
        floor_speed_m_s (calibrated camera), geometric features
        (direction_change_entropy, short_excursion_rate) are returned
        as None.  Never silently compute geometry from uncalibrated
        floor positions.

    window_minutes:
        Expected window duration in minutes.  Used only to convert the
        raw excursion count to a per-hour rate when the window is shorter
        than expected.
    """

    motion_active_floor: float = 0.10
    locomotion_floor: float = 0.25
    min_displacement_m: float = 0.20
    excursion_radius_m: float = 2.0
    excursion_departure_m: float = 0.50
    excursion_max_seconds: float = 60.0
    min_calibrated_fraction: float = 0.50
    window_minutes: int = 30


# ---------------------------------------------------------------------------
# Feature record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RestlessnessFeatures:
    """Per-identity, per-window restlessness summary.

    Scalar features (in_place_motion_ratio, motion_energy_p75) are None only
    when no eligible points exist in the window.

    Geometric features (direction_change_entropy, short_excursion_rate) are
    None when fewer than min_calibrated_fraction of window points are
    calibrated, or when there are too few displacement vectors to compute
    a meaningful distribution.

    observed_minutes:
        Point-count proxy for minutes (same docstring caveat as M1 task 1:
        each trajectory point represents roughly one observation minute
        at normal sampling rates; this is not a wall-clock measurement).
    """

    in_place_motion_ratio: float | None
    direction_change_entropy: float | None
    short_excursion_rate: float | None
    motion_energy_p75: float | None
    observed_minutes: int


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_restlessness(
    points: Sequence[PersonTrajectoryPoint],
    cfg: RestlessnessConfig | None = None,
) -> RestlessnessFeatures:
    """Compute restlessness features over a rolling window of trajectory points.

    Points must all belong to a single identity.  Sorting is performed
    internally (ascending by observed_at), matching the convention in
    DementiaSignalWorker._compute_pacing.

    Uncalibrated deployments: if fewer than cfg.min_calibrated_fraction of
    points have non-null floor_speed_m_s, direction_change_entropy and
    short_excursion_rate are returned as None.
    """
    if cfg is None:
        cfg = RestlessnessConfig()

    sorted_pts = sorted(points, key=lambda p: p.observed_at)
    observed_minutes = len(sorted_pts)

    if not sorted_pts:
        return RestlessnessFeatures(
            in_place_motion_ratio=None,
            direction_change_entropy=None,
            short_excursion_rate=None,
            motion_energy_p75=None,
            observed_minutes=0,
        )

    n_calibrated = sum(1 for p in sorted_pts if p.floor_speed_m_s is not None)
    geo_ok = (n_calibrated / len(sorted_pts)) >= cfg.min_calibrated_fraction

    return RestlessnessFeatures(
        in_place_motion_ratio=_in_place_motion_ratio(sorted_pts, cfg),
        direction_change_entropy=_direction_change_entropy(sorted_pts, cfg) if geo_ok else None,
        short_excursion_rate=_short_excursion_rate(sorted_pts, cfg) if geo_ok else None,
        motion_energy_p75=_motion_energy_p75(sorted_pts, cfg),
        observed_minutes=observed_minutes,
    )


# ---------------------------------------------------------------------------
# Feature helpers (pure functions, no I/O)
# ---------------------------------------------------------------------------


def _in_place_motion_ratio(
    sorted_pts: list[PersonTrajectoryPoint],
    cfg: RestlessnessConfig,
) -> float | None:
    """Fraction of points with active body motion but no locomotion.

    Requires both motion_energy and floor_speed_m_s to be non-null.
    Points missing either field are excluded from numerator and denominator.
    """
    # Unpack to (motion_energy, floor_speed) tuples so mypy can track non-None narrowing.
    eligible: list[tuple[float, float]] = [
        (p.motion_energy, p.floor_speed_m_s)
        for p in sorted_pts
        if p.motion_energy is not None and p.floor_speed_m_s is not None
    ]
    if not eligible:
        return None
    in_place = sum(
        me > cfg.motion_active_floor and fs < cfg.locomotion_floor for me, fs in eligible
    )
    return round(in_place / len(eligible), 4)


def _direction_bin(dx: float, dy: float) -> int:
    """Map a 2-D displacement vector into one of 8 equal compass bins [0..7]."""
    angle = math.atan2(dy, dx)  # [-pi, pi]
    return int((angle + math.pi) / (2.0 * math.pi) * 8) % 8


def _direction_change_entropy(
    sorted_pts: list[PersonTrajectoryPoint],
    cfg: RestlessnessConfig,
) -> float | None:
    """Shannon entropy of 8-bin compass histogram, normalised to [0, 1].

    Only calibrated points (non-null floor_speed_m_s) contribute.
    Consecutive displacement vectors shorter than min_displacement_m are
    skipped to suppress jitter.

    Returns None when fewer than 2 qualifying vectors exist.
    """
    cal = [p for p in sorted_pts if p.floor_speed_m_s is not None]
    bins: list[int] = [0] * 8
    n_vecs = 0

    for i in range(1, len(cal)):
        dx = cal[i].ground_x - cal[i - 1].ground_x
        dy = cal[i].ground_y - cal[i - 1].ground_y
        if math.hypot(dx, dy) < cfg.min_displacement_m:
            continue
        bins[_direction_bin(dx, dy)] += 1
        n_vecs += 1

    if n_vecs < 2:
        return None

    entropy = 0.0
    for count in bins:
        if count > 0:
            p_bin = count / n_vecs
            entropy -= p_bin * math.log2(p_bin)

    # Max entropy for 8 bins = log2(8) = 3.0 bits
    return round(entropy / 3.0, 4)


def _short_excursion_rate(
    sorted_pts: list[PersonTrajectoryPoint],
    cfg: RestlessnessConfig,
) -> float | None:
    """Count per hour of excursion loops (leave beyond radius, return within it).

    An excursion is a sub-room loop in which the person:
      1. departs at least excursion_departure_m from the current origin,
      2. reaches a peak distance >= excursion_radius_m from that origin,
      3. returns to within excursion_radius_m of the origin,
      4. all within excursion_max_seconds and without a room change.

    The origin resets on room change, timeout, or completed excursion.

    Returns None when fewer than 3 calibrated points exist or the calibrated
    window spans zero seconds.
    """
    cal = [p for p in sorted_pts if p.floor_speed_m_s is not None]
    if len(cal) < 3:
        return None

    excursion_count = 0
    ox, oy = cal[0].ground_x, cal[0].ground_y
    origin_room = cal[0].room_name
    state = "home"
    away_start: PersonTrajectoryPoint | None = None
    peak_departure = 0.0

    for p in cal[1:]:
        dist = math.hypot(p.ground_x - ox, p.ground_y - oy)

        if state == "home":
            if dist >= cfg.excursion_departure_m:
                state = "away"
                away_start = p
                peak_departure = dist
        else:
            # Room change: this is locomotion, not sub-room restlessness.
            if p.room_name != origin_room:
                state = "home"
                ox, oy = p.ground_x, p.ground_y
                origin_room = p.room_name
                away_start = None
                peak_departure = 0.0
                continue

            assert away_start is not None
            elapsed = (p.observed_at - away_start.observed_at).total_seconds()
            if elapsed > cfg.excursion_max_seconds:
                state = "home"
                ox, oy = p.ground_x, p.ground_y
                origin_room = p.room_name
                away_start = None
                peak_departure = 0.0
                continue

            if dist > peak_departure:
                peak_departure = dist

            # Count a return only after the person reached beyond excursion_radius_m.
            # This prevents a 0.6 m departure from immediately triggering a return
            # at the 2.0 m return threshold.
            if peak_departure >= cfg.excursion_radius_m and dist < cfg.excursion_radius_m:
                excursion_count += 1
                state = "home"
                ox, oy = p.ground_x, p.ground_y
                origin_room = p.room_name
                away_start = None
                peak_departure = 0.0

    window_seconds = (cal[-1].observed_at - cal[0].observed_at).total_seconds()
    if window_seconds <= 0:
        return None

    return round(excursion_count / (window_seconds / 3600.0), 3)


def _motion_energy_p75(
    sorted_pts: list[PersonTrajectoryPoint],
    cfg: RestlessnessConfig,
) -> float | None:
    """75th-percentile motion energy over active points (nearest-rank method).

    Only points with motion_energy > motion_active_floor are included so that
    the statistic captures the sustained elevation level during agitated
    periods rather than being diluted by idle frames.
    """
    active = sorted(
        p.motion_energy
        for p in sorted_pts
        if p.motion_energy is not None and p.motion_energy > cfg.motion_active_floor
    )
    if not active:
        return None
    idx = max(0, math.ceil(len(active) * 0.75) - 1)
    return round(active[idx], 6)

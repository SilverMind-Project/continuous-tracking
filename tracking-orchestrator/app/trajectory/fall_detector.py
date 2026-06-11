"""FallDetector: deterministic rules over FallFeatures to detect suspected falls.

Pure logic, no I/O, no Triton, no DB. Thresholds are initial values;
task 2.5 calibrates them against replay fixtures.

Detection rules (emit fall_suspected warning when ALL hold):
  1. features.samples >= min_samples
  2. features.max_descent_rate_hps >= max_descent_rate_hps_threshold
  3. features.height_ratio_now <= height_ratio_threshold
  4. features.lying_score_now >= lying_score_threshold  OR  pose unavailable
  5. features.floor_speed_at_event_m_s <= floor_speed_max_m_s  or None
  6. room_name not in resting_rooms

Escalation (warning → emergency within escalation_window_s):
  post_event_motion_nu_s <= stillness_motion_floor AND posture lying/unknown

Cancellation: height_ratio above standing_clear_height_ratio AND
  lying_score below standing_clear_lying_score (person returned to standing).
"""

from __future__ import annotations

from dataclasses import dataclass

from .fall_features import FallFeatures


@dataclass(frozen=True)
class FallDetectorConfig:
    """Thresholds for FallDetector rules.

    All values are initial estimates; task 2.5 calibrates these against
    replay fixtures from real fall scenarios.
    """

    # Rule 1: minimum frame count before detection is attempted.
    min_samples: int = 5

    # Rule 2: downward velocity in body-heights per second.
    # Falls exceed 0.8 hps; controlled sitting is typically 0.2-0.4 hps.
    max_descent_rate_hps_threshold: float = 0.8

    # Rule 3: hip height relative to standing anchor (1.0 = upright, ~0.3 = floor).
    height_ratio_threshold: float = 0.55

    # Rule 4: minimum lying score when pose IS available.
    lying_score_threshold: float = 0.4

    # Rule 5: maximum Kalman floor speed at event (m/s).
    # Values above this are almost certainly projection glitches.
    floor_speed_max_m_s: float = 2.0

    # Escalation: seconds after warning emission to check for post-event stillness.
    escalation_window_s: float = 60.0

    # Escalation: motion energy floor below which the person is considered still.
    # Matches dementia_signals.SignalConfig.stillness_motion_floor.
    stillness_motion_floor: float = 0.05

    # Cancel: height ratio above which the person is considered standing again.
    standing_clear_height_ratio: float = 0.75

    # Cancel: lying score below which the person is considered standing again.
    standing_clear_lying_score: float = 0.2

    # Cooldown: seconds before re-emitting a fall_suspected for the same PH
    # after an episode closes. 10 min expressed in seconds.
    cooldown_s: float = 600.0


@dataclass(frozen=True)
class FallDecision:
    """All impact rules fired for a PH.

    Returned by :meth:`FallDetector.check_impact` when a fall is detected.
    Carries derived values used by the stage to populate signal context.
    """

    descent_rate_hps: float
    height_ratio: float
    floor_speed_at_event_m_s: float | None


class FallDetector:
    """Deterministic rule-based fall detection over :class:`FallFeatures`.

    Stateless: all per-PH episode tracking lives in :class:`FallDetectionStage`.
    """

    def __init__(self, config: FallDetectorConfig | None = None) -> None:
        self._cfg = config or FallDetectorConfig()

    def check_impact(
        self,
        features: FallFeatures,
        room_name: str,
        resting_rooms: tuple[str, ...],
    ) -> FallDecision | None:
        """Return :class:`FallDecision` when all impact rules fire, else None."""
        cfg = self._cfg

        # Rule 1: sufficient samples in buffer
        if features.samples < cfg.min_samples:
            return None

        # Rule 2: rapid descent detected
        if features.max_descent_rate_hps < cfg.max_descent_rate_hps_threshold:
            return None

        # Rule 3: person is low relative to their standing height
        if features.height_ratio_now > cfg.height_ratio_threshold:
            return None

        # Rule 4: lying evidence OR pose unavailable (flat on floor loses keypoints)
        if features.pose_available_now and features.lying_score_now < cfg.lying_score_threshold:
            return None

        # Rule 5: floor speed within plausible range (eliminates projection glitches)
        if (
            features.floor_speed_at_event_m_s is not None
            and features.floor_speed_at_event_m_s > cfg.floor_speed_max_m_s
        ):
            return None

        # Rule 6: not in a resting room (bed-flop is dominant false positive)
        room_lower = room_name.lower()
        if any(rr.lower() in room_lower for rr in resting_rooms):
            return None

        return FallDecision(
            descent_rate_hps=features.max_descent_rate_hps,
            height_ratio=features.height_ratio_now,
            floor_speed_at_event_m_s=features.floor_speed_at_event_m_s,
        )

    def is_escalatable(self, features: FallFeatures) -> bool:
        """Return True if post-event stillness confirms the fall.

        When the post-event window is not yet complete (post_event_motion_nu_s
        is None), falls back to checking that posture remains lying/unknown
        and height is still low. This avoids waiting the full post_window_s
        before escalating in clear cases.
        """
        cfg = self._cfg
        post_motion = features.post_event_motion_nu_s
        if post_motion is None:
            # Window not complete; use posture-only proxy.
            return features.height_ratio_now <= cfg.height_ratio_threshold and (
                features.lying_score_now >= cfg.lying_score_threshold
                or not features.pose_available_now
            )
        return post_motion <= cfg.stillness_motion_floor

    def is_standing_cleared(self, features: FallFeatures) -> bool:
        """Return True if the person appears to have returned to standing/walking."""
        return (
            features.height_ratio_now >= self._cfg.standing_clear_height_ratio
            and features.lying_score_now <= self._cfg.standing_clear_lying_score
        )

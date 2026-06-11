"""FallDetectionStage: per-frame fall detection fast path.

Wired after PostureStage and before TrajectoryStage. Feeds
FallFeatureExtractor every frame, runs FallDetector rules, and on
detection emits a fall_suspected DementiaSignal immediately to the
tracking.signals stream and the signal repository without waiting for
the 60-second signal worker cycle.

Severity policy
---------------
- ``warning``:   impact rules fire (rapid descent + low height + lying evidence).
- ``emergency``: within escalation_window_s, post-event stillness or lying
  posture confirms the person has not risen.

Debounce
--------
SignalHysteresis with min_consecutive=1: falls must not wait two worker
runs (the 60-second worker is too slow for the clinically critical first
minutes). Every unique impact event is bucketed to a 10-second episode key
so repeated detections of the same fall do not re-page.

Unidentified PH
---------------
When current_identity_id is None (identity not yet committed) the signal
is suppressed; a ``fall_suspected_unidentified`` warning is logged and a
Prometheus counter is incremented. This is a known limitation: a fall
during the first few frames of a PH may be missed. Defense in depth:
the stillness pathway will catch the long-lie outcome regardless.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from structlog import get_logger

from ...domain import DementiaSignal, DementiaSignalKind, DementiaSignalSeverity
from ...observability import metrics as _metrics
from ...trajectory.dementia_signals import SignalHysteresis
from ...trajectory.fall_detector import FallDetector, FallDetectorConfig
from ...trajectory.fall_features import FallFeatureExtractor, FallFrameInput
from ...trajectory.signal_specs import FALL_SUSPECTED_SPEC
from ..frame_context import FrameContext
from .base import FrameStage

if TYPE_CHECKING:
    from ...storage.signals import DementiaSignalRepository
    from ...trajectory.motion_energy import MotionEnergyTracker
    from ...transport.signal_publisher import SignalPublisher

logger = get_logger(__name__)

_SIGNAL_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # NAMESPACE_URL
_FALL_KIND: DementiaSignalKind = "fall_suspected"
_EPISODE_BUCKET_S = 10  # seconds for episode_key bucketing


def _bucket_ts(ts: datetime, bucket_s: int = _EPISODE_BUCKET_S) -> datetime:
    """Floor a timestamp to the nearest bucket boundary."""
    epoch_s = ts.timestamp()
    bucketed = math.floor(epoch_s / bucket_s) * bucket_s
    return datetime.fromtimestamp(bucketed, tz=UTC)


def _stable_signal_id(identity_id: str, window_start: datetime, window_end: datetime) -> str:
    key = f"{identity_id}\x00{_FALL_KIND}\x00{window_start.isoformat()}\x00{window_end.isoformat()}"
    return str(uuid.uuid5(_SIGNAL_NS, key))


@dataclass(frozen=True)
class FallDetectionConfig:
    """Config for FallDetectionStage.

    All fields correspond to settings.yaml fall_detection: block keys.
    """

    enabled: bool = False
    # Detector thresholds (initial values; task 2.5 calibrates).
    min_samples: int = 5
    max_descent_rate_hps_threshold: float = 0.8
    height_ratio_threshold: float = 0.55
    lying_score_threshold: float = 0.4
    floor_speed_max_m_s: float = 2.0
    escalation_window_s: float = 60.0
    # Shared with dementia_signals: must stay in sync.
    stillness_motion_floor: float = 0.05
    standing_clear_height_ratio: float = 0.75
    standing_clear_lying_score: float = 0.2
    # Cooldown in minutes (converted to seconds in stage).
    cooldown_minutes: int = 10
    # Resting rooms: falls in these rooms are classified as bed-flops.
    resting_rooms: tuple[str, ...] = ("bed", "bedroom")


class FallDetectionStage(FrameStage):
    """Per-frame fall detection using pose, posture, and Kalman state.

    Consumes ctx.det_pose_result, ctx.det_posture_scores (PostureStage),
    and ctx.world_snapshots (WorldTrackingStage). Emits fall_suspected
    DementiaSignal immediately on detection; does not wait for the
    signal worker cycle.
    """

    name = "fall_detection"

    def __init__(
        self,
        config: FallDetectionConfig,
        signal_repo: DementiaSignalRepository,
        signal_publisher: SignalPublisher,
        motion_energy_tracker: MotionEnergyTracker | None = None,
    ) -> None:
        self._cfg = config
        self._signal_repo = signal_repo
        self._signal_publisher = signal_publisher
        self._motion_energy_tracker = motion_energy_tracker

        detector_cfg = FallDetectorConfig(
            min_samples=config.min_samples,
            max_descent_rate_hps_threshold=config.max_descent_rate_hps_threshold,
            height_ratio_threshold=config.height_ratio_threshold,
            lying_score_threshold=config.lying_score_threshold,
            floor_speed_max_m_s=config.floor_speed_max_m_s,
            escalation_window_s=config.escalation_window_s,
            stillness_motion_floor=config.stillness_motion_floor,
            standing_clear_height_ratio=config.standing_clear_height_ratio,
            standing_clear_lying_score=config.standing_clear_lying_score,
            cooldown_s=config.cooldown_minutes * 60.0,
        )
        self._detector = FallDetector(detector_cfg)
        self._extractor = FallFeatureExtractor()

        # min_consecutive=1: falls must not wait two runs; override documented here.
        # The 60-second worker debounce is clinically unacceptable for a fall event
        # where the first minutes are the most critical intervention window.
        self._hysteresis = SignalHysteresis(min_consecutive=1)

        # Per-PH episode state (not tracked by hysteresis itself).
        self._emission_at: dict[str, datetime] = {}  # ph_id → warning emit time
        self._episode_bucket: dict[str, datetime] = {}  # ph_id → bucketed impact ts
        self._standing_clear_frames: dict[str, int] = {}  # ph_id → consecutive clear count

    def evict(self, ph_id: str) -> None:
        """Free per-PH state (call on PH close, mirroring MotionEnergyTracker.evict_track)."""
        self._extractor.evict(ph_id)
        self._emission_at.pop(ph_id, None)
        self._episode_bucket.pop(ph_id, None)
        self._standing_clear_frames.pop(ph_id, None)

    async def run(self, ctx: FrameContext) -> None:
        now = ctx.event_time
        self._hysteresis.begin_run(now)

        snap_by_ph = {s.ph_id: s for s in ctx.world_snapshots}

        for det in ctx.domain_detections:
            ph_id = det.ph_id
            if not ph_id:
                continue

            snap = snap_by_ph.get(ph_id)
            if snap is None:
                continue

            # Build FallFrameInput from already-computed ctx fields.
            pose_result = ctx.det_pose_result.get(det.detection_id)
            posture_scores = ctx.det_posture_scores.get(det.detection_id)
            keypoints = tuple(pose_result.keypoints) if pose_result is not None else None

            # Floor speed from Kalman velocity (m/s = sqrt(vx^2 + vy^2)).
            floor_speed: float | None = None
            vx, vy = snap.floor_vx_m_s, snap.floor_vy_m_s
            if vx != 0.0 or vy != 0.0:
                floor_speed = (vx**2 + vy**2) ** 0.5

            # Read last-computed motion energy without updating the tracker
            # (TrajectoryStage, which runs after us, calls update() for the same frame).
            motion_energy_nu_s: float | None = None
            if self._motion_energy_tracker is not None:
                me = self._motion_energy_tracker.get_current_energy(ph_id)
                if me is not None:
                    motion_energy_nu_s = me.mean_keypoint_velocity_nu_s

            frame_input = FallFrameInput(
                captured_at=ctx.capture_time,
                bbox=det.bbox,
                keypoints=keypoints,
                posture_scores=posture_scores,
                floor_speed_m_s=floor_speed,
                motion_energy_nu_s=motion_energy_nu_s,
            )

            features = self._extractor.update(ph_id, frame_input)

            room_name = snap.room_name or ""
            decision = self._detector.check_impact(features, room_name, self._cfg.resting_rooms)

            if decision is None:
                # No impact: check if episode should be cleared.
                if self._detector.is_standing_cleared(features):
                    self._standing_clear_frames[ph_id] = (
                        self._standing_clear_frames.get(ph_id, 0) + 1
                    )
                    if self._standing_clear_frames[ph_id] >= 2:
                        identity_id = snap.identity_id
                        if identity_id and ph_id in self._emission_at:
                            episode_key = self._episode_bucket[ph_id].isoformat()
                            self._hysteresis.clear_trigger(identity_id, _FALL_KIND, episode_key)
                            self._emission_at.pop(ph_id, None)
                            self._episode_bucket.pop(ph_id, None)
                        self._standing_clear_frames.pop(ph_id, None)
                else:
                    self._standing_clear_frames.pop(ph_id, None)
                continue

            # Impact rules fired. Reset standing-clear counter.
            self._standing_clear_frames.pop(ph_id, None)

            identity_id = snap.identity_id
            if not identity_id:
                # Known limitation: PH has no committed identity yet.
                logger.warning(
                    "fall_suspected_unidentified",
                    ph_id=ph_id,
                    descent_rate=round(decision.descent_rate_hps, 3),
                )
                _metrics.metrics.cts_fall_suspected_unidentified_total.inc()
                continue

            # Determine severity: warning on impact, emergency on confirmed stillness.
            severity: DementiaSignalSeverity
            if ph_id in self._emission_at:
                elapsed = (now - self._emission_at[ph_id]).total_seconds()
                if elapsed <= self._cfg.escalation_window_s and self._detector.is_escalatable(
                    features
                ):
                    severity = "emergency"
                else:
                    severity = "warning"
            else:
                severity = "warning"

            # Episode key: bucket the current time to 10-second windows.
            if ph_id not in self._episode_bucket:
                self._episode_bucket[ph_id] = _bucket_ts(now)
            bucket_ts = self._episode_bucket[ph_id]
            episode_key = bucket_ts.isoformat()

            if not self._hysteresis.should_emit(
                identity_id,
                _FALL_KIND,
                severity,
                now,
                cooldown_minutes=self._cfg.cooldown_minutes,
                episode_key=episode_key,
            ):
                continue

            # Record emission time on first warning.
            if severity == "warning" and ph_id not in self._emission_at:
                self._emission_at[ph_id] = now

            await self._emit_signal(
                identity_id=identity_id,
                severity=severity,
                window_start=bucket_ts,
                window_end=bucket_ts + timedelta(seconds=_EPISODE_BUCKET_S),
                now=now,
                decision_descent_rate=decision.descent_rate_hps,
                decision_height_ratio=decision.height_ratio,
                floor_speed=decision.floor_speed_at_event_m_s,
                ph_id=ph_id,
                room_name=room_name,
            )

    async def _emit_signal(
        self,
        *,
        identity_id: str,
        severity: str,
        window_start: datetime,
        window_end: datetime,
        now: datetime,
        decision_descent_rate: float,
        decision_height_ratio: float,
        floor_speed: float | None,
        ph_id: str,
        room_name: str,
    ) -> None:
        signal_id = _stable_signal_id(identity_id, window_start, window_end)
        spec = FALL_SUSPECTED_SPEC
        signal = DementiaSignal(
            signal_id=signal_id,
            identity_id=identity_id,
            signal_kind=_FALL_KIND,
            severity=severity,  # type: ignore[arg-type]
            value=round(decision_descent_rate, 4),
            window_start=window_start,
            window_end=window_end,
            emitted_at=now,
            context={
                "ph_id": ph_id,
                "room_name": room_name,
                "height_ratio": round(decision_height_ratio, 4),
                "floor_speed_m_s": (round(floor_speed, 4) if floor_speed is not None else None),
            },
            algorithm_version=spec.version,
            algorithm_name=spec.name,
            evidence_grade=spec.evidence_grade,
            algorithm_spec_json=json.dumps(
                {
                    "name": spec.name,
                    "version": spec.version,
                    "evidence_grade": spec.evidence_grade,
                    "required_inputs": list(spec.required_inputs),
                    "disclaimer": spec.disclaimer,
                }
            ),
        )

        await self._signal_repo.upsert_signal(signal)
        await self._signal_publisher.publish_signal(signal)

        _metrics.metrics.cts_fall_suspected_total.labels(severity=severity).inc()
        _metrics.metrics.cts_fall_descent_rate.observe(decision_descent_rate)

        logger.info(
            "fall_suspected_emitted",
            signal_id=signal_id,
            identity_id=identity_id,
            ph_id=ph_id,
            severity=severity,
            descent_rate=round(decision_descent_rate, 3),
        )

"""Device-free anonymous position sensors (mmWave radar, room LiDAR) and their tracks.

A position sensor reports where people are on the shared floor plan, never who they
are. Cognitive Companion owns the registry and serves it in the site-config; a track
from a sensor that is not registered and active never reaches the world tracker.
Positions are metres in the household floor-plan frame; covariance is row-major
``(xx, xy, xy, yy)`` in square metres.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Literal, Protocol

PositionSensorMode = Literal["off", "shadow", "enabled"]


class ObservationSource(StrEnum):
    """What produced a ``WorldObservation``; the tracker branches on this, never on ids."""

    CAMERA = "camera"
    DEVICE_FREE = "device_free"


@dataclass(frozen=True)
class PositionSensor:
    """One registered, active position sensor from the CC site-config.

    ``x_mm``/``y_mm`` and ``yaw_deg`` are the mounting pose in the floor-plan frame
    (``yaw_deg`` is ``None`` when the gateway reports floor-plan coordinates).
    ``calibration_residual_m`` is the sensor's systematic placement error: the floor
    below which the tracker never lets its tracks shrink a PH's position uncertainty.
    """

    sensor_id: str
    room_id: str | None
    x_mm: int
    y_mm: int
    yaw_deg: float | None
    calibration_residual_m: float


@dataclass(frozen=True)
class AnonymousPositionObservation:
    """One validated floor-plane measurement of a person by a position sensor.

    ``sensor_track_id`` is the sensor's own short-lived target id; it groups samples of
    one target and carries no identity. ``calibration_residual_m`` comes from the
    sensor's registry entry.
    """

    sensor_id: str
    sensor_track_id: str
    measured_at: datetime
    x_m: float
    y_m: float
    cov_m2: tuple[float, float, float, float]
    calibration_residual_m: float
    floor_plan_id: str = ""
    speed_m_s: float | None = None

    @property
    def track_key(self) -> tuple[str, str]:
        """The ``(sensor_id, sensor_track_id)`` pair identifying one sensor target."""
        return (self.sensor_id, self.sensor_track_id)


class PositionTrackSource(Protocol):
    """Source of anonymous position tracks for device-free tracker rounds."""

    def take_round_tracks(
        self, *, now: datetime, max_age_s: float, min_track_age_s: float
    ) -> Sequence[AnonymousPositionObservation]:
        """Return each confirmed track's newest sample not yet handed to a round.

        Consuming: a sample is returned at most once, so no Kalman update ever counts
        the same measurement twice.
        """
        ...

    def peek_round_tracks(
        self, *, now: datetime, max_age_s: float, min_track_age_s: float
    ) -> Sequence[AnonymousPositionObservation]:
        """Return what ``take_round_tracks`` would return, without consuming it."""
        ...

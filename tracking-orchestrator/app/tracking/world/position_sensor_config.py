"""Configuration for device-free position sensor ingestion and device-free tracker rounds."""

from __future__ import annotations

from dataclasses import dataclass

from app.domain import PositionSensorMode

POSITION_SENSOR_MODES: tuple[PositionSensorMode, ...] = ("off", "shadow", "enabled")


@dataclass(frozen=True)
class PositionSensorConfig:
    """Parameters for ``position.tracks`` intake and the device-free round driver.

    ``mode``: "off" starts no subscriber and runs no device-free round, so live tracking is
    untouched; "shadow" ingests and buffers tracks and measures what a round would do against
    the open PHs without changing any state; "enabled" runs device-free rounds.
    """

    mode: PositionSensorMode = "off"
    # Ingest staleness horizon: a track sample older than this is dropped as stale.
    max_track_age_s: float = 5.0
    # How long samples are kept in the buffer.
    buffer_max_age_s: float = 10.0
    # Device-free round cadence while unconsumed confirmed tracks exist.
    round_interval_s: float = 0.5
    # A sample older than this is never handed to a round.
    round_max_track_age_s: float = 1.5
    # A sensor track enters rounds only once it has existed this long: radar multipath
    # ghosts are mostly shorter-lived, and a ghost that spawned a PH would be a phantom person.
    min_track_age_s: float = 1.0
    # A gap this long restarts a track's age, because sensors reuse track ids.
    track_gap_s: float = 2.0
    # Bound on buffered sensor tracks; the least recently heard is evicted first.
    max_tracks: int = 64

    def __post_init__(self) -> None:
        if self.mode not in POSITION_SENSOR_MODES:
            raise ValueError(f"position_sensor.mode must be one of {POSITION_SENSOR_MODES}")
        positive = {
            "max_track_age_s": self.max_track_age_s,
            "buffer_max_age_s": self.buffer_max_age_s,
            "round_interval_s": self.round_interval_s,
            "round_max_track_age_s": self.round_max_track_age_s,
            "track_gap_s": self.track_gap_s,
        }
        for name, value in positive.items():
            if not value > 0.0:
                raise ValueError(f"position_sensor.{name} must be positive")
        if self.min_track_age_s < 0.0:
            raise ValueError("position_sensor.min_track_age_s must not be negative")
        if self.round_max_track_age_s > self.buffer_max_age_s:
            raise ValueError(
                "position_sensor.round_max_track_age_s must not exceed buffer_max_age_s"
            )
        if self.max_tracks < 1:
            raise ValueError("position_sensor.max_tracks must be >= 1")

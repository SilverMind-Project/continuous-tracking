"""Central Prometheus metrics and structlog setup for rf-ingress.

All enum-valued Prometheus labels are closed ``StrEnum``s declared in this
module. Identity/member IDs and free-text strings never appear in any metric
label or log message.
"""

from __future__ import annotations

import logging
from enum import StrEnum

import structlog
from prometheus_client import Counter, Gauge, Histogram


class ReportKind(StrEnum):
    """Closed report-type label for ``rf_ingress_reports_total``."""

    RANGE = "range"
    POSITION = "position"
    ROOM = "room"
    POSITION_TRACK = "position_track"


class DropReason(StrEnum):
    """Closed drop-reason label for ``rf_ingress_fixes_dropped_total``."""

    NO_SITE_CONFIG = "no_site_config"
    UNREGISTERED_TAG = "unregistered_tag"
    INACTIVE_DEVICE = "inactive_device"
    UNBOUND_TAG = "unbound_tag"
    SOLVER_REJECTED = "solver_rejected"
    MISSING_FRAME_TRANSFORM = "missing_frame_transform"
    MALFORMED_INPUT = "malformed_input"
    SITE_CONFIG_RELOADED = "site_config_reloaded"
    INVALID_CONTRACT = "invalid_contract"
    RANGE_OUT_OF_BOUNDS = "range_out_of_bounds"
    LATE_EPOCH_MEASUREMENT = "late_epoch_measurement"
    PUBLISH_FAILED = "publish_failed"


class PositionTrackDropReason(StrEnum):
    """Closed drop-reason label for ``rf_ingress_position_tracks_dropped_total``."""

    NO_SITE_CONFIG = "no_site_config"
    UNREGISTERED_SENSOR = "unregistered_sensor"
    TOPIC_SENSOR_MISMATCH = "topic_sensor_mismatch"
    MISSING_SENSOR_YAW = "missing_sensor_yaw"
    FLOOR_PLAN_MISMATCH = "floor_plan_mismatch"
    INVALID_COVARIANCE = "invalid_covariance"
    MALFORMED_INPUT = "malformed_input"
    INVALID_CONTRACT = "invalid_contract"
    PUBLISH_FAILED = "publish_failed"


REPORTS_TOTAL = Counter(
    "rf_ingress_reports_total",
    "Total internal RF reports received from adapters",
    labelnames=("adapter_id", "type"),
)

FIXES_PUBLISHED_TOTAL = Counter(
    "rf_ingress_fixes_published_total",
    "Total hardware-neutral TagFix messages published to wearable.fixes",
    labelnames=("kind",),
)

FIXES_DROPPED_TOTAL = Counter(
    "rf_ingress_fixes_dropped_total",
    "Total candidate fixes or reports dropped before publishing",
    labelnames=("reason",),
)

POSITION_TRACKS_PUBLISHED_TOTAL = Counter(
    "rf_ingress_position_tracks_published_total",
    "Total AnonymousPositionTrack messages published to position.tracks",
    labelnames=("adapter_id",),
)

POSITION_TRACKS_DROPPED_TOTAL = Counter(
    "rf_ingress_position_tracks_dropped_total",
    "Total anonymous position tracks dropped before publishing",
    labelnames=("reason",),
)

UNREGISTERED_TAG_TOTAL = Counter(
    "rf_ingress_unregistered_tag_total",
    "Total reports dropped because the tag or hardware key is not registered in site config",
    labelnames=("adapter_id",),
)

OFF_BODY_FIXES_TOTAL = Counter(
    "rf_ingress_off_body_fixes_total",
    "Total published fixes carrying wear_state=off_body (passed through truthfully to CTS)",
    labelnames=("adapter_id",),
)

ANCHOR_OUTLIERS_TOTAL = Counter(
    "rf_ingress_anchor_outliers_total",
    "Total epochs where an anchor range was rejected as a chi-square outlier",
    labelnames=("anchor_id",),
)

SITE_CONFIG_POLL_FAILURES_TOTAL = Counter(
    "rf_ingress_site_config_poll_failures_total",
    "Total failed Cognitive Companion site-config polls",
)

SOLVE_SECONDS = Histogram(
    "rf_ingress_solve_seconds",
    "Wall-clock duration of multilateration epoch solves in seconds",
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)

FIX_SIGMA_M = Histogram(
    "rf_ingress_fix_sigma_m",
    "Horizontal 1-sigma uncertainty of published position fixes in metres",
    buckets=(0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5, 4.0, 8.0),
)

CLOCK_SKEW_SECONDS = Gauge(
    "rf_ingress_clock_skew_seconds",
    "Estimated median (arrival - device) clock skew per adapter over the sliding window",
    labelnames=("adapter_id",),
)

DEVICE_CLOCK_OFFSET_MS = Gauge(
    "rf_ingress_device_clock_offset_ms",
    "Estimated minimum-delay clock offset in milliseconds per registered tag_id",
    labelnames=("tag_id",),
)


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structured JSON logging via structlog."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

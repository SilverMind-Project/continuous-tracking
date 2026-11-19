"""Wire-contract validation helpers for wearable protobuf messages.

Enforces invariants that protobuf syntax alone cannot express:
- ``FIX_KIND_POSITION`` fixes must carry ``has_position=True`` and a finite,
  positive-definite 2x2 covariance matrix in square metres.
- Optional scalar fields with explicit presence flags must carry finite,
  in-range values when their presence flag is set.
"""

from __future__ import annotations

import math

from app.proto.continuoustracking.v1 import wearable_pb2

FIELD_WEARABLE_FIX = "fix"
FIELD_WEARABLE_CAPABILITY = "capability"
WEARABLE_STREAM_RETENTION_S = 72 * 3600


class WearableContractError(ValueError):
    """Raised when a ``TagFix`` violates the hardware-neutral wire contract."""


def validate_tag_fix(fix: wearable_pb2.TagFix) -> None:
    """Validate *fix* against the ``TagFix`` wire-contract invariants."""
    if not fix.tag_id:
        raise WearableContractError("TagFix.tag_id must be non-empty")
    if fix.measured_at_unix_ns == 0:
        raise WearableContractError("TagFix.measured_at_unix_ns must be non-zero")
    if fix.kind == wearable_pb2.FIX_KIND_UNSPECIFIED:
        raise WearableContractError("TagFix.kind must not be FIX_KIND_UNSPECIFIED")

    if fix.kind == wearable_pb2.FIX_KIND_POSITION:
        if not fix.has_position:
            raise WearableContractError("FIX_KIND_POSITION requires has_position=True")
        xx = fix.cov_xx_m2
        xy = fix.cov_xy_m2
        yy = fix.cov_yy_m2
        if not (math.isfinite(xx) and math.isfinite(xy) and math.isfinite(yy)):
            raise WearableContractError("FIX_KIND_POSITION covariance values must be finite")
        det = xx * yy - xy * xy
        if xx <= 0.0 or yy <= 0.0 or det <= 0.0:
            raise WearableContractError(
                "FIX_KIND_POSITION requires a positive-definite covariance "
                "(cov_xx_m2 > 0, cov_yy_m2 > 0, det > 0)"
            )

    if fix.has_room_probability and not (0.0 <= fix.room_probability <= 1.0):
        raise WearableContractError("TagFix.room_probability must be in [0, 1]")
    if fix.has_distance and (not math.isfinite(fix.distance_m) or fix.distance_m < 0.0):
        raise WearableContractError("TagFix.distance_m must be finite and non-negative")
    if fix.has_quality and not (0.0 <= fix.quality <= 1.0):
        raise WearableContractError("TagFix.quality must be in [0, 1]")
    if fix.has_residual and (not math.isfinite(fix.residual_rms_m) or fix.residual_rms_m < 0.0):
        raise WearableContractError("TagFix.residual_rms_m must be finite and non-negative")
    if fix.has_battery and not (0.0 <= fix.battery_fraction <= 1.0):
        raise WearableContractError("TagFix.battery_fraction must be in [0, 1]")

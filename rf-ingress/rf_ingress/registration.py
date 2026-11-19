"""Registration, active-status, and time-bounded binding gate for emitted fixes.

A fix is published only when a valid site config exists, the tag is registered
and active, and a binding covers ``measured_at`` (half-open ``[valid_from,
valid_to)``). Unbound or unregistered devices carry no identity meaning and are
dropped and counted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from rf_ingress.domain import SiteConfigSnapshot, ensure_utc
from rf_ingress.observability import FIXES_DROPPED_TOTAL, UNREGISTERED_TAG_TOTAL, DropReason


@dataclass(frozen=True)
class RegistrationDecision:
    """Outcome of checking whether a tag may publish a fix at ``measured_at``."""

    allowed: bool
    drop_reason: DropReason | None = None


class RegistrationGate:
    """Fail-closed publication gate backed by the latest site-config snapshot."""

    def check(
        self,
        *,
        snapshot: SiteConfigSnapshot | None,
        tag_id: str,
        adapter_id: str,
        measured_at: datetime,
        record_metrics: bool = True,
    ) -> RegistrationDecision:
        """Return whether *tag_id* is registered, active, and bound at *measured_at*."""
        measured_utc = ensure_utc(measured_at)
        if snapshot is None:
            if record_metrics:
                FIXES_DROPPED_TOTAL.labels(reason=DropReason.NO_SITE_CONFIG).inc()
            return RegistrationDecision(allowed=False, drop_reason=DropReason.NO_SITE_CONFIG)

        device = snapshot.device_by_tag_id(tag_id)
        if device is None:
            if record_metrics:
                UNREGISTERED_TAG_TOTAL.labels(adapter_id=adapter_id).inc()
                FIXES_DROPPED_TOTAL.labels(reason=DropReason.UNREGISTERED_TAG).inc()
            return RegistrationDecision(allowed=False, drop_reason=DropReason.UNREGISTERED_TAG)

        if not device.is_active:
            if record_metrics:
                FIXES_DROPPED_TOTAL.labels(reason=DropReason.INACTIVE_DEVICE).inc()
            return RegistrationDecision(allowed=False, drop_reason=DropReason.INACTIVE_DEVICE)

        if snapshot.member_at(tag_id, measured_utc) is None:
            if record_metrics:
                FIXES_DROPPED_TOTAL.labels(reason=DropReason.UNBOUND_TAG).inc()
            return RegistrationDecision(allowed=False, drop_reason=DropReason.UNBOUND_TAG)

        return RegistrationDecision(allowed=True, drop_reason=None)

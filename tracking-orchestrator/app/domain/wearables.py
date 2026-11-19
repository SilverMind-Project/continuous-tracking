"""Hardware-neutral wearable and RF-node site configuration domain objects.

All floor-plane positions are expressed in millimetres in the shared household
floor-plan frame, matching :class:`app.domain.FloorPoint`. Time boundaries on
bindings are timezone-aware UTC datetimes representing a half-open interval
``[valid_from, valid_to)`` (open-ended when ``valid_to`` is ``None``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from cts_contracts import DeviceKind, RangingTechnology, RfNodeRole


def _to_utc(dt: datetime) -> datetime:
    """Normalize a timezone-aware datetime to UTC; reject naive datetimes."""
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("Wearable binding timestamps must be timezone-aware")
    return dt.astimezone(UTC)


@dataclass(frozen=True)
class RfNode:
    """One fixed RF node (receiver or ranging anchor) in the shared floor frame.

    Coordinates ``x_mm``, ``y_mm``, and ``z_mm`` are millimetres in the
    household floor-plan frame (``x_mm``/``y_mm`` on the floor plane, ``z_mm``
    mounting height above the floor).
    """

    anchor_id: str
    adapter_id: str
    node_role: RfNodeRole
    technology: RangingTechnology
    x_mm: int
    y_mm: int
    z_mm: int
    room_id: str | None = None
    uwb_address_hex: str | None = None
    uwb_channel: int | None = None
    uwb_preamble_index: int | None = None


@dataclass(frozen=True)
class WearableDevice:
    """One registered wearable device known to Cognitive Companion."""

    tag_id: str
    adapter_id: str
    hardware_key: str
    device_kind: DeviceKind
    is_active: bool


@dataclass(frozen=True)
class WearableBinding:
    """Time-bounded binding of a wearable ``tag_id`` to a household ``member_id``.

    Valid on the half-open UTC interval ``[valid_from, valid_to)`` (inclusive
    start, exclusive end; open-ended when ``valid_to`` is ``None``).
    """

    tag_id: str
    member_id: str
    valid_from: datetime
    valid_to: datetime | None = None

    def covers(self, at: datetime) -> bool:
        """Return ``True`` when *at* falls inside ``[valid_from, valid_to)``."""
        ts = _to_utc(at)
        start = _to_utc(self.valid_from)
        if ts < start:
            return False
        if self.valid_to is None:
            return True
        return ts < _to_utc(self.valid_to)


@dataclass(frozen=True)
class WearableSiteConfig:
    """Immutable snapshot of household RF nodes, devices, and bindings from CC.

    ``version`` is the content hash produced by Cognitive Companion so consumers
    can detect configuration changes atomically and never mix two floor-plan
    scales within one epoch.
    """

    floor_plan_id: str | None = None
    nodes: tuple[RfNode, ...] = ()
    bindings: tuple[WearableBinding, ...] = ()
    devices: tuple[WearableDevice, ...] = ()
    version: str = ""

    def member_at(self, tag_id: str, at: datetime) -> str | None:
        """Resolve which ``member_id`` wore *tag_id* at timestamp *at*, or ``None``.

        Boundaries follow half-open interval semantics: ``valid_from`` is
        inclusive and ``valid_to`` is exclusive.
        """
        for binding in self.bindings:
            if binding.tag_id == tag_id and binding.covers(at):
                return binding.member_id
        return None

"""Hardware-neutral wearable and RF-node site configuration domain objects.

All floor-plane positions are expressed in millimetres in the shared household
floor-plan frame, matching :class:`app.domain.FloorPoint`. Time boundaries on
bindings are timezone-aware UTC datetimes representing a half-open interval
``[valid_from, valid_to)`` (open-ended when ``valid_to`` is ``None``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Literal, Protocol

from cts_contracts import DeviceKind, FixKind, MotionState, RangingTechnology, RfNodeRole, WearState

from app.domain.position_sensors import PositionSensor

WearableMode = Literal["off", "shadow", "enabled"]
WEARABLE_MODES: tuple[WearableMode, ...] = ("off", "shadow", "enabled")


class WearableEvidenceClass(StrEnum):
    """Evidence class derived purely from fix kind and covariance geometry."""

    TRAJECTORY = "trajectory"
    ROOM = "room"


class WearableAttestationGrade(StrEnum):
    """Attestation tier reached by a sustained wearable-to-PH association."""

    NONE = "none"
    ROOM = "room"
    TRAJECTORY = "trajectory"


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
    """One registered wearable device known to Cognitive Companion.

    ``physical_device_id`` groups the logical tags of one radio (a dual-mode watch's
    BLE and UWB tags); ``None`` means the tag is its own physical device.
    """

    tag_id: str
    adapter_id: str
    hardware_key: str
    device_kind: DeviceKind
    is_active: bool
    physical_device_id: str | None = None


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
    """Immutable snapshot of household RF nodes, devices, bindings, and position sensors from CC.

    ``version`` is the content hash produced by Cognitive Companion so consumers
    can detect configuration changes atomically and never mix two floor-plan
    scales within one epoch.
    """

    floor_plan_id: str | None = None
    nodes: tuple[RfNode, ...] = ()
    bindings: tuple[WearableBinding, ...] = ()
    devices: tuple[WearableDevice, ...] = ()
    position_sensors: tuple[PositionSensor, ...] = ()
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

    def position_sensor(self, sensor_id: str) -> PositionSensor | None:
        """Return the registered, active position sensor *sensor_id*, or ``None``."""
        for sensor in self.position_sensors:
            if sensor.sensor_id == sensor_id:
                return sensor
        return None

    def physical_device_of(self, tag_id: str) -> str | None:
        """Return the ``physical_device_id`` shared by *tag_id*'s radio, or ``None``."""
        for device in self.devices:
            if device.tag_id == tag_id:
                return device.physical_device_id
        return None

    def sibling_tag_ids(self, tag_id: str) -> tuple[str, ...]:
        """Return every logical tag of *tag_id*'s physical device, *tag_id* first.

        A tag without a ``physical_device_id`` (or unknown to the config) is its own device.
        """
        physical = self.physical_device_of(tag_id)
        if physical is None:
            return (tag_id,)
        others = sorted(
            d.tag_id
            for d in self.devices
            if d.physical_device_id == physical and d.tag_id != tag_id
        )
        return (tag_id, *others)


@dataclass(frozen=True)
class WearableFix:
    """One validated, member-bound wearable localization fix from ``wearable.fixes``.

    Positions are in metres in the shared household floor-plan frame; covariance
    is row-major ``(cov_xx, cov_xy, cov_xy, cov_yy)`` in square metres. Absence
    is represented by ``None`` (enforced from protobuf presence flags at decode).
    """

    tag_id: str
    member_id: str
    measured_at: datetime
    kind: FixKind
    x_m: float | None = None
    y_m: float | None = None
    cov_m2: tuple[float, float, float, float] | None = None
    floor_plan_id: str = ""
    room_id: str | None = None
    room_probability: float | None = None
    anchor_id: str | None = None
    distance_m: float | None = None
    wear_state: WearState = WearState.UNSPECIFIED
    motion_state: MotionState = MotionState.UNSPECIFIED
    quality: float | None = None
    anchors_used: int = 0
    residual_rms_m: float | None = None
    adapter_id: str = ""
    epoch_seq: int = 0
    received_at: datetime | None = None
    # Shared by the logical tags of one radio; stamped at decode from the site config.
    physical_device_id: str | None = None

    @property
    def device_key(self) -> str:
        """Identity of the physical radio: its ``physical_device_id``, else the tag itself."""
        return self.physical_device_id or f"tag:{self.tag_id}"


@dataclass(frozen=True)
class WearableEvidence:
    """Accepted wearable association evidence for one PH in one tracker round."""

    ph_id: str
    tag_id: str
    identity_id: str
    evidence_class: WearableEvidenceClass
    attestation_grade: WearableAttestationGrade
    window_llr: float
    margin_llr: float
    matched_fixes: int
    mean_residual_m: float | None
    newest_fix_at: datetime
    # Persisted association segment this evidence belongs to; provenance for anything the
    # evidence teaches (gallery seeds, attested faces), so a wrong binding is undone per segment.
    segment_id: str | None = None

    @property
    def attested(self) -> bool:
        """Return True when this pair has reached room or trajectory attestation."""
        return self.attestation_grade != WearableAttestationGrade.NONE


@dataclass(frozen=True)
class WearableAssociationSegment:
    """Persisted interval during which a wearable tag was associated with a PH."""

    segment_id: str
    ph_id: str
    tag_id: str
    identity_id: str
    started_at: datetime
    ended_at: datetime | None
    attestation_grade: WearableAttestationGrade
    window_llr: float
    margin_llr: float
    matched_fixes: int
    mean_residual_m: float | None
    swap_suspected: bool = False
    close_reason: str | None = None
    revoked_at: datetime | None = None
    revoked_by: str | None = None
    revoke_reason: str | None = None


@dataclass(frozen=True)
class WearableTeachingHold:
    """Pause on wearable-driven learning for one tag until an operator resumes it."""

    hold_id: str
    tag_id: str
    held_at: datetime
    reason: str
    cleared_at: datetime | None = None
    cleared_by: str | None = None


@dataclass(frozen=True)
class WearableAssociationTransition:
    """One lifecycle change to a wearable association segment in a tracker round."""

    action: Literal["open", "update", "close"]
    segment: WearableAssociationSegment


@dataclass(frozen=True)
class ImageOnlyEvaluation:
    """What image-only identity decided for one attested PH in one sampled round.

    The truth is the attested tag's member. All arms leave out the wearable channel and
    every gallery row seen after ``captured_at - exclusion``; ``control_identity_id`` also leaves
    out every wearable-taught row, ``treatment_identity_id`` keeps them, and when
    ``classifier_evaluated`` is True ``classifier_identity_id`` additionally includes the
    household appearance classifier channel. ``None`` is Unknown. ``captured_at`` is the round's
    capture time.
    """

    ph_id: str
    truth_identity_id: str
    attestation_grade: WearableAttestationGrade
    control_identity_id: str | None
    treatment_identity_id: str | None
    captured_at: datetime
    classifier_identity_id: str | None = None
    classifier_evaluated: bool = False


@dataclass(frozen=True)
class ImageOnlyDailyTotal:
    """Accumulated image-only seconds for one identity, UTC day, arm, and attestation grade."""

    day: date
    identity_id: str
    arm: str
    attestation_grade: WearableAttestationGrade
    correct_s: float = 0.0
    wrong_s: float = 0.0
    unknown_s: float = 0.0


class WearableFixSource(Protocol):
    """Read-only source of recent member-bound wearable fixes for world tracking."""

    def recent(self, *, now: datetime, max_age_s: float) -> Sequence[WearableFix]: ...

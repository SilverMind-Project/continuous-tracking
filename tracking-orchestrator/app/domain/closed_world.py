"""Domain types for closed-world household presence and joint elimination.

All domain records are frozen dataclasses. Dimensional fields name their unit
(``_m``, ``_s``) and reference frame (shared household floor plane in metres).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from cts_contracts.household import PresenceStatus

ClosedWorldMode = Literal["off", "shadow", "enabled"]
CLOSED_WORLD_MODES: tuple[ClosedWorldMode, ...] = ("off", "shadow", "enabled")


@dataclass(frozen=True)
class MemberPresenceSnapshot:
    """One household member's fused presence state received from Cognitive Companion."""

    member_id: str
    status: PresenceStatus
    confidence: float
    room_id: str | None
    inferred_at: datetime


@dataclass(frozen=True)
class HouseholdPresenceSnapshot:
    """Latest household presence and roster-completeness snapshot from ``cc.household_presence``."""

    members: tuple[MemberPresenceSnapshot, ...]
    roster_complete: bool
    visitors_present: bool
    published_at: datetime
    received_at: datetime

    def by_member_id(self) -> dict[str, MemberPresenceSnapshot]:
        """Return member snapshots keyed by ``member_id``."""
        return {m.member_id: m for m in self.members}

    def member(self, member_id: str) -> MemberPresenceSnapshot | None:
        """Return the snapshot for *member_id*, or ``None`` if absent."""
        for m in self.members:
            if m.member_id == member_id:
                return m
        return None


@runtime_checkable
class HouseholdPresenceSource(Protocol):
    """Read-only source of the latest household presence snapshot."""

    def latest(self) -> HouseholdPresenceSnapshot | None: ...


@dataclass(frozen=True)
class OpenPHOccupancy:
    """Per-open-PH occupancy and spatial state used by joint elimination.

    ``floor_x_m`` and ``floor_y_m`` are floor-plane coordinates in metres
    (shared household floor frame) and are meaningful only when
    ``floor_calibrated`` is True.
    """

    ph_id: str
    identity_id: str | None
    last_independent_evidence_at: datetime | None
    last_seen_at: datetime
    coasting: bool = False
    floor_calibrated: bool = False
    floor_x_m: float = 0.0
    floor_y_m: float = 0.0
    room_id: str = ""
    last_seen_camera: str = ""
    active_cameras: frozenset[str] = frozenset()
    elim_eligible: bool = True

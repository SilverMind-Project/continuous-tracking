"""Hardware-neutral adapter protocols, dependency bundle, and health status.

Every RF hardware integration implements one or more of :class:`RangeAdapter`,
:class:`PositionAdapter`, and :class:`RoomAdapter`, declares an accurate
:class:`rf_ingress.domain.AdapterCapabilityDescriptor`, and passes the adapter
conformance suite under ``tests/conformance/`` unchanged.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from rf_ingress.domain import (
    AdapterCapabilityDescriptor,
    PositionReport,
    RangeMeasurement,
    RoomReport,
)
from rf_ingress.site_config import SiteConfigProvider


@dataclass(frozen=True)
class AdapterHealth:
    """Point-in-time adapter liveness exposed by ``GET /health``."""

    adapter_id: str
    connected: bool
    last_report_at: datetime | None
    malformed_reports: int = 0
    unregistered_reports: int = 0


@dataclass(frozen=True)
class AdapterDeps:
    """Explicit dependencies injected into every adapter factory by the registry."""

    site_config: SiteConfigProvider
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)


@runtime_checkable
class BaseAdapter(Protocol):
    """Lifecycle and capability metadata shared by every rf-ingress adapter."""

    adapter_id: str
    capability: AdapterCapabilityDescriptor

    def health(self) -> AdapterHealth: ...

    async def close(self) -> None: ...


@runtime_checkable
class RangeAdapter(BaseAdapter, Protocol):
    """Adapter emitting tag-to-node distance measurements."""

    def ranges(self) -> AsyncIterator[RangeMeasurement]: ...


@runtime_checkable
class PositionAdapter(BaseAdapter, Protocol):
    """Adapter emitting positions already solved in a native vendor frame."""

    def positions(self) -> AsyncIterator[PositionReport]: ...


@runtime_checkable
class RoomAdapter(BaseAdapter, Protocol):
    """Adapter emitting room-level presence reports in CC room vocabulary."""

    def rooms(self) -> AsyncIterator[RoomReport]: ...


AnyAdapter = RangeAdapter | PositionAdapter | RoomAdapter

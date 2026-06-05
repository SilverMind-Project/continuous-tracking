"""Thread-safe live transit-zone map.

Populated by ``CCConfigSyncService`` on every poll cycle from CC's transit
zone registry. Stored polygons and direction vectors are in floor-plan metres.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime

from ..domain import TransitZone


@dataclass(frozen=True)
class TransitZoneBinding:
    zone: TransitZone
    bound_at: datetime


class TransitZoneMap:
    """Thread-safe live map of transit zones in floor-plan metre coordinates."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._zones: dict[str, TransitZoneBinding] = {}
        self._version: int = 0

    async def set_all(self, bindings: list[TransitZoneBinding]) -> None:
        async with self._lock:
            self._zones = {binding.zone.zone_id: binding for binding in bindings}
            self._version += 1

    async def snapshot(self) -> list[TransitZone]:
        async with self._lock:
            return [binding.zone for binding in self._zones.values()]

    async def version(self) -> int:
        async with self._lock:
            return self._version

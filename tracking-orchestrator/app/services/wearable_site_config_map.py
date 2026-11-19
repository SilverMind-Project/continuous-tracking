"""Thread-safe live holder for the synced wearable site configuration.

Populated and refreshed by :class:`app.services.cc_config_sync.CCConfigSyncService`
from Cognitive Companion's ``GET /api/v1/wearables/site-config`` endpoint. When an
identity is purged via :class:`app.services.identity_roster_sync.IdentityRosterSyncService`,
in-memory bindings for that identity are removed immediately so bound tags stop
resolving to a deleted member before the next poll cycle.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

from app.domain.wearables import WearableSiteConfig


class WearableSiteConfigMap:
    """Thread-safe live holder for :class:`WearableSiteConfig`.

    Supports both async reads and lock-free synchronous reads (``get_sync``)
    because ``set_config`` and ``purge_identity`` atomically swap the frozen
    snapshot reference.
    """

    def __init__(self, initial: WearableSiteConfig | None = None) -> None:
        self._lock = asyncio.Lock()
        self._config: WearableSiteConfig = initial if initial is not None else WearableSiteConfig()

    async def get(self) -> WearableSiteConfig:
        async with self._lock:
            return self._config

    def get_sync(self) -> WearableSiteConfig:
        """Return the latest immutable site config without awaiting the lock."""
        return self._config

    async def set_config(self, config: WearableSiteConfig) -> None:
        async with self._lock:
            self._config = config

    def purge_identity(self, identity_id: str) -> int:
        """Drop every in-memory binding for *identity_id*; return count removed."""
        current = self._config
        kept = tuple(b for b in current.bindings if b.member_id != identity_id)
        removed = len(current.bindings) - len(kept)
        if removed > 0:
            self._config = replace(current, bindings=kept)
        return removed

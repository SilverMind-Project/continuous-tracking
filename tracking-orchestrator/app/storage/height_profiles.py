"""Storage protocols and in-memory peers for identity height profiles and PH height samples.

Postgres peers live in ``app/storage/postgres/height_profile_repo.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from app.domain import IdentityHeightProfile, PHHeightSample


class IdentityHeightProfileRepository(Protocol):
    """Per-identity standing height profile store (operator-entered and learned)."""

    async def get(self, identity_id: str) -> IdentityHeightProfile | None: ...

    async def list_all(self) -> list[IdentityHeightProfile]: ...

    async def upsert(self, profile: IdentityHeightProfile) -> None:
        """Insert or replace the stored profile for ``profile.identity_id``."""
        ...

    async def upsert_operator_if_not_learned(
        self,
        identity_id: str,
        *,
        height_m: float,
        height_sigma_m: float,
        updated_at: datetime,
    ) -> bool:
        """Write an ``operator`` profile unless a ``learned`` profile already exists.

        Returns ``True`` when the profile was inserted or updated, ``False`` when a
        learned profile took precedence or the identity does not exist yet.
        """
        ...

    async def delete_operator_profile(self, identity_id: str) -> bool:
        """Delete an ``operator`` profile (never removes a ``learned`` profile)."""
        ...

    async def delete(self, identity_id: str) -> bool:
        """Delete any profile for *identity_id* (used on identity purge)."""
        ...


class InMemoryIdentityHeightProfileRepository:
    """In-memory peer of :class:`IdentityHeightProfileRepository` for unit tests."""

    def __init__(self, initial: Sequence[IdentityHeightProfile] = ()) -> None:
        self._lock = asyncio.Lock()
        self._profiles: dict[str, IdentityHeightProfile] = {p.identity_id: p for p in initial}

    async def get(self, identity_id: str) -> IdentityHeightProfile | None:
        return self._profiles.get(identity_id)

    async def list_all(self) -> list[IdentityHeightProfile]:
        return sorted(self._profiles.values(), key=lambda p: p.identity_id)

    async def upsert(self, profile: IdentityHeightProfile) -> None:
        async with self._lock:
            self._profiles[profile.identity_id] = profile

    async def upsert_operator_if_not_learned(
        self,
        identity_id: str,
        *,
        height_m: float,
        height_sigma_m: float,
        updated_at: datetime,
    ) -> bool:
        async with self._lock:
            existing = self._profiles.get(identity_id)
            if existing is not None and existing.source == "learned":
                return False
            self._profiles[identity_id] = IdentityHeightProfile(
                identity_id=identity_id,
                height_m=height_m,
                height_sigma_m=height_sigma_m,
                source="operator",
                samples=0,
                cameras=0,
                updated_at=updated_at,
            )
            return True

    async def delete_operator_profile(self, identity_id: str) -> bool:
        async with self._lock:
            existing = self._profiles.get(identity_id)
            if existing is None or existing.source != "operator":
                return False
            del self._profiles[identity_id]
            return True

    async def delete(self, identity_id: str) -> bool:
        async with self._lock:
            return self._profiles.pop(identity_id, None) is not None


class PHHeightSampleRepository(Protocol):
    """Append-only store of accepted standing height measurements per PH."""

    async def append_many(self, rows: Sequence[PHHeightSample]) -> None:
        """Persist accepted samples. An empty sequence is a no-op."""
        ...

    async def list_between(
        self,
        start: datetime,
        end: datetime,
        *,
        ph_id: str | None = None,
        camera_id: str | None = None,
    ) -> list[PHHeightSample]:
        """Return samples with ``start <= observed_at < end``, oldest first."""
        ...


class InMemoryPHHeightSampleRepository:
    """In-memory peer of :class:`PHHeightSampleRepository` for unit tests."""

    def __init__(self, initial: Sequence[PHHeightSample] = ()) -> None:
        self._lock = asyncio.Lock()
        self._rows: list[PHHeightSample] = list(initial)

    async def append_many(self, rows: Sequence[PHHeightSample]) -> None:
        if not rows:
            return
        async with self._lock:
            self._rows.extend(rows)

    async def list_between(
        self,
        start: datetime,
        end: datetime,
        *,
        ph_id: str | None = None,
        camera_id: str | None = None,
    ) -> list[PHHeightSample]:
        matched = [
            r
            for r in self._rows
            if start <= r.observed_at < end
            and (ph_id is None or r.ph_id == ph_id)
            and (camera_id is None or r.camera_id == camera_id)
        ]
        return sorted(matched, key=lambda r: (r.observed_at, r.ph_id, r.camera_id))

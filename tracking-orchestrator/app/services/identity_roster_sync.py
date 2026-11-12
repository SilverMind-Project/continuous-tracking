"""Periodic synchronization and cleanup of enrolled identities and anchors.

Reconciles the continuous-tracking identity roster against the authoritative
enrollment store in person-identification-service:
- Upserts newly enrolled members so they are known to the Bayesian resolver
  before their first face observation arrives.
- Purges stale identities and anchors for people no longer enrolled in the
  system, invalidating in-memory resolver and gallery caches.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from structlog import get_logger

from ..domain import Identity

if TYPE_CHECKING:
    from ..pipeline.gallery_cache import GalleryCache
    from ..storage.gallery import GalleryRepository
    from ..tracking.identity_resolver import IdentityResolver

logger = get_logger(__name__)


class EnrolledMembersSource(Protocol):
    """Protocol for fetching currently enrolled members."""

    async def list_enrolled_members(self) -> list[tuple[str, str]] | None: ...


class FaceTemplateIdentityReset(Protocol):
    """Drops in-memory face templates tied to an identity (the face template service)."""

    def reset_identity(self, identity_id: str) -> None: ...


class IdentityRosterSyncService:
    """Reconciles CTS identities with the enrolled household roster."""

    def __init__(
        self,
        gallery_repo: GalleryRepository,
        members_source: EnrolledMembersSource | None = None,
        *,
        identity_resolver: IdentityResolver | None = None,
        gallery_cache: GalleryCache | None = None,
        face_templates: FaceTemplateIdentityReset | None = None,
        interval_s: float = 60.0,
    ) -> None:
        self._gallery_repo = gallery_repo
        # Face templates hold faces recognized as the purged identity; they go with it.
        self._face_templates = face_templates
        self._members_source = members_source
        self._identity_resolver = identity_resolver
        self._gallery_cache = gallery_cache
        self._interval_s = interval_s
        self._task: asyncio.Task[None] | None = None

    async def purge_identity(self, identity_id: str) -> bool:
        """Purge one unenrolled identity and invalidate in-memory caches."""
        removed = await self._gallery_repo.purge_identity(identity_id)
        if removed:
            self._invalidate_caches()
            self._reset_face_templates(identity_id)
            logger.info("identity_purged_from_roster")
        return removed

    async def sync_once(self) -> tuple[int, int]:
        """Run one roster reconciliation pass.

        Returns ``(upserted_count, purged_count)``. When the upstream source is
        unavailable or returns ``None``, performs no mutations and returns
        ``(0, 0)``. When the upstream roster is empty, skips purging as a safety
        guard against transient empty-DB states.
        """
        if self._members_source is None:
            return (0, 0)
        members = await self._members_source.list_enrolled_members()
        if members is None:
            return (0, 0)

        existing = await self._gallery_repo.list_identities(active_only=False)
        existing_by_id = {ident.identity_id: ident for ident in existing}
        enrolled_ids: set[str] = set()
        upserted = 0

        now = datetime.now(UTC)
        for person_id, display_name in members:
            enrolled_ids.add(person_id)
            prev = existing_by_id.get(person_id)
            if prev is None or prev.display_name != display_name or not prev.is_active:
                await self._gallery_repo.upsert_identity(
                    Identity(
                        identity_id=person_id,
                        display_name=display_name,
                        enrolled_at=prev.enrolled_at if prev is not None else now,
                        metadata=prev.metadata if prev is not None else {},
                        is_active=True,
                    )
                )
                upserted += 1

        purged = 0
        if enrolled_ids:
            for ident in existing:
                if (
                    ident.identity_id not in enrolled_ids
                    and await self._gallery_repo.purge_identity(ident.identity_id)
                ):
                    self._reset_face_templates(ident.identity_id)
                    purged += 1

        if upserted > 0 or purged > 0:
            self._invalidate_caches()
            logger.info(
                "identity_roster_reconciled",
                upserted_count=upserted,
                purged_count=purged,
                enrolled_count=len(enrolled_ids),
            )
        return (upserted, purged)

    def _reset_face_templates(self, identity_id: str) -> None:
        if self._face_templates is not None:
            self._face_templates.reset_identity(identity_id)

    def _invalidate_caches(self) -> None:
        if self._identity_resolver is not None:
            self._identity_resolver.invalidate_identity_cache()
        if self._gallery_cache is not None:
            self._gallery_cache.invalidate()

    async def start(self) -> None:
        """Run an initial sync pass and start the periodic background loop."""
        try:
            await self.sync_once()
        except Exception:  # noqa: BLE001
            logger.warning("identity_roster_initial_sync_failed", exc_info=True)
        self._task = asyncio.create_task(self._loop(), name="identity-roster-sync")

    async def stop(self) -> None:
        """Cancel the periodic sync loop cleanly."""
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self._interval_s)
            try:
                await self.sync_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.warning("identity_roster_sync_failed", exc_info=True)

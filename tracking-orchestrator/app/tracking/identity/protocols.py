"""Dependency-inversion protocols for the identity subsystem.

Defines ``Protocol`` interfaces that the identity policy modules depend on.
No implementation is provided here — concrete implementations live in
``storage/`` (Postgres) and test helpers (in-memory). This module may only
import from ``app.domain`` and the standard library.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from ...domain import GalleryEmbedding, Identity


@runtime_checkable
class GalleryReadsProtocol(Protocol):
    """Read interface for the ReID gallery used by identity resolution."""

    async def list_identities(self, *, active_only: bool = True) -> list[Identity]:
        """Return enrolled identities, optionally filtering inactive ones."""
        ...

    async def get_embeddings_for_identity(
        self,
        identity_id: str,
        *,
        limit: int = 50,
    ) -> list[GalleryEmbedding]:
        """Return operator-verified embeddings for one identity."""
        ...


@runtime_checkable
class IdentityDecisionPersistenceProtocol(Protocol):
    """Write interface for persisting identity decisions.

    Not implemented in M01 — M04 provides the concrete Postgres adapter.
    Declared here so policy modules can depend on the interface without
    coupling to storage.
    """

    async def record_decision(
        self,
        ph_id: str,
        identity_id: str | None,
        authority: str,
        evidence_json: dict[str, object],
        decided_at: datetime,
    ) -> str:
        """Persist one identity decision and return its decision_id."""
        ...

    async def record_revision(
        self,
        revision_id: str,
        ph_id: str,
        previous_identity_id: str | None,
        new_identity_id: str | None,
        reason: str,
        actor: str,
        applied_at: datetime,
    ) -> None:
        """Persist an identity revision record."""
        ...

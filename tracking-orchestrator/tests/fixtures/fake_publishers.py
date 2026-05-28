"""Fake publishers for use in pipeline stage tests.

These are plain classes with the same async surface as the production
publishers. They record calls rather than touching Redis.
"""

from __future__ import annotations

from app.domain import IdentityRevision


class FakeRevisionPublisher:
    """Records published revisions without touching Redis."""

    def __init__(self) -> None:
        self.published: list[IdentityRevision] = []

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def publish(self, revision: IdentityRevision) -> str:
        self.published.append(revision)
        return f"fake-{len(self.published)}"

    async def publish_many(self, revisions: list[IdentityRevision]) -> list[str]:
        return [await self.publish(r) for r in revisions]

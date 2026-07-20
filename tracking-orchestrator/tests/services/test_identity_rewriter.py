"""Tests for IdentityRewriter (identity-continuity M04 backfill_null_rows).

InMemory-only; the Postgres NULL-only invariant and window-bounds proofs live
in tests/integration/test_identity_rewriter_backfill_postgres.py (real DB
required for FK-backed person_trajectories/room_dwells rows).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.services.identity_rewriter import InMemoryIdentityRewriter

T0 = datetime(2026, 7, 20, 8, 0, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_inmemory_backfill_null_rows_is_a_noop() -> None:
    """InMemoryIdentityRewriter is a zero-I/O stub, mirroring rewrite()'s peer."""
    rewriter = InMemoryIdentityRewriter()

    # Must not raise; there is no state to assert against.
    await rewriter.backfill_null_rows("rev-1", "ph-1", "alice", T0, T0 + timedelta(hours=1))

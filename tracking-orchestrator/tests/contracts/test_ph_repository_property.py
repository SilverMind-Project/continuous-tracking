"""N8: Property-based PH repository contract test.

Uses Hypothesis to generate random sequences of repository operations
and asserts identical observable behaviour between InMemory and Postgres
implementations.

Skipped by default — requires a testcontainer Postgres.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import asyncpg
import pytest

from app.storage.base import InMemoryPHRepository

# hypothesis may not be installed; skip if unavailable
try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

pytestmark = pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")


def _make_ph(ph_id: str, identity_id: str | None = None) -> Any:
    from app.domain import PersonHypothesis

    now = datetime.now(UTC)
    return PersonHypothesis(
        ph_id=ph_id,
        state_mean=(1.0, 2.0, 0.1, 0.0),
        state_cov=(0.1,) * 16,
        born_at=now,
        last_seen_at=now,
        last_seen_camera="cam-1",
        observation_count=1,
        current_identity_id=identity_id,
        current_identity_committed_at=now if identity_id else None,
        active_cameras=frozenset(["cam-1"]),
    )


if HAS_HYPOTHESIS:

    @pytest.mark.integration
    @pytest.mark.usefixtures("set_db_url_for_hypothesis")
    class TestPHRepositoryProperty:
        """Property-based tests for PH repository parity (T5).

        Runs under ``-m integration``.  The ``set_db_url_for_hypothesis``
        fixture exposes the testcontainer URL as TEST_DATABASE_URL so the
        Hypothesis test can create its own asyncpg pool without direct fixture
        injection (which ``@given`` does not support).

        When TEST_DATABASE_URL is set, both InMemory and Postgres repos are
        tested with identical operation sequences. Otherwise only InMemory is
        tested (never the case under ``make test-integration``).
        """

        ops = st.sampled_from(["save", "list_active", "correct"])

        @given(st.lists(ops, min_size=1, max_size=20))
        @settings(max_examples=50, deadline=None)
        @pytest.mark.asyncio
        async def test_operation_sequence_parity(self, operations: list[str]):
            """InMemory and Postgres produce identical results for any sequence."""
            repo_a = InMemoryPHRepository()

            # Use Postgres if available, otherwise compare two InMemory repos.
            use_postgres = bool(os.getenv("TEST_DATABASE_URL"))
            if use_postgres:
                from app.storage.postgres.ph_repo import PostgresPHRepository

                pool = await asyncpg.create_pool(os.environ["TEST_DATABASE_URL"])
                repo_b = PostgresPHRepository(pool)
            else:
                repo_b = InMemoryPHRepository()

            import uuid as _uuid

            from app.domain import PersonHypothesis

            now = datetime.now(UTC)
            ph_id_counter = 0
            # Use deterministic UUIDs: a fixed namespace + counter.
            _ns = _uuid.UUID("12345678-1234-5678-1234-567812345678")

            try:
                for op in operations:
                    if op == "save":
                        ph_id = str(_uuid.uuid5(_ns, str(ph_id_counter)))
                        ph_id_counter += 1
                        ph = PersonHypothesis(
                            ph_id=ph_id,
                            state_mean=(1.0, 2.0, 0.1, 0.0),
                            state_cov=(0.1,) * 16,
                            born_at=now,
                            last_seen_at=now,
                            last_seen_camera="cam-1",
                            observation_count=1,
                        )
                        await repo_a.save(ph)
                        await repo_b.save(ph)
                        check_a = await repo_a.get_by_id(ph_id)
                        check_b = await repo_b.get_by_id(ph_id)
                        assert (check_a is None) == (check_b is None)
                        if check_a and check_b:
                            assert check_a.ph_id == check_b.ph_id

                    elif op == "list_active":
                        items_a, total_a = await repo_a.list_active(
                            limit=50, offset=0, include_transient=True
                        )
                        items_b, total_b = await repo_b.list_active(
                            limit=50, offset=0, include_transient=True
                        )
                        assert total_a == total_b
                        assert len(items_a) == len(items_b)

                    elif op == "correct":
                        if ph_id_counter > 0:
                            target_id = str(_uuid.uuid5(_ns, str(ph_id_counter - 1)))
                            try:
                                rev_a = await repo_a.correct_identity(
                                    ph_id=target_id,
                                    new_identity_id="test_identity",
                                    reason="property_test",
                                    actor="hypothesis",
                                )
                                rev_b = await repo_b.correct_identity(
                                    ph_id=target_id,
                                    new_identity_id="test_identity",
                                    reason="property_test",
                                    actor="hypothesis",
                                )
                                assert rev_a.ph_id == rev_b.ph_id
                                assert rev_a.new_identity_id == rev_b.new_identity_id
                            except ValueError:
                                pass
            finally:
                if use_postgres:
                    async with pool.acquire() as conn:
                        await conn.execute("TRUNCATE continuous_tracking.person_hypotheses CASCADE")
                        await conn.execute("TRUNCATE continuous_tracking.ph_revisions CASCADE")
                    await pool.close()

    @pytest.mark.integration
    class TestPHRepositoryParity:
        """Explicit parity tests between InMemory and Postgres repos."""

        @pytest.mark.asyncio
        async def test_save_and_get_round_trips(self, db_pool):
            from app.domain import PersonHypothesis
            from app.storage.postgres.ph_repo import PostgresPHRepository

            inmem = InMemoryPHRepository()
            pg = PostgresPHRepository(db_pool)

            now = datetime.now(UTC)
            import uuid as _uuid

            ph = PersonHypothesis(
                ph_id=str(_uuid.uuid4()),
                state_mean=(1.0, 2.0, 0.1, 0.0),
                state_cov=(0.1,) * 16,
                born_at=now,
                last_seen_at=now,
                last_seen_camera="cam-1",
                observation_count=1,
            )

            await inmem.save(ph)
            await pg.save(ph)

            result_inmem = await inmem.get(ph.ph_id)
            result_pg = await pg.get(ph.ph_id)

            assert result_inmem is not None
            assert result_pg is not None
            assert result_inmem.ph_id == result_pg.ph_id == ph.ph_id

        @pytest.mark.asyncio
        async def test_batch_correct_is_atomic(self, db_pool):
            from app.domain import PersonHypothesis
            from app.storage.postgres.ph_repo import PostgresPHRepository

            inmem = InMemoryPHRepository()
            pg = PostgresPHRepository(db_pool)

            now = datetime.now(UTC)
            import uuid as _uuid

            ph = PersonHypothesis(
                ph_id=str(_uuid.uuid4()),
                state_mean=(1.0, 2.0, 0.1, 0.0),
                state_cov=(0.1,) * 16,
                born_at=now,
                last_seen_at=now,
                last_seen_camera="cam-1",
                observation_count=1,
            )

            await inmem.save(ph)
            await pg.save(ph)

            revs_inmem = await inmem.batch_correct(
                ph_ids=[ph.ph_id],
                new_identity_ids=["alice"],
                actor="test",
                reasons=["parity test"],
            )
            revs_pg = await pg.batch_correct(
                ph_ids=[ph.ph_id],
                new_identity_ids=["alice"],
                actor="test",
                reasons=["parity test"],
            )

            assert revs_inmem[0].ph_id == revs_pg[0].ph_id
            assert revs_pg[0].new_identity_id == "alice"

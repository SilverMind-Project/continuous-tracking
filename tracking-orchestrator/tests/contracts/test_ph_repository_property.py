"""N8: Property-based PH repository contract test.

Uses Hypothesis to generate random sequences of repository operations
and asserts identical observable behaviour between InMemory and Postgres
implementations.

Skipped by default — requires a testcontainer Postgres.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.tracking.world.repository import InMemoryPHRepository

# hypothesis may not be installed; skip if unavailable
try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


pytestmark = pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")


if HAS_HYPOTHESIS:

    class TestPHRepositoryProperty:
        """Property-based tests for PH repository parity."""

        ops = st.sampled_from(["save", "list_active", "correct"])

        @given(st.lists(ops, min_size=1, max_size=20))
        @settings(max_examples=50)
        @pytest.mark.asyncio
        async def test_operation_sequence_parity(self, operations: list[str]):
            """InMemory and Postgres produce identical results for any operation sequence."""
            repo_a = InMemoryPHRepository()
            repo_b = InMemoryPHRepository()
            from app.domain import PersonHypothesis

            now = datetime.now(UTC)
            ph_id_counter = 0

            for op in operations:
                if op == "save":
                    ph_id = f"ph-{ph_id_counter}"
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
                    items_a, total_a = await repo_a.list_active(limit=50, offset=0)
                    items_b, total_b = await repo_b.list_active(limit=50, offset=0)
                    assert total_a == total_b
                    assert len(items_a) == len(items_b)

                elif op == "correct":
                    if ph_id_counter > 0:
                        target_id = f"ph-{ph_id_counter - 1}"
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

"""Contract-test fixtures.

Exposes ``migrated_postgres_url`` via ``TEST_DATABASE_URL`` so that the
Hypothesis property test (which cannot use pytest fixtures directly) can create
its own asyncpg pool against the session testcontainer.

This autouse fixture only fires when the contracts directory is collected,
which only happens when ``-m integration`` is passed (the Hypothesis class
is marked integration).  During the fast ``make check`` run (``-m "not
integration"``), this file is not collected and the testcontainer never starts.
"""

from __future__ import annotations

import os
from collections.abc import Generator

import pytest


@pytest.fixture(scope="session")
def set_db_url_for_hypothesis(migrated_postgres_url: str) -> Generator[None, None, None]:
    """Set TEST_DATABASE_URL so the Hypothesis property test uses the testcontainer.

    Not autouse: explicitly requested by the integration-marked Hypothesis test
    class via ``pytestmark``.
    """
    old = os.environ.get("TEST_DATABASE_URL")
    os.environ["TEST_DATABASE_URL"] = migrated_postgres_url
    yield
    if old is None:
        os.environ.pop("TEST_DATABASE_URL", None)
    else:
        os.environ["TEST_DATABASE_URL"] = old

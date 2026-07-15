"""Shared fixture and expected-set definitions for the M03 gallery state-parity matrix.

Both the InMemory suite (``tests/storage/test_gallery_state_parity.py``, fast gate) and the
Postgres suite (``tests/integration/test_gallery_state_parity_postgres.py``, ``make ci``) seed the
identical scenario from here and assert against the identical expected entry-ID sets, so a
state-filter bug in either implementation cannot hide behind a divergent fixture.

IDs are real UUIDs and embeddings are 768-dim because the Postgres half must satisfy the
``reid_gallery`` schema (``id``/``origin_tracklet_id`` are ``UUID``, ``embedding`` is
``vector(768)``) -- see ``migrations/0001_init.up.sql``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from app.domain import GalleryEmbedding, Identity

T0 = datetime(2026, 7, 15, 9, 0, 0, tzinfo=UTC)

ALICE = "alice"
BOB = "bob"

# Every entry carries a non-empty identity_id so the fixture never touches the
# pre-existing (not-fixed-here) Postgres-only `identity_id != ''` filter in
# _SQL_SEARCH_SIMILAR that InMemory's search_similar lacks.
_STATES = ("pending_review", "operator_verified", "rejected")
_DIM = 768
_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "m03-gallery-state-parity")


def _stable_uuid(name: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, name))


def make_identities() -> list[Identity]:
    return [
        Identity(identity_id=ALICE, display_name="Alice", enrolled_at=T0, is_active=True),
        Identity(identity_id=BOB, display_name="Bob", enrolled_at=T0, is_active=True),
    ]


def make_entries() -> list[GalleryEmbedding]:
    """Three lifecycle states x two identities = six entries, one tracklet each."""
    entries: list[GalleryEmbedding] = []
    for person_idx, person in enumerate((ALICE, BOB)):
        vec = [0.0] * _DIM
        vec[person_idx] = 1.0
        for state_idx, state in enumerate(_STATES):
            entries.append(
                GalleryEmbedding(
                    gallery_entry_id=_stable_uuid(f"{person}-{state}-entry"),
                    identity_id=person,
                    embedding=vec,
                    seen_at=T0,
                    quality=0.9,
                    origin_tracklet_id=_stable_uuid(f"{person}-{state_idx}-tracklet"),
                    face_confirmed=True,
                    state=state,
                )
            )
    return entries


def entry_for(person: str, state: str) -> GalleryEmbedding:
    return next(e for e in make_entries() if e.identity_id == person and e.state == state)


ALL_TRACKLET_IDS = {e.origin_tracklet_id for e in make_entries()}


def expected_ids(*, states: frozenset[str] | None, identity_id: str | None = None) -> set[str]:
    """Entry IDs the fixture defines for a given state filter (and optional identity)."""
    ids = set()
    for entry in make_entries():
        if identity_id is not None and entry.identity_id != identity_id:
            continue
        if states is not None and entry.state not in states:
            continue
        ids.add(entry.gallery_entry_id)
    return ids


# The four `states`/`allowed_states` values the milestone requires the matrix to cover.
STATE_FILTER_CASES: tuple[frozenset[str] | None, ...] = (
    frozenset({"operator_verified"}),  # the default, passed explicitly for clarity
    None,
    frozenset({"pending_review"}),
    frozenset({"pending_review", "operator_verified"}),
)


def query_embedding() -> list[float]:
    """A query vector aligned with alice's basis direction (index 0)."""
    vec = [0.0] * _DIM
    vec[0] = 1.0
    return vec

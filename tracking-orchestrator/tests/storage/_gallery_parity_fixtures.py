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
from datetime import UTC, datetime, timedelta

from app.domain import GalleryEmbedding, Identity

T0 = datetime(2026, 7, 15, 9, 0, 0, tzinfo=UTC)

ALICE = "alice"
BOB = "bob"
CAROL_INACTIVE = "carol"

# Every entry carries a non-empty identity_id so the fixture never touches the
# Postgres-only `identity_id != ''` filter in _SQL_SEARCH_SIMILAR.
_STATES = ("pending_review", "auto_verified", "operator_verified", "rejected")
_ALL_COUNT_STATES = (*_STATES, "expired")
_DIM = 768
_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "m03-gallery-state-parity")


def _stable_uuid(name: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, name))


def make_identities() -> list[Identity]:
    return [
        Identity(identity_id=ALICE, display_name="Alice", enrolled_at=T0, is_active=True),
        Identity(identity_id=BOB, display_name="Bob", enrolled_at=T0, is_active=True),
        Identity(
            identity_id=CAROL_INACTIVE,
            display_name="Carol",
            enrolled_at=T0,
            is_active=False,
        ),
    ]


def make_entries() -> list[GalleryEmbedding]:
    """Four lifecycle states x two active identities + one inactive identity entry."""
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
    carol_vec = [0.0] * _DIM
    carol_vec[0] = 0.8
    carol_vec[2] = 0.6
    entries.append(
        GalleryEmbedding(
            gallery_entry_id=_stable_uuid(f"{CAROL_INACTIVE}-operator_verified-entry"),
            identity_id=CAROL_INACTIVE,
            embedding=carol_vec,
            seen_at=T0,
            quality=0.9,
            origin_tracklet_id=_stable_uuid(f"{CAROL_INACTIVE}-0-tracklet"),
            face_confirmed=True,
            state="operator_verified",
        )
    )
    return entries


def entry_for(person: str, state: str) -> GalleryEmbedding:
    return next(e for e in make_entries() if e.identity_id == person and e.state == state)


ALL_TRACKLET_IDS = {e.origin_tracklet_id for e in make_entries()}
_ACTIVE_IDENTITIES = {i.identity_id for i in make_identities() if i.is_active}


def expected_ids(
    *,
    states: frozenset[str] | None,
    identity_id: str | None = None,
    active_only: bool = True,
) -> set[str]:
    """Entry IDs the fixture defines for a given state filter, identity, and active flag."""
    ids = set()
    for entry in make_entries():
        if active_only and entry.identity_id not in _ACTIVE_IDENTITIES:
            continue
        if identity_id is not None and entry.identity_id != identity_id:
            continue
        if states is not None and entry.state not in states:
            continue
        ids.add(entry.gallery_entry_id)
    return ids


# The `states`/`allowed_states` values the milestone requires the matrix to cover.
# The addition of auto_verified as a fourth lifecycle state and VOTING_STATES
# vote-path default) as a fifth case.
STATE_FILTER_CASES: tuple[frozenset[str] | None, ...] = (
    frozenset({"operator_verified"}),  # the old VERIFIED_ONLY default
    frozenset({"operator_verified", "auto_verified"}),  # VOTING_STATES, the default
    None,
    frozenset({"pending_review"}),
    frozenset({"pending_review", "operator_verified"}),
    frozenset({"auto_verified"}),
)


def query_embedding() -> list[float]:
    """A query vector aligned with alice's basis direction (index 0)."""
    vec = [0.0] * _DIM
    vec[0] = 1.0
    return vec


def vote_window_entries(
    as_of: datetime,
) -> tuple[
    GalleryEmbedding,
    GalleryEmbedding,
    GalleryEmbedding,
    GalleryEmbedding,
    GalleryEmbedding,
]:
    """Five operator_verified entries bracketing ``(as_of - 12h, as_of]``.

    Returns ``(stale_13h, at_lower_12h, inside_11h59m59s, at_upper_0s, future_plus_1s)``.
    Strict lower bound (``seen_at > seen_after``) excludes ``at_lower_12h``; inclusive
    upper bound (``seen_at <= seen_before``) includes ``at_upper_0s`` and excludes
    ``future_plus_1s``.
    """
    vec = query_embedding()

    def _entry(name: str, delta: timedelta) -> GalleryEmbedding:
        return GalleryEmbedding(
            gallery_entry_id=_stable_uuid(f"vote-window-{name}-entry"),
            identity_id=ALICE,
            embedding=vec,
            seen_at=as_of + delta,
            quality=0.9,
            origin_tracklet_id=_stable_uuid(f"vote-window-{name}-tracklet"),
            face_confirmed=True,
            state="operator_verified",
        )

    return (
        _entry("stale-13h", -timedelta(hours=13)),
        _entry("at-lower-12h", -timedelta(hours=12)),
        _entry("inside-11h59m59s", -timedelta(hours=11, minutes=59, seconds=59)),
        _entry("at-upper-0s", timedelta(0)),
        _entry("future-plus-1s", timedelta(seconds=1)),
    )


def expected_state_counts() -> dict[str, int]:
    """``count_review_queue`` over the seeded entries: every row counts, whoever wrote it."""
    counts = dict.fromkeys(_ALL_COUNT_STATES, 0)
    for entry in make_entries():
        counts[entry.state] += 1
    return counts

"""Shared scenario for the gallery provenance-filter parity matrix.

Both the InMemory suite (``tests/storage/test_gallery_seed_filters_parity.py``, fast gate) and the
Postgres suite (``tests/integration/test_gallery_seed_filters_postgres.py``, ``make ci``) create
these rows through ``create_review_candidate`` (the pipeline's only write path) and assert the
same expected id sets for ``exclude_seed_sources`` and ``seen_before``, on ``search_similar`` and
``list_gallery_entries_for_tracklets``. Ids are UUIDs and embeddings 768-dim because Postgres
stores them in ``UUID`` and ``vector(768)`` columns.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from app.domain import GallerySeedSource, Identity, NewReviewCandidate

T0 = datetime(2026, 6, 1, 9, 0, 0, tzinfo=UTC)
MEMBER = "seed-member"
SEGMENT = "33333333-3333-3333-3333-333333333333"
_DIM = 768
_NS = uuid.uuid5(uuid.NAMESPACE_DNS, "gallery-seed-filter-parity")

# (name, seed_source, minutes after T0). All rows vote (auto_verified).
_ROWS: tuple[tuple[str, GallerySeedSource, int], ...] = (
    ("face-early", GallerySeedSource.FACE, 0),
    ("wear-early", GallerySeedSource.WEARABLE_ATTESTED, 5),
    ("room-mid", GallerySeedSource.WEARABLE_ROOM_ATTESTED, 10),
    ("face-late", GallerySeedSource.FACE, 20),
    ("wear-late", GallerySeedSource.WEARABLE_ATTESTED, 30),
)

CUTOFF = T0 + timedelta(minutes=10)

# (exclude_seed_sources, seen_before, expected row names); seen_before is inclusive (<=).
FILTER_CASES: tuple[tuple[frozenset[str], datetime | None, frozenset[str]], ...] = (
    (frozenset(), None, frozenset(name for name, _, _ in _ROWS)),
    (
        frozenset({"wearable_attested", "wearable_room_attested"}),
        None,
        frozenset({"face-early", "face-late"}),
    ),
    (frozenset(), CUTOFF, frozenset({"face-early", "wear-early", "room-mid"})),
    (frozenset({"wearable_attested"}), CUTOFF, frozenset({"face-early", "room-mid"})),
)


def row_id(name: str) -> str:
    return str(uuid.uuid5(_NS, name))


def tracklet_id(name: str) -> str:
    return str(uuid.uuid5(_NS, f"{name}-tracklet"))


def identity() -> Identity:
    return Identity(identity_id=MEMBER, display_name="Seed Member", enrolled_at=T0)


def query_embedding() -> list[float]:
    vec = [0.0] * _DIM
    vec[0] = 1.0
    return vec


def candidates() -> list[NewReviewCandidate]:
    out: list[NewReviewCandidate] = []
    for i, (name, source, minutes) in enumerate(_ROWS):
        vec = [0.0] * _DIM
        vec[0] = 1.0
        vec[i + 1] = 0.1
        at = T0 + timedelta(minutes=minutes)
        out.append(
            NewReviewCandidate(
                candidate_id=row_id(name),
                identity_id=MEMBER,
                embedding=vec,
                quality=0.9,
                orientation=2,
                camera_id="cam-1",
                capture_time=at,
                ph_id=str(uuid.uuid5(_NS, "ph")),
                observation_id=tracklet_id(name),
                origin_tracklet_id=tracklet_id(name),
                keyframe_id=None,
                crop_key=f"reid/{name}.jpg",
                source_frame_key=None,
                crop_hash=None,
                frame_hash=None,
                dimensions=(64, 128),
                is_truncated=False,
                is_occluded=False,
                candidate_reason="face_derived" if source == GallerySeedSource.FACE else source,
                source_episode_id=str(uuid.uuid5(_NS, f"{name}-episode")),
                created_actor="pipeline",
                model_version="reid",
                preprocessing_version="v1",
                confidence=None if source != GallerySeedSource.FACE else 0.95,
                state="auto_verified",
                seed_source=source,
                seed_association_id=None if source == GallerySeedSource.FACE else SEGMENT,
            )
        )
    return out


def names_for(ids: set[str]) -> frozenset[str]:
    by_id = {row_id(name): name for name, _, _ in _ROWS}
    return frozenset(by_id[i] for i in ids)


def all_tracklets() -> set[str]:
    return {tracklet_id(name) for name, _, _ in _ROWS}


def seed_source_of(name: str) -> GallerySeedSource:
    return next(source for row, source, _ in _ROWS if row == name)

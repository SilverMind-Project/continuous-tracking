"""M00 characterization for duplicate active household identities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from app.domain import GlobalTrack, Identity
from app.storage.base import InMemoryGalleryRepository
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig

_FIXTURE = Path(__file__).parents[1] / "fixtures/identity_integrity/duplicate_active_identity.json"


@pytest.mark.xfail(
    strict=True,
    reason="M02 removes this xfail when duplicate-active identity enforcement is authoritative",
)
async def test_active_identity_is_held_by_at_most_one_ph() -> None:
    data = json.loads(_FIXTURE.read_text())
    captured_at = datetime.fromisoformat(data["captured_at"])
    hypotheses = [
        GlobalTrack(
            global_track_id=item["ph_id"],
            camera_ids=[f"camera-synthetic-{index}"],
            tracklet_ids=[f"observation-synthetic-{index}"],
            started_at=captured_at,
            last_seen_at=captured_at,
            current_identity_id=item["previous_identity_id"],
            current_identity_committed_at=captured_at,
        )
        for index, item in enumerate(data["contenders"], start=1)
    ]
    resolver = IdentityResolver(
        gallery_repo=InMemoryGalleryRepository(),
        identities=[
            Identity(
                identity_id=data["identity_id"],
                display_name="Synthetic Resident Alpha",
                enrolled_at=captured_at,
            )
        ],
        config=ResolverConfig(enable_duplicate_active_identity_guard=False),
    )

    outcome = await resolver.resolve(
        hypotheses=hypotheses,
        new_face_anchors=[],
        captured_at=captured_at,
        open_ph_identities={ph.entity_id: data["identity_id"] for ph in hypotheses},
    )

    active_holders = [
        decision for decision in outcome.decisions if decision.identity_id == data["identity_id"]
    ]
    assert len(active_holders) <= 1

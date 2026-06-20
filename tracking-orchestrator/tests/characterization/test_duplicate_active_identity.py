"""M02 characterization: at most one active PH holds a household identity."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from app.domain import GlobalTrack, Identity
from app.storage.base import InMemoryGalleryRepository
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig

_FIXTURE = Path(__file__).parents[1] / "fixtures/identity_integrity/duplicate_active_identity.json"


async def test_active_identity_is_held_by_at_most_one_ph() -> None:
    """The duplicate-active guard ensures at most one open PH holds an identity.

    Both contenders start with the same prior identity and no direct face or
    ReID evidence.  With enable_duplicate_active_identity_guard=True (the M02
    default), the guard must clear all contenders when the evidence is tied.
    """
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
        config=ResolverConfig(enable_duplicate_active_identity_guard=True),
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
    assert len(active_holders) <= 1, (
        f"Expected at most one PH to hold {data['identity_id']!r}, "
        f"got {len(active_holders)}: {[d.ph_id for d in active_holders]}"
    )


async def test_tied_contenders_are_all_cleared() -> None:
    """When contenders are evidence-tied, all are cleared to Unknown."""
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
            # No last_independent_identity_evidence_at → no evidence clock set
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
        config=ResolverConfig(enable_duplicate_active_identity_guard=True),
    )

    outcome = await resolver.resolve(
        hypotheses=hypotheses,
        new_face_anchors=[],
        captured_at=captured_at,
        open_ph_identities={ph.entity_id: data["identity_id"] for ph in hypotheses},
    )

    # Both PHs have no evidence clock → no maintenance window → prior doesn't
    # fire → decisions are Unknown.  Zero holders is the correct outcome.
    active_holders = [d for d in outcome.decisions if d.identity_id == data["identity_id"]]
    assert len(active_holders) == 0

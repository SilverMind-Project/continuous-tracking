"""M00 characterization for prior-only identity timestamp renewal."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from app.domain import Identity, PersonHypothesis
from app.storage.base import (
    InMemoryGalleryRepository,
    InMemoryPHRepository,
    InMemoryWorldObservationRepository,
)
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.tracker import _resolve_identities

_FIXTURE = (
    Path(__file__).parents[1] / "fixtures/identity_integrity/prior_only_timestamp_renewal.json"
)


@pytest.mark.xfail(
    strict=True,
    reason="M02 replaces this xfail with a positive independent-evidence clock assertion",
)
async def test_prior_only_maintenance_does_not_refresh_evidence_time() -> None:
    data = json.loads(_FIXTURE.read_text())
    committed_at = datetime.fromisoformat(data["initial_independent_evidence_at"])
    evaluated_at = datetime.fromisoformat(data["resolver_evaluated_at"])
    ph = PersonHypothesis(
        ph_id=data["ph_id"],
        state_mean=(0.0, 0.0, 0.0, 0.0),
        state_cov=(1.0,) * 16,
        born_at=committed_at,
        last_seen_at=evaluated_at,
        last_seen_camera="camera-synthetic-a",
        observation_count=3,
        current_identity_id=data["identity_id"],
        current_identity_committed_at=committed_at,
        active_cameras=frozenset({"camera-synthetic-a"}),
        mean_quality=0.9,
    )
    ph_repo = InMemoryPHRepository()
    await ph_repo.save(ph)
    resolver = IdentityResolver(
        gallery_repo=InMemoryGalleryRepository(),
        identities=[
            Identity(
                identity_id=data["identity_id"],
                display_name="Synthetic Resident Alpha",
                enrolled_at=committed_at,
            )
        ],
        config=ResolverConfig(prior_maintenance_max_age_s=120.0),
    )

    await _resolve_identities(
        resolver=resolver,
        obs_repo=InMemoryWorldObservationRepository(),
        ph_repo=ph_repo,
        phs=[ph],
        ph_obs_meta={ph.ph_id: (1, None, 0.9)},
        face_anchors=[],
        now=evaluated_at,
        config=WorldTrackerConfig(),
    )

    stored = await ph_repo.get(ph.ph_id)
    assert stored is not None
    assert stored.current_identity_committed_at == committed_at

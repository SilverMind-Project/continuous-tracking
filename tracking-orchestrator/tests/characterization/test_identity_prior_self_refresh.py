"""M02 characterization: prior-only maintenance does not advance the evidence clock."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

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


async def test_prior_only_maintenance_does_not_refresh_evidence_time() -> None:
    """Prior-only maintenance must not advance last_independent_identity_evidence_at.

    The PH starts with an evidence-backed identity committed at T0.
    The resolver is called at T0+5s with no face or ReID evidence.
    Afterward, last_independent_identity_evidence_at must still equal T0.
    """
    data = json.loads(_FIXTURE.read_text())
    evidence_at = datetime.fromisoformat(data["initial_independent_evidence_at"])
    evaluated_at = datetime.fromisoformat(data["resolver_evaluated_at"])
    ph = PersonHypothesis(
        ph_id=data["ph_id"],
        state_mean=(0.0, 0.0, 0.0, 0.0),
        state_cov=(1.0,) * 16,
        born_at=evidence_at,
        last_seen_at=evaluated_at,
        last_seen_camera="camera-synthetic-a",
        observation_count=3,
        current_identity_id=data["identity_id"],
        current_identity_committed_at=evidence_at,
        last_independent_identity_evidence_at=evidence_at,
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
                enrolled_at=evidence_at,
            )
        ],
        config=ResolverConfig(prior_maintenance_max_age_s=30.0),
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
    # Identity is still held within the 30 s window.
    assert stored.current_identity_id == data["identity_id"]
    # The evidence clock must NOT have advanced — prior-only maintenance does
    # not constitute independent evidence.
    assert stored.last_independent_identity_evidence_at == evidence_at


async def test_prior_expires_after_30_seconds_not_120() -> None:
    """Identity prior expires at 30 s, not the old 120 s default."""
    data = json.loads(_FIXTURE.read_text())
    evidence_at = datetime.fromisoformat(data["initial_independent_evidence_at"])
    # Evaluate 31 s after the last independent evidence — outside the 30 s window.
    from datetime import timedelta

    evaluated_at = evidence_at + timedelta(seconds=31)
    ph = PersonHypothesis(
        ph_id=data["ph_id"],
        state_mean=(0.0, 0.0, 0.0, 0.0),
        state_cov=(1.0,) * 16,
        born_at=evidence_at,
        last_seen_at=evaluated_at,
        last_seen_camera="camera-synthetic-a",
        observation_count=3,
        current_identity_id=data["identity_id"],
        current_identity_committed_at=evidence_at,
        last_independent_identity_evidence_at=evidence_at,
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
                enrolled_at=evidence_at,
            )
        ],
        config=ResolverConfig(
            prior_maintenance_max_age_s=30.0,
            enable_sticky_maintenance=False,
        ),
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
    # The prior has expired; identity must be cleared to Unknown.
    assert stored.current_identity_id is None
    # Evidence clock is unchanged after a clear.
    assert stored.last_independent_identity_evidence_at == evidence_at

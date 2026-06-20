"""Tests for the M06 IdentityCorrectionService.

Covers proposal boundaries, frame-only vs bounded application, stale-version
conflict, explicit Unknown vs empty rejection, live-edge identity update,
historical corrections, handoff-vs-ordinary split behavior, the inferred-cannot-
supersede-operator guard, idempotent acks, and compensating revisions. All
InMemory; no database or Redis.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.domain import (
    BoundingBox,
    FloorPoint,
    PersonHypothesis,
    ProjectionAck,
    WorldObservation,
)
from app.services.identity_correction_service import (
    CorrectionConfig,
    CorrectionConflictError,
    EmptyIdentityError,
    IdentityCorrectionService,
    StaleVersionError,
)
from app.storage.base import InMemoryPHRepository
from app.storage.corrections import InMemoryIdentityCorrectionRepository

T0 = datetime(2026, 6, 20, 12, 0, 0, tzinfo=UTC)


def _ph(ph_id: str, *, last_seen_at: datetime, closed: bool = False) -> PersonHypothesis:
    return PersonHypothesis(
        ph_id=ph_id,
        state_mean=(1.0, 2.0, 0.0, 0.0),
        state_cov=(0.1,) * 16,
        born_at=T0,
        last_seen_at=last_seen_at,
        last_seen_camera="cam-1",
        observation_count=0,  # version is observation_count + #corrections
        current_identity_id=None,
        active_cameras=frozenset({"cam-1"}),
        closed_at=last_seen_at if closed else None,
    )


def _obs(obs_id: str, at: datetime) -> WorldObservation:
    o = WorldObservation(
        camera_id="cam-1",
        frame_index=0,
        captured_at=at,
        floor_point=FloorPoint(1000, 2000, calibrated=True),
        bbox=BoundingBox(10, 20, 30, 40),
        embedding=[0.0] * 4,
        detection_confidence=0.9,
        observation_id=obs_id,
    )
    return o


async def _seed(
    *,
    last_seen_offset_s: float = 5.0,
    gaps: list[float] | None = None,
    closed: bool = False,
) -> tuple[
    IdentityCorrectionService,
    InMemoryPHRepository,
    InMemoryIdentityCorrectionRepository,
    list[WorldObservation],
]:
    """Build a service with a PH and a chain of observations.

    ``gaps`` lists inter-observation gaps in seconds (len = N-1 for N obs).
    """
    ph_repo = InMemoryPHRepository()
    corr_repo = InMemoryIdentityCorrectionRepository()
    gaps = gaps if gaps is not None else [1.0, 1.0, 1.0, 1.0]
    times = [T0]
    for g in gaps:
        times.append(times[-1] + timedelta(seconds=g))
    obs = [_obs(f"obs-{i}", t) for i, t in enumerate(times)]
    last_seen = times[-1] + timedelta(seconds=last_seen_offset_s)
    ph = _ph("ph-1", last_seen_at=last_seen, closed=closed)
    await ph_repo.save(ph)
    ph_repo._observations["ph-1"] = obs
    svc = IdentityCorrectionService(
        ph_repo=ph_repo,
        correction_repo=corr_repo,
        config=CorrectionConfig(prior_window_s=30.0, discontinuity_gap_s=10.0),
    )
    return svc, ph_repo, corr_repo, obs


# -- proposal ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_proposal_stops_at_discontinuity() -> None:
    # Gap of 30s between obs-2 and obs-3 is a discontinuity.
    svc, _ph_repo, _corr, _obs = await _seed(gaps=[1.0, 1.0, 30.0, 1.0])
    proposal = await svc.propose_segment("ph-1", observation_id="obs-1")
    assert proposal.start.observation_id == "obs-0"
    assert proposal.start.reason == "segment_edge"
    assert proposal.end.observation_id == "obs-2"
    assert proposal.end.reason == "association_discontinuity"
    assert proposal.observation_ids == ["obs-0", "obs-1", "obs-2"]


@pytest.mark.asyncio
async def test_proposal_full_track_when_contiguous() -> None:
    svc, _ph_repo, _corr, obs = await _seed(gaps=[1.0, 1.0, 1.0, 1.0])
    proposal = await svc.propose_segment("ph-1", observation_id="obs-2")
    assert proposal.observation_ids == [o.observation_id for o in obs]
    assert proposal.start.reason == "segment_edge"
    assert proposal.end.reason == "segment_edge"


# -- apply: frame-only vs bounded -------------------------------------------


@pytest.mark.asyncio
async def test_frame_only_correction_single_timestamp() -> None:
    svc, _ph_repo, corr_repo, obs = await _seed()
    version = (await svc.propose_segment("ph-1", observation_id="obs-1")).ph_version
    result = await svc.apply_correction(
        ph_id="ph-1",
        actor="user:carol",
        reason_code="bad_bbox",
        observation_start=obs[1].captured_at,
        observation_end=obs[3].captured_at,
        base_ph_version=version,
        target_identity_id="alice",
        frame_only=True,
    )
    correction = await corr_repo.get_correction(result.correction_id)
    assert correction is not None
    assert correction.frame_only is True
    # frame_only collapses the range to a single instant.
    assert correction.observation_start == correction.observation_end == obs[1].captured_at


@pytest.mark.asyncio
async def test_bounded_correction_writes_operator_range() -> None:
    svc, _ph_repo, corr_repo, obs = await _seed()
    version = (await svc.propose_segment("ph-1", observation_id="obs-1")).ph_version
    result = await svc.apply_correction(
        ph_id="ph-1",
        actor="user:carol",
        reason_code="wrong_person",
        observation_start=obs[0].captured_at,
        observation_end=obs[2].captured_at,
        base_ph_version=version,
        target_identity_id="alice",
    )
    ident, authority = await corr_repo.effective_identity("ph-1", obs[1].captured_at)
    assert ident == "alice"
    assert authority == "operator"
    assert result.range_id


# -- stale version -----------------------------------------------------------


@pytest.mark.asyncio
async def test_stale_version_rejected_without_writes() -> None:
    svc, _ph_repo, corr_repo, obs = await _seed()
    with pytest.raises(StaleVersionError):
        await svc.apply_correction(
            ph_id="ph-1",
            actor="user:carol",
            reason_code="wrong_person",
            observation_start=obs[0].captured_at,
            observation_end=obs[2].captured_at,
            base_ph_version=999,
            target_identity_id="alice",
        )
    assert await corr_repo.list_corrections("ph-1") == []
    assert await corr_repo.list_ranges("ph-1") == []


# -- explicit unknown vs empty ----------------------------------------------


@pytest.mark.asyncio
async def test_explicit_unknown_allowed() -> None:
    svc, _ph_repo, corr_repo, obs = await _seed()
    version = (await svc.propose_segment("ph-1", observation_id="obs-1")).ph_version
    result = await svc.apply_correction(
        ph_id="ph-1",
        actor="user:carol",
        reason_code="identity_uncertain",
        observation_start=obs[0].captured_at,
        observation_end=obs[2].captured_at,
        base_ph_version=version,
        set_unknown=True,
    )
    assert result.new_identity_id is None
    ident, authority = await corr_repo.effective_identity("ph-1", obs[1].captured_at)
    assert ident is None
    assert authority == "operator"


@pytest.mark.asyncio
async def test_empty_identity_rejected() -> None:
    svc, _ph_repo, _corr, obs = await _seed()
    version = (await svc.propose_segment("ph-1", observation_id="obs-1")).ph_version
    with pytest.raises(EmptyIdentityError):
        await svc.apply_correction(
            ph_id="ph-1",
            actor="user:carol",
            reason_code="wrong_person",
            observation_start=obs[0].captured_at,
            observation_end=obs[2].captured_at,
            base_ph_version=version,
            target_identity_id=None,
            set_unknown=False,
        )


# -- live edge ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_edge_updates_current_identity_and_seeds_prior() -> None:
    svc, ph_repo, _corr, obs = await _seed(last_seen_offset_s=2.0)
    version = (await svc.propose_segment("ph-1", observation_id="obs-3")).ph_version
    await svc.apply_correction(
        ph_id="ph-1",
        actor="user:carol",
        reason_code="wrong_person",
        observation_start=obs[2].captured_at,
        observation_end=obs[4].captured_at,
        base_ph_version=version,
        target_identity_id="alice",
    )
    ph = await ph_repo.get("ph-1")
    assert ph is not None
    assert ph.current_identity_id == "alice"
    # Prior clock seeded from the confirmed observation_end (not "now"/indefinite).
    assert ph.last_independent_identity_evidence_at == obs[4].captured_at


@pytest.mark.asyncio
async def test_historical_correction_does_not_touch_current_identity() -> None:
    # last_seen far ahead -> the corrected window is historical, not live-edge.
    svc, ph_repo, corr_repo, obs = await _seed(last_seen_offset_s=600.0)
    version = (await svc.propose_segment("ph-1", observation_id="obs-1")).ph_version
    result = await svc.apply_correction(
        ph_id="ph-1",
        actor="user:carol",
        reason_code="wrong_person",
        observation_start=obs[0].captured_at,
        observation_end=obs[2].captured_at,
        base_ph_version=version,
        target_identity_id="alice",
    )
    ph = await ph_repo.get("ph-1")
    assert ph is not None
    assert ph.current_identity_id is None  # current identity untouched
    # But the historical correction still completed its effective projection.
    ident, _ = await corr_repo.effective_identity("ph-1", obs[1].captured_at)
    assert ident == "alice"
    assert result.job.status in {"applying", "completed"}


# -- split behavior ----------------------------------------------------------


@pytest.mark.asyncio
async def test_ordinary_correction_does_not_split() -> None:
    svc, ph_repo, _corr, obs = await _seed()
    version = (await svc.propose_segment("ph-1", observation_id="obs-1")).ph_version
    result = await svc.apply_correction(
        ph_id="ph-1",
        actor="user:carol",
        reason_code="wrong_person",
        observation_start=obs[0].captured_at,
        observation_end=obs[2].captured_at,
        base_ph_version=version,
        target_identity_id="alice",
    )
    assert result.new_ph_id is None
    assert len(await ph_repo.list_open()) == 1


@pytest.mark.asyncio
async def test_handoff_correction_splits_ph() -> None:
    svc, ph_repo, _corr, obs = await _seed()
    version = (await svc.propose_segment("ph-1", observation_id="obs-2")).ph_version
    result = await svc.apply_correction(
        ph_id="ph-1",
        actor="user:carol",
        reason_code="track_handoff",
        observation_start=obs[2].captured_at,
        observation_end=obs[4].captured_at,
        base_ph_version=version,
        target_identity_id="alice",
        at_observation_id="obs-2",
    )
    assert result.new_ph_id is not None
    assert await ph_repo.get(result.new_ph_id) is not None
    # The split produced a distinct second PH.
    assert result.new_ph_id != "ph-1"


# -- inferred cannot supersede operator -------------------------------------


@pytest.mark.asyncio
async def test_inferred_revision_cannot_supersede_operator_range() -> None:
    svc, _ph_repo, _corr, obs = await _seed()
    version = (await svc.propose_segment("ph-1", observation_id="obs-1")).ph_version
    await svc.apply_correction(
        ph_id="ph-1",
        actor="user:carol",
        reason_code="wrong_person",
        observation_start=obs[0].captured_at,
        observation_end=obs[3].captured_at,
        base_ph_version=version,
        target_identity_id="alice",
    )
    with pytest.raises(CorrectionConflictError):
        await svc.record_inferred_range(
            ph_id="ph-1",
            revision_id="auto-rev",
            effective_identity_id="bob",
            start=obs[1].captured_at,
            end=obs[2].captured_at,
        )


@pytest.mark.asyncio
async def test_inferred_range_outside_operator_window_is_recorded() -> None:
    svc, _ph_repo, corr_repo, obs = await _seed(last_seen_offset_s=600.0)
    version = (await svc.propose_segment("ph-1", observation_id="obs-0")).ph_version
    await svc.apply_correction(
        ph_id="ph-1",
        actor="user:carol",
        reason_code="wrong_person",
        observation_start=obs[0].captured_at,
        observation_end=obs[1].captured_at,
        base_ph_version=version,
        target_identity_id="alice",
    )
    far_start = obs[4].captured_at + timedelta(seconds=120)
    rng = await svc.record_inferred_range(
        ph_id="ph-1",
        revision_id="auto-rev",
        effective_identity_id="bob",
        start=far_start,
        end=far_start + timedelta(seconds=10),
    )
    assert rng.authority == "inferred"
    ident, authority = await corr_repo.effective_identity(
        "ph-1", far_start + timedelta(seconds=5)
    )
    assert ident == "bob"
    assert authority == "inferred"


# -- idempotent ack ----------------------------------------------------------


@pytest.mark.asyncio
async def test_projection_ack_completes_job_idempotently() -> None:
    svc, _ph_repo, corr_repo, obs = await _seed(last_seen_offset_s=600.0)
    version = (await svc.propose_segment("ph-1", observation_id="obs-1")).ph_version
    result = await svc.apply_correction(
        ph_id="ph-1",
        actor="user:carol",
        reason_code="wrong_person",
        observation_start=obs[0].captured_at,
        observation_end=obs[2].captured_at,
        base_ph_version=version,
        target_identity_id="alice",
    )
    # cts_internal already acked inside apply; job awaits cc.
    job = await corr_repo.get_job(result.revision_id)
    assert job is not None and job.status == "applying"

    ack = ProjectionAck(
        revision_id=result.revision_id, consumer="cc", schema_version="1"
    )
    assert await svc.record_projection_ack(ack) is True
    # Replay is a no-op (still completed, single ack row).
    assert await svc.record_projection_ack(ack) is False
    job = await corr_repo.get_job(result.revision_id)
    assert job is not None and job.status == "completed"
    assert len(await corr_repo.list_acks(result.revision_id)) == 2  # cts_internal + cc


# -- compensation ------------------------------------------------------------


@pytest.mark.asyncio
async def test_compensation_restores_effective_and_retains_audit() -> None:
    svc, _ph_repo, corr_repo, obs = await _seed(last_seen_offset_s=600.0)
    version = (await svc.propose_segment("ph-1", observation_id="obs-1")).ph_version
    applied = await svc.apply_correction(
        ph_id="ph-1",
        actor="user:carol",
        reason_code="wrong_person",
        observation_start=obs[0].captured_at,
        observation_end=obs[2].captured_at,
        base_ph_version=version,
        target_identity_id="alice",
    )
    ident, _ = await corr_repo.effective_identity("ph-1", obs[1].captured_at)
    assert ident == "alice"

    comp = await svc.compensate(applied.correction_id, actor="user:dave")
    # Effective identity restored to pre-correction (Unknown here).
    ident_after, _ = await corr_repo.effective_identity("ph-1", obs[1].captured_at)
    assert ident_after is None
    # Original correction row is retained (audit chain intact).
    assert await corr_repo.get_correction(applied.correction_id) is not None
    comp_row = await corr_repo.get_correction(comp.correction_id)
    assert comp_row is not None
    assert comp_row.kind == "compensation"
    assert comp_row.compensates_correction_id == applied.correction_id

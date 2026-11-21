"""Wearable supervision replays: real tracker rounds feed the real learners.

Each round drives ``WorldTracker.step`` (association, attestation, resolver) and then the
post-world learners exactly as the pipeline does: ``ReIDCandidateStage`` reads the round's
``det_to_ph``, committed identities, and per-PH wearable evidence, and the image-only
evaluation service accounts the resolver's sampled evaluations. No hardware, no database.

Scenarios: a sole occupant with a moving tag mints one pending room-grade row; a second person
entering stops minting; a band left on a table mints nothing; two tags of one member that
disagree mint nothing; and the teacher loop: an hour wearing a tag teaches the gallery, the
paired evaluation shows the taught rows helping, and after the tag comes off the taught
cameras name the person where untaught cameras cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from cts_contracts import FixKind, MotionState, WearState
from prometheus_client import CollectorRegistry

from app.domain import (
    BoundingBox,
    Detection,
    FloorPoint,
    GallerySeedSource,
    Identity,
    OrientationBin,
    WearableFix,
    WorldObservation,
)
from app.observability import metrics as metrics_mod
from app.observability.metrics import build_metrics
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.reid_candidates import ReIDCandidateStage
from app.services.image_only_evaluator import ImageOnlyEvaluationService
from app.services.wearable_fix_buffer import WearableFixBuffer
from app.storage.base import (
    InMemoryGalleryRepository,
    InMemoryPHRepository,
    InMemoryWorldObservationRepository,
)
from app.storage.gallery import PENDING_AND_VERIFIED
from app.storage.image_only_evaluation import InMemoryImageOnlyEvaluationRepository
from app.tracking.identity.candidate_eligibility import CandidatePolicy
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.tracker import WorldTracker
from app.tracking.world.wearable_config import WearableConfig
from app.transport.redis_streams import FrameReady

# Fixed capture-time anchor: vote-window and recency decay use captured_at, never wall clock.
T0 = datetime(2026, 6, 1, 9, 0, 0, tzinfo=UTC)
_ROOMS = {
    "1": [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)],
    "2": [(5.0, 0.0), (10.0, 0.0), (10.0, 5.0), (5.0, 5.0)],
}
_RESIDENT = [1.0, 0.0, 0.0, 0.0]
_VISITOR = [0.0, 1.0, 0.0, 0.0]


class _CropStorage:
    async def put_bytes(self, key: str, data: bytes, content_type: str = "image/jpeg") -> None:
        return None

    async def delete_object(self, key: str) -> None:
        return None


@dataclass
class _Person:
    det_prefix: str
    embedding: list[float]
    x_m: float
    y_m: float


@dataclass
class _Stack:
    tracker: WorldTracker
    gallery: InMemoryGalleryRepository
    stage: ReIDCandidateStage
    evaluation: ImageOnlyEvaluationService
    evaluation_repo: InMemoryImageOnlyEvaluationRepository
    fixes: WearableFixBuffer = field(default_factory=WearableFixBuffer)


async def _stack(
    *,
    seeding: bool = True,
    sample_rate: float = 0.0,
    exclusion_s: float = 600.0,
    resolver: ResolverConfig | None = None,
    max_similarity: float = 1.0,
) -> _Stack:
    gallery = InMemoryGalleryRepository()
    await gallery.upsert_identity(
        Identity(identity_id="resident-1", display_name="r", enrolled_at=T0)
    )
    wearable = WearableConfig(
        mode="enabled", evaluation_sample_rate=sample_rate, evaluation_exclusion_s=exclusion_s
    )
    tracker = WorldTracker(
        ph_repo=InMemoryPHRepository(),
        obs_repo=InMemoryWorldObservationRepository(),
        config=WorldTrackerConfig(min_observations_to_publish=1),
        identity_resolver=IdentityResolver(
            gallery_repo=gallery,
            config=resolver
            or ResolverConfig(
                enable_duplicate_active_identity_guard=True,
                prior_maintenance_max_age_s=65.0,
                enable_sticky_maintenance=True,
            ),
            wearable_config=wearable,
        ),
        gallery_repo=gallery,
        wearable_config=wearable,
    )
    # The diversity bar is relaxed by default: these synthetic people wear one fixed embedding,
    # so spacing (120 s) is what limits minting here.
    policy = CandidatePolicy(
        model_version="reid",
        wearable_seeding_enabled=seeding,
        wearable_max_similarity_to_recent=max_similarity,
    )
    repo = InMemoryImageOnlyEvaluationRepository()
    return _Stack(
        tracker=tracker,
        gallery=gallery,
        stage=ReIDCandidateStage(gallery, _CropStorage(), policy),
        evaluation=ImageOnlyEvaluationService(repo, max_gap_s=30.0, flush_interval_s=60.0),
        evaluation_repo=repo,
    )


def _room_fix(tag: str, at: datetime, room: str, motion: MotionState, seq: int) -> WearableFix:
    return WearableFix(
        tag_id=tag,
        member_id="resident-1",
        measured_at=at,
        kind=FixKind.ROOM,
        room_id=room,
        room_probability=0.95,
        motion_state=motion,
        epoch_seq=seq,
    )


def _position_fix(at: datetime, x_m: float, y_m: float, seq: int) -> WearableFix:
    return WearableFix(
        tag_id="tag-traj",
        member_id="resident-1",
        measured_at=at,
        kind=FixKind.POSITION,
        x_m=x_m,
        y_m=y_m,
        cov_m2=(0.015, 0.0, 0.0, 0.015),
        wear_state=WearState.ON_BODY,
        motion_state=MotionState.MOVING,
        epoch_seq=seq,
    )


async def _round(
    stack: _Stack,
    at: datetime,
    people: list[_Person],
    *,
    camera_id: str = "cam-uncal",
    calibrated: bool = False,
    frame_index: int,
) -> dict[str, str | None]:
    """One tracker round and its learners; returns committed identity per person prefix."""
    observations = [
        WorldObservation(
            camera_id=camera_id,
            frame_index=frame_index,
            captured_at=at,
            floor_point=FloorPoint(
                x_mm=int(p.x_m * 1000), y_mm=int(p.y_m * 1000), calibrated=calibrated
            ),
            bbox=BoundingBox(x_min=100, y_min=100, x_max=220, y_max=400),
            embedding=p.embedding,
            detection_confidence=0.92,
            detection_id=f"{p.det_prefix}-{frame_index}",
            quality=0.65,
            orientation=OrientationBin.BACK,
            orientation_confidence=0.9,
            floor_cov_random=(0.04, 0.0, 0.0, 0.04) if calibrated else None,
            footpoint_reliable=True,
        )
        for p in people
    ]
    result = await stack.tracker.step(
        observations=observations,
        now=at,
        room_polygons=_ROOMS,
        camera_room_map={camera_id: "1"},
        wearable_fixes=stack.fixes.recent(now=at, max_age_s=15.0),
    )
    ctx = FrameContext(
        frame=FrameReady(
            camera_id=camera_id,
            minio_key="k",
            width=640,
            height=480,
            frame_index=frame_index,
            capture_time_unix_ns=int(at.timestamp() * 1e9),
        ),
        event_time=at,
        capture_time=at,
    )
    ctx.effective_width, ctx.effective_height = 640, 480
    ctx.domain_detections = [
        Detection(
            detection_id=o.detection_id,
            camera_id=camera_id,
            bbox=o.bbox,
            embedding=o.embedding or [],
            capture_time=at,
            event_time=at,
            confidence=0.92,
            floor_point=o.floor_point,
            crop_quality=0.9,
        )
        for o in observations
    ]
    ctx.crops = [np.zeros((256, 128, 3), dtype=np.uint8) for _ in observations]
    ctx.orientation_by_detection = {
        o.detection_id: (OrientationBin.BACK, 0.9) for o in observations
    }
    ctx.det_to_ph = dict(result.det_to_ph)
    ctx.det_to_observation_id = dict(result.det_to_observation_id)
    ctx.committed_ids = {ph.ph_id: ph.current_identity_id for ph in result.updated_phs}
    ctx.wearable_evidence_by_ph = dict(result.wearable_evidence_by_ph)
    await stack.stage.run(ctx)
    await stack.evaluation.record_round(list(result.image_only_evaluations), captured_at=at)
    return {
        p.det_prefix: ctx.committed_ids.get(ctx.det_to_ph.get(f"{p.det_prefix}-{frame_index}", ""))
        for p in people
    }


async def _wearable_rows(stack: _Stack) -> list[object]:
    rows = await stack.gallery.list_gallery_entries(active_only=False, states=None)
    return [r for r in rows if r.seed_source != GallerySeedSource.FACE]


@pytest.fixture(autouse=True)
def _fresh_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics_mod, "metrics", build_metrics(CollectorRegistry()))


@pytest.mark.asyncio
async def test_sole_occupant_mints_one_pending_row_then_a_second_person_stops_minting() -> None:
    stack = await _stack()
    resident = _Person("res", _RESIDENT, 200.0, 200.0)
    for sec in range(0, 75):
        at = T0 + timedelta(seconds=sec)
        stack.fixes.add(_room_fix("tag-r1", at, "1", MotionState.MOVING, sec))
        await _round(stack, at, [resident], frame_index=sec)

    rows = await _wearable_rows(stack)
    assert len(rows) == 1
    (row,) = rows
    assert row.seed_source == GallerySeedSource.WEARABLE_ROOM_ATTESTED
    assert row.state == "pending_review"  # votes only after an operator approves it

    # A second person walks in: the room is shared, the room grade is gone, nothing more mints,
    # although the seed spacing would allow two more rows by the end.
    visitor = _Person("vis", _VISITOR, 203.0, 203.0)
    for sec in range(75, 330):
        at = T0 + timedelta(seconds=sec)
        stack.fixes.add(_room_fix("tag-r1", at, "1", MotionState.MOVING, sec))
        await _round(stack, at, [resident, visitor], frame_index=sec)
    assert len(await _wearable_rows(stack)) == 1


@pytest.mark.asyncio
async def test_band_left_on_a_table_mints_nothing() -> None:
    stack = await _stack()
    resident = _Person("res", _RESIDENT, 200.0, 200.0)
    for sec in range(0, 300):
        at = T0 + timedelta(seconds=sec)
        # The band lies still in the room before and after the person walks in.
        stack.fixes.add(_room_fix("tag-band", at, "1", MotionState.STILL, sec))
        people = [resident] if sec >= 30 else []
        await _round(stack, at, people, frame_index=sec)
    assert await _wearable_rows(stack) == []


@pytest.mark.asyncio
async def test_two_tags_of_one_member_that_disagree_mint_nothing() -> None:
    stack = await _stack()
    resident = _Person("res", _RESIDENT, 200.0, 200.0)
    for sec in range(0, 200):
        at = T0 + timedelta(seconds=sec)
        stack.fixes.add(_room_fix("tag-wrist", at, "1", MotionState.MOVING, sec))
        stack.fixes.add(_room_fix("tag-pendant", at, "2", MotionState.MOVING, sec))
        await _round(stack, at, [resident], frame_index=sec)
    assert await _wearable_rows(stack) == []


def _walk(t_s: float) -> tuple[float, float]:
    """A slow loop around room 1 (radius 1.2 m around (2.5, 2.5), one lap per 60 s)."""
    angle = 2.0 * np.pi * t_s / 60.0
    return 2.5 + 1.2 * float(np.cos(angle)), 2.5 + 1.2 * float(np.sin(angle))


async def _wear_then_remove(stack: _Stack, *, wear_s: int, away_s: int, back_s: int) -> int:
    """Wear a trajectory tag while walking, leave, come back without it.

    Returns the number of rounds after the return in which the resident's PH held the
    resident's identity (no one else is in the house, so a wrong name is impossible to hide).
    """
    step_s = 2
    frame = 0
    for t in range(0, wear_s, step_s):
        at = T0 + timedelta(seconds=t)
        for k in range(step_s):
            fx, fy = _walk(t + k)
            stack.fixes.add(_position_fix(at + timedelta(seconds=k), fx, fy, t + k))
        x, y = _walk(t)
        frame += 1
        await _round(
            stack,
            at,
            [_Person("worn", _RESIDENT, x, y)],
            camera_id="cam-cal",
            calibrated=True,
            frame_index=frame,
        )

    # Nobody in view while away: the cameras keep running, so the worn track closes.
    for t in range(wear_s, wear_s + away_s, 10):
        frame += 1
        await _round(
            stack,
            T0 + timedelta(seconds=t),
            [],
            camera_id="cam-cal",
            calibrated=True,
            frame_index=frame,
        )

    named = 0
    start = wear_s + away_s
    for t in range(start, start + back_s, step_s):
        at = T0 + timedelta(seconds=t)
        x, y = _walk(t)
        frame += 1
        held = await _round(
            stack,
            at,
            [_Person("back", _RESIDENT, x, y)],
            camera_id="cam-cal",
            calibrated=True,
            frame_index=frame,
        )
        assert held["back"] in (None, "resident-1")
        named += held["back"] == "resident-1"
    return named


@pytest.mark.asyncio
async def test_the_tag_teaches_the_cameras_and_the_paired_evaluation_shows_it() -> None:
    resolver = ResolverConfig(
        enable_duplicate_active_identity_guard=True,
        prior_maintenance_max_age_s=65.0,
        enable_sticky_maintenance=True,
        enable_multiview_gallery=True,
    )
    taught = await _stack(sample_rate=1.0, exclusion_s=300.0, resolver=resolver)
    untaught = await _stack(seeding=False, sample_rate=1.0, exclusion_s=300.0, resolver=resolver)

    taught_named = await _wear_then_remove(taught, wear_s=900, away_s=120, back_s=300)
    untaught_named = await _wear_then_remove(untaught, wear_s=900, away_s=120, back_s=300)

    rows = await _wearable_rows(taught)
    assert 5 <= len(rows) <= CandidatePolicy().max_per_identity_orientation
    assert {(r.seed_source, r.state) for r in rows} == {
        (GallerySeedSource.WEARABLE_ATTESTED, "auto_verified")
    }
    assert await _wearable_rows(untaught) == []

    # While the tag was worn: once taught rows were older than the exclusion window, images
    # alone named the resident in the treatment arm, never in the control arm.
    await taught.evaluation.flush()
    stored = await taught.evaluation_repo.list_totals()  # may span a UTC midnight

    def arm_sum(arm: str, outcome: str) -> float:
        return sum(getattr(t, f"{outcome}_s") for t in stored if t.arm == arm)

    # The first exclusion window (300 s) is Unknown in both arms: no taught row is old enough.
    assert arm_sum("treatment", "correct") >= 500.0
    assert arm_sum("control", "correct") == 0.0
    assert arm_sum("treatment", "unknown") == pytest.approx(300.0, abs=4.0)
    assert arm_sum("treatment", "wrong") <= arm_sum("control", "wrong") == 0.0

    # After the tag came off (150 rounds), only the taught cameras name the resident.
    assert untaught_named == 0
    assert taught_named >= 135
    counted = await taught.gallery.count_gallery_entries(
        identity_id="resident-1", orientation=int(OrientationBin.BACK), states=PENDING_AND_VERIFIED
    )
    assert counted == len(rows)


@pytest.mark.asyncio
async def test_two_day_replay_mints_fresh_rows_on_day_two_after_day_one_hits_cap() -> None:
    """Task 2 regression: filling the cap of 10 on day 1 must not starve day 2."""
    policy = CandidatePolicy(
        model_version="reid",
        max_per_identity_orientation=10,
        wearable_seeding_enabled=True,
        wearable_min_seed_interval_s=60.0,
        wearable_max_similarity_to_recent=1.0,
        gallery_vote_max_age_s=43200.0,
        wearable_channel_live=True,
    )
    stack = await _stack(seeding=True)
    stack.stage = ReIDCandidateStage(stack.gallery, _CropStorage(), policy)

    # Day 1: wear tag for 15 minutes with 60s spacing -> fills cap of 10.
    frame = 0
    for t in range(0, 900, 2):
        at = T0 + timedelta(seconds=t)
        for k in range(2):
            fx, fy = _walk(t + k)
            stack.fixes.add(_position_fix(at + timedelta(seconds=k), fx, fy, t + k))
        x, y = _walk(t)
        frame += 1
        await _round(
            stack,
            at,
            [_Person("d1", _RESIDENT, x, y)],
            camera_id="cam-cal",
            calibrated=True,
            frame_index=frame,
        )

    day1_rows = await _wearable_rows(stack)
    assert len(day1_rows) == 10

    # Day 2 (24h later, new outfit embedding): cap counts only the 12h vote window, so day 2 mints.
    day2_start = T0 + timedelta(hours=24)
    day2_emb = [0.0, 0.0, 1.0, 0.0]
    for t in range(0, 300, 2):
        at = day2_start + timedelta(seconds=t)
        for k in range(2):
            fx, fy = _walk(t + k)
            stack.fixes.add(_position_fix(at + timedelta(seconds=k), fx, fy, 10000 + t + k))
        x, y = _walk(t)
        frame += 1
        await _round(
            stack,
            at,
            [_Person("d2", day2_emb, x, y)],
            camera_id="cam-cal",
            calibrated=True,
            frame_index=frame,
        )

    total_rows = await _wearable_rows(stack)
    assert len(total_rows) > 10


@pytest.mark.asyncio
async def test_wrong_band_swap_auto_revokes_taught_rows_and_holds_until_resume() -> None:
    """Task 5 replay: a family member wears the resident's band; calibrated face conflicts
    trip swap_suspected, auto-revocation demotes every taught row, and no new seed mints
    until an operator resumes teaching."""
    from app.domain import FaceAnchor
    from app.pipeline.stages.wearable_associations import WearableAssociationPersistStage
    from app.services.wearable_revocation import WearableTeachingRevocationService
    from app.storage.wearable_associations import InMemoryWearableAssociationRepository
    from app.storage.wearable_teaching_holds import InMemoryWearableTeachingHoldRepository

    assoc_repo = InMemoryWearableAssociationRepository()
    holds_repo = InMemoryWearableTeachingHoldRepository()
    stack = await _stack(seeding=True)
    await stack.gallery.upsert_identity(
        Identity(identity_id="family-member", display_name="Family", enrolled_at=T0)
    )
    policy = CandidatePolicy(
        model_version="reid",
        wearable_seeding_enabled=True,
        wearable_min_seed_interval_s=30.0,
        wearable_max_similarity_to_recent=1.0,
        gallery_vote_max_age_s=43200.0,
        wearable_channel_live=True,
    )
    stack.stage = ReIDCandidateStage(
        stack.gallery, _CropStorage(), policy, teaching_holds=holds_repo
    )
    revocation = WearableTeachingRevocationService(
        association_repo=assoc_repo,
        teaching_holds=holds_repo,
        gallery_repo=stack.gallery,
    )
    persist_stage = WearableAssociationPersistStage(
        repo=assoc_repo,
        revocation_service=revocation,
        auto_on_swap=True,
        swap_conflict_window_s=600.0,
    )

    frame = 0
    # Phase 1: family member wears resident-1's band (no face yet) and teaches auto_verified rows.
    for t in range(0, 180, 2):
        at = T0 + timedelta(seconds=t)
        for k in range(2):
            fx, fy = _walk(t + k)
            stack.fixes.add(_position_fix(at + timedelta(seconds=k), fx, fy, t + k))
        x, y = _walk(t)
        frame += 1
        obs = WorldObservation(
            camera_id="cam-cal",
            frame_index=frame,
            captured_at=at,
            floor_point=FloorPoint(x_mm=int(x * 1000), y_mm=int(y * 1000), calibrated=True),
            bbox=BoundingBox(x_min=100, y_min=100, x_max=220, y_max=400),
            embedding=_VISITOR,
            detection_confidence=0.92,
            detection_id=f"fm-{frame}",
            quality=0.65,
            orientation=OrientationBin.BACK,
            orientation_confidence=0.9,
            floor_cov_random=(0.04, 0.0, 0.0, 0.04),
            footpoint_reliable=True,
        )
        res = await stack.tracker.step(
            observations=[obs],
            now=at,
            room_polygons=_ROOMS,
            camera_room_map={"cam-cal": "1"},
            wearable_fixes=stack.fixes.recent(now=at, max_age_s=15.0),
        )
        ctx = FrameContext(
            frame=FrameReady(
                camera_id="cam-cal",
                minio_key="k",
                width=640,
                height=480,
                frame_index=frame,
                capture_time_unix_ns=int(at.timestamp() * 1e9),
            ),
            event_time=at,
            capture_time=at,
        )
        ctx.effective_width, ctx.effective_height = 640, 480
        ctx.domain_detections = [
            Detection(
                detection_id=obs.detection_id,
                camera_id="cam-cal",
                bbox=obs.bbox,
                embedding=_VISITOR,
                capture_time=at,
                event_time=at,
                confidence=0.92,
                floor_point=obs.floor_point,
                crop_quality=0.9,
            )
        ]
        ctx.crops = [np.zeros((256, 128, 3), dtype=np.uint8)]
        ctx.orientation_by_detection = {obs.detection_id: (OrientationBin.BACK, 0.9)}
        ctx.det_to_ph = dict(res.det_to_ph)
        ctx.det_to_observation_id = dict(res.det_to_observation_id)
        ctx.committed_ids = {ph.ph_id: ph.current_identity_id for ph in res.updated_phs}
        ctx.wearable_evidence_by_ph = dict(res.wearable_evidence_by_ph)
        ctx.wearable_association_transitions = list(res.wearable_association_transitions)
        await persist_stage.run(ctx)
        await stack.stage.run(ctx)

    taught_before = await stack.gallery.list_gallery_entries(active_only=False, states=None)
    assert len(taught_before) >= 1
    assert all(r.state == "auto_verified" for r in taught_before)

    # Phase 2: 3 calibrated face conflicts identifying "family-member" trip swap_suspected.
    # Run continuously from t=180 so the PH stays alive while each conflict closes the active
    # segment and the tag re-associates before the next conflict.
    last_obs_id: str | None = None
    for conflict_idx in range(3):
        t = 180 + conflict_idx * 40
        for sub in range(0, 40, 2):
            at = T0 + timedelta(seconds=t + sub)
            for k in range(2):
                fx, fy = _walk(t + sub + k)
                stack.fixes.add(
                    _position_fix(at + timedelta(seconds=k), fx, fy, 5000 + t + sub + k)
                )
            x, y = _walk(t + sub)
            frame += 1
            anchor = (
                FaceAnchor(
                    person_id="family-member",
                    match_score=0.95,
                    quality=0.9,
                    tracklet_id=last_obs_id,
                    detection_id=f"fm-{frame}",
                    captured_at=at,
                    similarity=0.95,
                    calibrated_confidence=0.98,
                )
                if sub == 38 and last_obs_id is not None
                else None
            )
            obs = WorldObservation(
                camera_id="cam-cal",
                frame_index=frame,
                captured_at=at,
                floor_point=FloorPoint(x_mm=int(x * 1000), y_mm=int(y * 1000), calibrated=True),
                bbox=BoundingBox(x_min=100, y_min=100, x_max=220, y_max=400),
                embedding=_VISITOR,
                detection_confidence=0.92,
                detection_id=f"fm-{frame}",
                quality=0.65,
                orientation=OrientationBin.BACK,
                orientation_confidence=0.9,
                floor_cov_random=(0.04, 0.0, 0.0, 0.04),
                footpoint_reliable=True,
            )
            res = await stack.tracker.step(
                observations=[obs],
                now=at,
                room_polygons=_ROOMS,
                camera_room_map={"cam-cal": "1"},
                face_anchors=[anchor] if anchor is not None else None,
                wearable_fixes=stack.fixes.recent(now=at, max_age_s=15.0),
            )
            last_obs_id = res.det_to_observation_id.get(obs.detection_id) or last_obs_id
            ctx = FrameContext(
                frame=FrameReady(
                    camera_id="cam-cal",
                    minio_key="k",
                    width=640,
                    height=480,
                    frame_index=frame,
                    capture_time_unix_ns=int(at.timestamp() * 1e9),
                ),
                event_time=at,
                capture_time=at,
            )
            ctx.effective_width, ctx.effective_height = 640, 480
            ctx.domain_detections = [
                Detection(
                    detection_id=obs.detection_id,
                    camera_id="cam-cal",
                    bbox=obs.bbox,
                    embedding=_VISITOR,
                    capture_time=at,
                    event_time=at,
                    confidence=0.92,
                    floor_point=obs.floor_point,
                    crop_quality=0.9,
                )
            ]
            ctx.crops = [np.zeros((256, 128, 3), dtype=np.uint8)]
            ctx.orientation_by_detection = {obs.detection_id: (OrientationBin.BACK, 0.9)}
            ctx.det_to_ph = dict(res.det_to_ph)
            ctx.det_to_observation_id = dict(res.det_to_observation_id)
            ctx.committed_ids = {ph.ph_id: ph.current_identity_id for ph in res.updated_phs}
            ctx.wearable_evidence_by_ph = dict(res.wearable_evidence_by_ph)
            ctx.wearable_association_transitions = list(res.wearable_association_transitions)
            await persist_stage.run(ctx)
            await stack.stage.run(ctx)

    # Every row taught before the swap is demoted to pending_review, and the tag is held.
    assert await holds_repo.is_held("tag-traj") is True
    after_swap = await stack.gallery.list_gallery_entries(active_only=False, states=None)
    assert all(r.state == "pending_review" for r in after_swap)
    count_held = len(after_swap)

    # Phase 3: further rounds while held mint zero new rows until operator resumes teaching.
    for t in range(400, 500, 2):
        at = T0 + timedelta(seconds=t)
        for k in range(2):
            fx, fy = _walk(t + k)
            stack.fixes.add(_position_fix(at + timedelta(seconds=k), fx, fy, 9000 + t + k))
        x, y = _walk(t)
        frame += 1
        await _round(
            stack,
            at,
            [_Person("after", _RESIDENT, x, y)],
            camera_id="cam-cal",
            calibrated=True,
            frame_index=frame,
        )
    assert (
        len(await stack.gallery.list_gallery_entries(active_only=False, states=None)) == count_held
    )

    # Operator resumes teaching -> tag hold cleared.
    resumed = await revocation.resume_teaching("tag-traj", actor="admin:alice")
    assert resumed is not None
    assert await holds_repo.is_held("tag-traj") is False

"""End-to-end wearable association and resolver replay scenarios.

Drives the real WorldTracker, IdentityResolver, WearableFixSubscriber,
WearableFixBuffer, WearableAssociationTracker, and WearableAssociationPersistStage
against deterministic multi-second scenarios (including IR-M13 YAML scenarios).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml
from cts_contracts import FixKind, MotionState, WearState
from prometheus_client import CollectorRegistry

from app.domain import (
    AnchorAggregation,
    AnchorOrigin,
    BoundingBox,
    FaceAnchor,
    FloorPoint,
    Identity,
    WearableAttestationGrade,
    WearableBinding,
    WearableFix,
    WearableSiteConfig,
    WorldObservation,
)
from app.observability import metrics as metrics_mod
from app.observability.metrics import WearableShadowOutcome, build_metrics
from app.proto.continuoustracking.v1 import wearable_pb2
from app.services.wearable_fix_buffer import WearableFixBuffer
from app.services.wearable_fix_subscriber import WearableFixSubscriber
from app.services.wearable_site_config_map import WearableSiteConfigMap
from app.storage.base import (
    InMemoryGalleryRepository,
    InMemoryPHRepository,
    InMemoryWorldObservationRepository,
)
from app.storage.wearable_associations import InMemoryWearableAssociationRepository
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.tracker import WorldTracker
from app.tracking.world.wearable_config import WearableConfig

_SCENARIOS_DIR = Path(__file__).resolve().parents[3] / "rf-ingress" / "config" / "scenarios"
T0 = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)


def _unix_ns(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000_000)


def _interp_waypoint(waypoints: list[dict[str, Any]], t_s: float) -> tuple[float, float]:
    if t_s <= waypoints[0]["t_s"]:
        return float(waypoints[0]["x_m"]), float(waypoints[0]["y_m"])
    if t_s >= waypoints[-1]["t_s"]:
        return float(waypoints[-1]["x_m"]), float(waypoints[-1]["y_m"])
    for i in range(len(waypoints) - 1):
        w0, w1 = waypoints[i], waypoints[i + 1]
        if w0["t_s"] <= t_s <= w1["t_s"]:
            dt = float(w1["t_s"] - w0["t_s"])
            alpha = (t_s - float(w0["t_s"])) / dt if dt > 0 else 0.0
            x = float(w0["x_m"]) + alpha * (float(w1["x_m"]) - float(w0["x_m"]))
            y = float(w0["y_m"]) + alpha * (float(w1["y_m"]) - float(w0["y_m"]))
            return x, y
    return float(waypoints[-1]["x_m"]), float(waypoints[-1]["y_m"])


def _schedule_state(schedule: list[dict[str, Any]], t_s: float, default: str) -> str:
    for entry in schedule:
        if float(entry["start_s"]) <= t_s < float(entry["end_s"]):
            return str(entry["state"])
    return default


def _obs(
    *,
    det_id: str,
    camera_id: str,
    frame_index: int,
    captured_at: datetime,
    x_m: float,
    y_m: float,
    calibrated: bool = True,
    embedding: list[float] | None = None,
    face_anchor: FaceAnchor | None = None,
) -> WorldObservation:
    return WorldObservation(
        camera_id=camera_id,
        frame_index=frame_index,
        captured_at=captured_at,
        floor_point=FloorPoint(x_mm=int(x_m * 1000), y_mm=int(y_m * 1000), calibrated=calibrated),
        bbox=BoundingBox(x_min=100, y_min=100, x_max=220, y_max=400),
        embedding=embedding if embedding is not None else [1.0, 0.0, 0.0, 0.0],
        detection_confidence=0.92,
        detection_id=det_id,
        quality=0.65,
        face_anchor=face_anchor,
        floor_cov_random=(0.04, 0.0, 0.0, 0.04) if calibrated else None,
        footpoint_reliable=True,
    )


async def _build_stack(
    *,
    identities: list[str],
    wearable_cfg: WearableConfig,
    resolver_cfg: ResolverConfig | None = None,
) -> tuple[WorldTracker, InMemoryPHRepository, InMemoryWearableAssociationRepository]:
    gallery = InMemoryGalleryRepository()
    for uid in identities:
        await gallery.upsert_identity(Identity(identity_id=uid, display_name=uid, enrolled_at=T0))
    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()
    assoc_repo = InMemoryWearableAssociationRepository()
    resolver = IdentityResolver(
        gallery_repo=gallery,
        config=resolver_cfg
        or ResolverConfig(
            enable_duplicate_active_identity_guard=True,
            arcface_authority_calibrated_confidence=0.80,
            prior_maintenance_max_age_s=65.0,
            enable_sticky_maintenance=True,
        ),
        wearable_config=wearable_cfg,
    )
    tracker = WorldTracker(
        ph_repo=ph_repo,
        obs_repo=obs_repo,
        config=WorldTrackerConfig(min_observations_to_publish=1),
        identity_resolver=resolver,
        gallery_repo=gallery,
        wearable_config=wearable_cfg,
    )
    return tracker, ph_repo, assoc_repo


class TestWearableReplayScenarios:
    @pytest.mark.asyncio
    async def test_single_wearer_commits_and_attests_without_face(self) -> None:
        scenario = yaml.safe_load((_SCENARIOS_DIR / "walker_uwb.yaml").read_text(encoding="utf-8"))
        rooms = {
            r["room_id"]: [
                (r["x_min_m"], r["y_min_m"]),
                (r["x_max_m"], r["y_min_m"]),
                (r["x_max_m"], r["y_max_m"]),
                (r["x_min_m"], r["y_max_m"]),
            ]
            for r in scenario["rooms"]
        }
        tag = scenario["tags"][0]
        tracker, ph_repo, assoc_repo = await _build_stack(
            identities=["resident-1", "caregiver-1"],
            wearable_cfg=WearableConfig(mode="enabled"),
        )
        buf = WearableFixBuffer()

        committed_id: str | None = None
        attested_grade = WearableAttestationGrade.NONE
        for sec in range(0, 16):
            ts = T0 + timedelta(seconds=sec)
            x, y = _interp_waypoint(tag["waypoints"], float(sec))
            buf.add(
                WearableFix(
                    tag_id=tag["tag_id"],
                    member_id=tag["member_id"],
                    measured_at=ts,
                    kind=FixKind.POSITION,
                    x_m=x,
                    y_m=y,
                    cov_m2=(0.015, 0.0, 0.0, 0.015),
                    wear_state=WearState.ON_BODY,
                    motion_state=MotionState.MOVING,
                    epoch_seq=sec + 1,
                )
            )
            res = await tracker.step(
                observations=[
                    _obs(
                        det_id=f"d-{sec}",
                        camera_id="cam-1",
                        frame_index=sec,
                        captured_at=ts,
                        x_m=x,
                        y_m=y,
                    )
                ],
                now=ts,
                room_polygons=rooms,
                camera_room_map={"cam-1": "1"},
                camera_room_ids={"cam-1": "1"},
                wearable_fixes=buf.recent(now=ts, max_age_s=15.0),
            )
            for tr in res.wearable_association_transitions:
                await assoc_repo.save_segment(tr.segment)
            if res.identity_decisions:
                committed_id = res.identity_decisions[0].identity_id
            for ev in res.wearable_evidence_by_ph.values():
                attested_grade = ev.attestation_grade

        assert committed_id == "resident-1"
        assert attested_grade == WearableAttestationGrade.TRAJECTORY
        persisted = await assoc_repo.list_recent()
        assert len(persisted) == 1
        assert persisted[0].attestation_grade == WearableAttestationGrade.TRAJECTORY
        assert persisted[0].identity_id == "resident-1"
        open_phs = await ph_repo.list_open()
        assert len(open_phs) == 1 and open_phs[0].current_identity_id == "resident-1"

    @pytest.mark.asyncio
    async def test_two_wearers_cross_paths_zero_swaps(self) -> None:
        rooms = {"1": [(0.0, 0.0), (8.0, 0.0), (8.0, 6.0), (0.0, 6.0)]}
        tracker, _, _ = await _build_stack(
            identities=["alice", "bob"],
            wearable_cfg=WearableConfig(mode="enabled"),
        )
        buf = WearableFixBuffer()
        emb_a = [1.0, 0.0, 0.0, 0.0]
        emb_b = [0.0, 1.0, 0.0, 0.0]

        decisions_by_ph: dict[str, set[str]] = {}
        for sec in range(0, 14):
            ts = T0 + timedelta(seconds=sec)
            # Alice walks left->right at y=1.5; Bob walks right->left at y=4.5 (clearly separated).
            ax, ay = 1.0 + 0.4 * sec, 1.5
            bx, by = 6.5 - 0.4 * sec, 4.5
            buf.add(
                WearableFix(
                    "tag-a",
                    "alice",
                    ts,
                    FixKind.POSITION,
                    x_m=ax,
                    y_m=ay,
                    cov_m2=(0.02, 0, 0, 0.02),
                    wear_state=WearState.ON_BODY,
                    motion_state=MotionState.MOVING,
                    epoch_seq=sec,
                )
            )
            buf.add(
                WearableFix(
                    "tag-b",
                    "bob",
                    ts,
                    FixKind.POSITION,
                    x_m=bx,
                    y_m=by,
                    cov_m2=(0.02, 0, 0, 0.02),
                    wear_state=WearState.ON_BODY,
                    motion_state=MotionState.MOVING,
                    epoch_seq=sec,
                )
            )
            res = await tracker.step(
                observations=[
                    _obs(
                        det_id=f"da-{sec}",
                        camera_id="cam-1",
                        frame_index=sec,
                        captured_at=ts,
                        x_m=ax,
                        y_m=ay,
                        embedding=emb_a,
                    ),
                    _obs(
                        det_id=f"db-{sec}",
                        camera_id="cam-1",
                        frame_index=sec,
                        captured_at=ts,
                        x_m=bx,
                        y_m=by,
                        embedding=emb_b,
                    ),
                ],
                now=ts,
                room_polygons=rooms,
                camera_room_map={"cam-1": "1"},
                camera_room_ids={"cam-1": "1"},
                wearable_fixes=buf.recent(now=ts, max_age_s=15.0),
            )
            for d in res.identity_decisions:
                if d.identity_id:
                    decisions_by_ph.setdefault(d.ph_id, set()).add(d.identity_id)

        assert len(decisions_by_ph) == 2
        # Each PH committed to exactly one identity, zero swaps!
        assert all(len(ids) == 1 for ids in decisions_by_ph.values())
        assert {next(iter(ids)) for ids in decisions_by_ph.values()} == {"alice", "bob"}

    @pytest.mark.asyncio
    async def test_close_walk_ambiguous_stays_unknown(self) -> None:
        rooms = {"1": [(0.0, 0.0), (8.0, 0.0), (8.0, 6.0), (0.0, 6.0)]}
        tracker, _, _ = await _build_stack(
            identities=["alice", "bob"],
            wearable_cfg=WearableConfig(mode="enabled"),
        )
        buf = WearableFixBuffer()
        emb_a = [1.0, 0.0, 0.0, 0.0]
        emb_b = [0.0, 1.0, 0.0, 0.0]

        for sec in range(0, 10):
            ts = T0 + timedelta(seconds=sec)
            # Two people walking 0.25 m apart side-by-side: within innovation covariance!
            ax, ay = 2.0 + 0.2 * sec, 2.0
            bx, by = 2.0 + 0.2 * sec, 2.25
            buf.add(
                WearableFix(
                    "tag-a",
                    "alice",
                    ts,
                    FixKind.POSITION,
                    x_m=ax,
                    y_m=ay,
                    cov_m2=(0.16, 0, 0, 0.16),
                    wear_state=WearState.ON_BODY,
                    motion_state=MotionState.MOVING,
                    epoch_seq=sec,
                )
            )
            res = await tracker.step(
                observations=[
                    _obs(
                        det_id=f"da-{sec}",
                        camera_id="cam-1",
                        frame_index=sec,
                        captured_at=ts,
                        x_m=ax,
                        y_m=ay,
                        embedding=emb_a,
                    ),
                    _obs(
                        det_id=f"db-{sec}",
                        camera_id="cam-1",
                        frame_index=sec,
                        captured_at=ts,
                        x_m=bx,
                        y_m=by,
                        embedding=emb_b,
                    ),
                ],
                now=ts,
                room_polygons=rooms,
                camera_room_map={"cam-1": "1"},
                camera_room_ids={"cam-1": "1"},
                wearable_fixes=buf.recent(now=ts, max_age_s=15.0),
            )
            assert res.wearable_evidence_by_ph == {}
            assert all(d.identity_id is None for d in res.identity_decisions)

    @pytest.mark.asyncio
    async def test_swapped_wearable_calibrated_face_conflict_and_swap_suspected(self) -> None:
        rooms = {"1": [(0.0, 0.0), (6.0, 0.0), (6.0, 5.0), (0.0, 5.0)]}
        tracker, _, assoc_repo = await _build_stack(
            identities=["alice", "bob"],
            wearable_cfg=WearableConfig(mode="enabled"),
        )
        buf = WearableFixBuffer()
        swap_flagged = False

        # Alice is wearing Bob's tag; calibrated direct face sees Alice on 3 rounds
        # starting when tag-bob first accumulates min_fixes (sec=4, 5, 6).
        conflict_secs = {4, 5, 6}
        for sec in range(0, 8):
            ts = T0 + timedelta(seconds=sec)
            x, y = 2.0 + 0.1 * sec, 2.5
            buf.add(
                WearableFix(
                    "tag-bob",
                    "bob",
                    ts,
                    FixKind.POSITION,
                    x_m=x,
                    y_m=y,
                    cov_m2=(0.02, 0, 0, 0.02),
                    wear_state=WearState.ON_BODY,
                    motion_state=MotionState.MOVING,
                    epoch_seq=sec,
                )
            )
            anchors: list[FaceAnchor] = []
            if sec in conflict_secs:
                anchors.append(
                    FaceAnchor(
                        person_id="alice",
                        match_score=0.90,
                        calibrated_confidence=0.95,
                        detection_id=f"d-{sec}",
                        camera_id="cam-1",
                        captured_at=ts,
                        origin=AnchorOrigin.NATIVE,
                        aggregation=AnchorAggregation.FRAME,
                    )
                )
            res = await tracker.step(
                observations=[
                    _obs(
                        det_id=f"d-{sec}",
                        camera_id="cam-1",
                        frame_index=sec,
                        captured_at=ts,
                        x_m=x,
                        y_m=y,
                        face_anchor=anchors[0] if anchors else None,
                    )
                ],
                now=ts,
                room_polygons=rooms,
                camera_room_map={"cam-1": "1"},
                camera_room_ids={"cam-1": "1"},
                face_anchors=anchors,
                wearable_fixes=buf.recent(now=ts, max_age_s=15.0),
            )
            for tr in res.wearable_association_transitions:
                await assoc_repo.save_segment(tr.segment)
                if tr.segment.swap_suspected:
                    swap_flagged = True
            if sec in conflict_secs:
                assert res.identity_decisions[0].identity_id is None
                assert res.identity_decisions[0].conflict == "wearable_face"

        assert swap_flagged is True
        segments = await assoc_repo.list_recent(tag_id="tag-bob")
        assert any(
            s.swap_suspected and s.close_reason == "wearable_face_conflict" for s in segments
        )

    @pytest.mark.asyncio
    async def test_off_body_dropped_and_uncalibrated_sole_vs_shared_room(self) -> None:
        # 1. Off-body from parked_tag_table.yaml is dropped at subscriber decode.
        site = WearableSiteConfig(
            bindings=(
                WearableBinding("tag-parked-band", "resident-1", T0 - timedelta(hours=1)),
                WearableBinding("tag-caregiver-band", "caregiver-1", T0 - timedelta(hours=1)),
            )
        )
        buf = WearableFixBuffer()
        sub = WearableFixSubscriber(
            redis_client=None,
            buffer=buf,
            site_config_map=WearableSiteConfigMap(site),
        )
        off_msg = wearable_pb2.TagFix(
            tag_id="tag-parked-band",
            measured_at_unix_ns=_unix_ns(T0),
            kind=wearable_pb2.FIX_KIND_ROOM,
            room_id="1",
            wear_state=wearable_pb2.WEAR_STATE_OFF_BODY,
            motion_state=wearable_pb2.MOTION_STATE_STILL,
        )
        sub.handle_fields({b"fix": off_msg.SerializeToString()}, now=T0)
        assert buf.recent(now=T0, max_age_s=15.0) == ()

        # 2. Uncalibrated camera: sole occupant commits on room-class; shared room rejects.
        rooms = {
            "1": [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)],
            "2": [(5.0, 0.0), (10.0, 0.0), (10.0, 5.0), (5.0, 5.0)],
        }
        tracker, _, _ = await _build_stack(
            identities=["resident-1", "caregiver-1"],
            wearable_cfg=WearableConfig(mode="enabled"),
        )
        room_buf = WearableFixBuffer()
        sole_committed = False
        for sec in range(0, 8):
            ts = T0 + timedelta(seconds=sec)
            room_buf.add(
                WearableFix(
                    "tag-r1",
                    "resident-1",
                    ts,
                    FixKind.ROOM,
                    room_id="1",
                    room_probability=0.95,
                    motion_state=MotionState.MOVING,
                    epoch_seq=sec,
                )
            )
            res = await tracker.step(
                observations=[
                    _obs(
                        det_id=f"u1-{sec}",
                        camera_id="cam-uncal",
                        frame_index=sec,
                        captured_at=ts,
                        x_m=200.0,
                        y_m=200.0,
                        calibrated=False,
                    )
                ],
                now=ts,
                room_polygons=rooms,
                camera_room_map={"cam-uncal": "1"},
                camera_room_ids={"cam-uncal": "1"},
                wearable_fixes=room_buf.recent(now=ts, max_age_s=15.0),
            )
            # Confirm uncalibrated virtual coordinates never entered PHFloorHistory!
            for ph in res.updated_phs:
                assert not tracker.ph_floor_history.has_calibrated_history(ph.ph_id)
            if res.identity_decisions and res.identity_decisions[0].identity_id == "resident-1":
                sole_committed = True
        assert sole_committed is True

        # Second uncalibrated PH enters same room -> room-class yields zero evidence.
        ts2 = T0 + timedelta(seconds=9)
        room_buf.add(
            WearableFix(
                "tag-r1",
                "resident-1",
                ts2,
                FixKind.ROOM,
                room_id="1",
                room_probability=0.95,
                motion_state=MotionState.MOVING,
                epoch_seq=9,
            )
        )
        res_shared = await tracker.step(
            observations=[
                _obs(
                    det_id="u1-9",
                    camera_id="cam-uncal",
                    frame_index=9,
                    captured_at=ts2,
                    x_m=200.0,
                    y_m=200.0,
                    calibrated=False,
                    embedding=[1.0, 0.0, 0.0, 0.0],
                ),
                _obs(
                    det_id="u2-9",
                    camera_id="cam-uncal",
                    frame_index=9,
                    captured_at=ts2,
                    x_m=203.0,
                    y_m=203.0,
                    calibrated=False,
                    embedding=[0.0, 1.0, 0.0, 0.0],
                ),
            ],
            now=ts2,
            room_polygons=rooms,
            camera_room_map={"cam-uncal": "1"},
            camera_room_ids={"cam-uncal": "1"},
            wearable_fixes=room_buf.recent(now=ts2, max_age_s=15.0),
        )
        assert res_shared.wearable_evidence_by_ph == {}

    @pytest.mark.asyncio
    async def test_hybrid_uwb_burst_and_ble_continuity(self) -> None:
        """Drive two_people_uwb_burst.yaml: 12 s UWB burst (3 s batches) + 60 s BLE continuity."""
        scenario = yaml.safe_load(
            (_SCENARIOS_DIR / "two_people_uwb_burst.yaml").read_text(encoding="utf-8")
        )
        rooms = {
            r["room_id"]: [
                (r["x_min_m"], r["y_min_m"]),
                (r["x_max_m"], r["y_min_m"]),
                (r["x_max_m"], r["y_max_m"]),
                (r["x_min_m"], r["y_max_m"]),
            ]
            for r in scenario["rooms"]
        }
        tags_by_id = {t["tag_id"]: t for t in scenario["tags"]}
        senior_ble = tags_by_id["tag-senior-ble"]
        watch_ble = tags_by_id["tag-watch-ble"]
        watch_uwb = tags_by_id["tag-watch-uwb"]

        tracker, ph_repo, assoc_repo = await _build_stack(
            identities=["resident-1", "caregiver-1"],
            wearable_cfg=WearableConfig(mode="enabled"),
            resolver_cfg=ResolverConfig(
                enable_duplicate_active_identity_guard=True,
                prior_maintenance_max_age_s=65.0,
                enable_sticky_maintenance=True,
            ),
        )
        buf = WearableFixBuffer(max_age_s=30.0)
        pending_uwb_batch: list[WearableFix] = []

        emb_senior = [1.0, 0.0, 0.0, 0.0]
        emb_caregiver = [0.0, 1.0, 0.0, 0.0]
        caregiver_ph_id: str | None = None
        caregiver_attested_at_s: int | None = None
        caregiver_identities_after_burst: set[str | None] = set()

        for sec in range(0, 73):
            ts = T0 + timedelta(seconds=sec)
            sx, sy = _interp_waypoint(senior_ble["waypoints"], float(sec))
            cx, cy = _interp_waypoint(watch_ble["waypoints"], float(sec))

            # Continuous wide-covariance BLE RSSI (sigma = 2.0 m) for both people.
            buf.add(
                WearableFix(
                    tag_id=senior_ble["tag_id"],
                    member_id=senior_ble["member_id"],
                    measured_at=ts,
                    kind=FixKind.POSITION,
                    x_m=sx,
                    y_m=sy,
                    cov_m2=(4.0, 0.0, 0.0, 4.0),
                    wear_state=WearState.UNSPECIFIED,
                    motion_state=MotionState(
                        _schedule_state(senior_ble["motion_schedule"], float(sec), "still")
                    ),
                    epoch_seq=sec + 1,
                )
            )
            buf.add(
                WearableFix(
                    tag_id=watch_ble["tag_id"],
                    member_id=watch_ble["member_id"],
                    measured_at=ts,
                    kind=FixKind.POSITION,
                    x_m=cx,
                    y_m=cy,
                    cov_m2=(4.0, 0.0, 0.0, 4.0),
                    wear_state=WearState.ON_BODY,
                    motion_state=MotionState(
                        _schedule_state(watch_ble["motion_schedule"], float(sec), "still")
                    ),
                    epoch_seq=sec + 1,
                )
            )

            # 12 s UWB burst in 3 s micro-batches with 0.15 s clock skew.
            if sec <= 12:
                ux, uy = _interp_waypoint(watch_uwb["waypoints"], float(sec))
                pending_uwb_batch.append(
                    WearableFix(
                        tag_id=watch_uwb["tag_id"],
                        member_id=watch_uwb["member_id"],
                        measured_at=ts - timedelta(seconds=0.15),
                        kind=FixKind.POSITION,
                        x_m=ux,
                        y_m=uy,
                        cov_m2=(0.0144, 0.0, 0.0, 0.0144),
                        wear_state=WearState.ON_BODY,
                        motion_state=MotionState.MOVING,
                        epoch_seq=sec + 1,
                    )
                )
                if sec % 3 == 0 and sec > 0:
                    for uf in pending_uwb_batch:
                        buf.add(uf)
                    pending_uwb_batch.clear()

            res = await tracker.step(
                observations=[
                    _obs(
                        det_id=f"s-{sec}",
                        camera_id="cam-1",
                        frame_index=sec,
                        captured_at=ts,
                        x_m=sx,
                        y_m=sy,
                        embedding=emb_senior,
                    ),
                    _obs(
                        det_id=f"c-{sec}",
                        camera_id="cam-1",
                        frame_index=sec,
                        captured_at=ts,
                        x_m=cx,
                        y_m=cy,
                        embedding=emb_caregiver,
                    ),
                ],
                now=ts,
                room_polygons=rooms,
                camera_room_map={"cam-1": "1"},
                camera_room_ids={"cam-1": "1"},
                wearable_fixes=buf.recent(now=ts, max_age_s=15.0),
            )
            for tr in res.wearable_association_transitions:
                await assoc_repo.save_segment(tr.segment)

            cg_ph = res.det_to_ph.get(f"c-{sec}")
            if cg_ph:
                caregiver_ph_id = cg_ph
                wev = res.wearable_evidence_by_ph.get(cg_ph)
                if (
                    wev is not None
                    and wev.attestation_grade == WearableAttestationGrade.TRAJECTORY
                    and caregiver_attested_at_s is None
                ):
                    caregiver_attested_at_s = sec

            if sec > 12 and caregiver_ph_id is not None:
                for d in res.identity_decisions:
                    if d.ph_id == caregiver_ph_id:
                        caregiver_identities_after_burst.add(d.identity_id)

        assert caregiver_ph_id is not None
        assert caregiver_attested_at_s is not None
        assert caregiver_attested_at_s <= 12
        # Retains identity for 60 s after UWB burst ends without swapping to resident-1!
        assert caregiver_identities_after_burst == {"caregiver-1"}
        # Persisted segment reached trajectory attestation.
        segs = await assoc_repo.list_by_ph(caregiver_ph_id)
        assert any(s.attestation_grade == WearableAttestationGrade.TRAJECTORY for s in segs)

        # Senior leaves room at t=73 (PH closes at t=88 after 15 s grace, sole-occupant
        # window opens at t=99); caregiver walks to center of room and watch's continuous
        # BLE resumes refreshing the independent-evidence clock!
        resumed_ble = False
        for sec in range(73, 106):
            ts = T0 + timedelta(seconds=sec)
            alpha = min(1.0, (sec - 72) / 10.0)
            cx = 5.2 + alpha * (3.5 - 5.2)
            cy = 4.2 + alpha * (3.0 - 4.2)
            buf.add(
                WearableFix(
                    tag_id=watch_ble["tag_id"],
                    member_id="caregiver-1",
                    measured_at=ts,
                    kind=FixKind.POSITION,
                    x_m=cx,
                    y_m=cy,
                    cov_m2=(4.0, 0.0, 0.0, 4.0),
                    wear_state=WearState.ON_BODY,
                    motion_state=MotionState.MOVING,
                    epoch_seq=sec + 1,
                )
            )
            res = await tracker.step(
                observations=[
                    _obs(
                        det_id=f"c-{sec}",
                        camera_id="cam-1",
                        frame_index=sec,
                        captured_at=ts,
                        x_m=cx,
                        y_m=cy,
                        embedding=emb_caregiver,
                    ),
                ],
                now=ts,
                room_polygons=rooms,
                camera_room_map={"cam-1": "1"},
                camera_room_ids={"cam-1": "1"},
                wearable_fixes=buf.recent(now=ts, max_age_s=15.0),
            )
            if caregiver_ph_id in res.wearable_evidence_by_ph:
                resumed_ble = True
        assert resumed_ble is True
        cg_final = await ph_repo.get(caregiver_ph_id)
        assert cg_final is not None and cg_final.current_identity_id == "caregiver-1"

    @pytest.mark.asyncio
    async def test_shadow_mode_zero_live_diff_from_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reg = CollectorRegistry()
        monkeypatch.setattr(metrics_mod, "metrics", build_metrics(reg))

        rooms = {"1": [(0.0, 0.0), (6.0, 0.0), (6.0, 5.0), (0.0, 5.0)]}
        off_tracker, _, _ = await _build_stack(
            identities=["alice", "bob"], wearable_cfg=WearableConfig(mode="off")
        )
        shadow_tracker, _, shadow_assoc = await _build_stack(
            identities=["alice", "bob"], wearable_cfg=WearableConfig(mode="shadow")
        )
        buf = WearableFixBuffer()

        for sec in range(0, 12):
            ts = T0 + timedelta(seconds=sec)
            x, y = 1.0 + 0.3 * sec, 2.0
            buf.add(
                WearableFix(
                    "tag-a",
                    "alice",
                    ts,
                    FixKind.POSITION,
                    x_m=x,
                    y_m=y,
                    cov_m2=(0.02, 0, 0, 0.02),
                    wear_state=WearState.ON_BODY,
                    motion_state=MotionState.MOVING,
                    epoch_seq=sec,
                )
            )
            obs = [
                _obs(
                    det_id=f"d-{sec}",
                    camera_id="cam-1",
                    frame_index=sec,
                    captured_at=ts,
                    x_m=x,
                    y_m=y,
                )
            ]
            r_off = await off_tracker.step(
                observations=obs,
                now=ts,
                room_polygons=rooms,
                camera_room_map={"cam-1": "1"},
                camera_room_ids={"cam-1": "1"},
            )
            r_sh = await shadow_tracker.step(
                observations=obs,
                now=ts,
                room_polygons=rooms,
                camera_room_map={"cam-1": "1"},
                camera_room_ids={"cam-1": "1"},
                wearable_fixes=buf.recent(now=ts, max_age_s=15.0),
            )
            for tr in r_sh.wearable_association_transitions:
                await shadow_assoc.save_segment(tr.segment)
            assert [d.identity_id for d in r_sh.identity_decisions] == [
                d.identity_id for d in r_off.identity_decisions
            ]

        assert (
            metrics_mod.metrics.cts_wearable_shadow_outcomes_total.labels(
                outcome=str(WearableShadowOutcome.WOULD_NAME_UNKNOWN)
            )._value.get()
            >= 1.0
        )
        assert len(await shadow_assoc.list_recent()) == 1

    @pytest.mark.asyncio
    async def test_attestation_independent_of_gallery_state(self) -> None:
        """Association and attestation match across empty, agreeing, and contradicting galleries."""
        from app.domain import GalleryEmbedding

        rooms = {"1": [(0.0, 0.0), (6.0, 0.0), (6.0, 5.0), (0.0, 5.0)]}
        emb = [1.0, 0.0, 0.0, 0.0]

        async def _run_with_gallery_identity(gallery_owner: str | None) -> tuple[str, float, str]:
            gallery = InMemoryGalleryRepository()
            for uid in ("alice", "bob"):
                await gallery.upsert_identity(
                    Identity(identity_id=uid, display_name=uid, enrolled_at=T0)
                )
            if gallery_owner is not None:
                await gallery.upsert_gallery_entry(
                    GalleryEmbedding(
                        gallery_entry_id=f"ge-{gallery_owner}",
                        identity_id=gallery_owner,
                        embedding=emb,
                        seen_at=T0,
                        state="operator_verified",
                    )
                )
            ph_repo = InMemoryPHRepository()
            obs_repo = InMemoryWorldObservationRepository()
            w_cfg = WearableConfig(mode="enabled")
            resolver = IdentityResolver(gallery_repo=gallery, wearable_config=w_cfg)
            tracker = WorldTracker(
                ph_repo=ph_repo,
                obs_repo=obs_repo,
                config=WorldTrackerConfig(min_observations_to_publish=1),
                identity_resolver=resolver,
                gallery_repo=gallery,
                wearable_config=w_cfg,
            )
            buf = WearableFixBuffer()
            last_ev = None
            for sec in range(0, 13):
                ts = T0 + timedelta(seconds=sec)
                x, y = 1.0 + 0.25 * sec, 2.5
                buf.add(
                    WearableFix(
                        "tag-a",
                        "alice",
                        ts,
                        FixKind.POSITION,
                        x_m=x,
                        y_m=y,
                        cov_m2=(0.015, 0.0, 0.0, 0.015),
                        wear_state=WearState.ON_BODY,
                        motion_state=MotionState.MOVING,
                        epoch_seq=sec,
                    )
                )
                res = await tracker.step(
                    observations=[
                        _obs(
                            det_id=f"d-{sec}",
                            camera_id="cam-1",
                            frame_index=sec,
                            captured_at=ts,
                            x_m=x,
                            y_m=y,
                            embedding=emb,
                        )
                    ],
                    now=ts,
                    room_polygons=rooms,
                    camera_room_map={"cam-1": "1"},
                    camera_room_ids={"cam-1": "1"},
                    wearable_fixes=buf.recent(now=ts, max_age_s=15.0),
                )
                if res.wearable_evidence_by_ph:
                    last_ev = next(iter(res.wearable_evidence_by_ph.values()))
            assert last_ev is not None
            return (
                last_ev.identity_id,
                round(last_ev.window_llr, 6),
                str(last_ev.attestation_grade),
            )

        empty_res = await _run_with_gallery_identity(None)
        agree_res = await _run_with_gallery_identity("alice")
        contradict_res = await _run_with_gallery_identity("bob")
        assert empty_res == agree_res == contradict_res
        assert empty_res[0] == "alice"
        assert empty_res[2] == str(WearableAttestationGrade.TRAJECTORY)

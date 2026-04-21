"""Tests for in-memory repository implementations."""

from datetime import UTC, datetime, timedelta

import pytest

from app.domain import (
    BoundingBox,
    CameraConfig,
    Detection,
    FrameRef,
    GalleryEmbedding,
    GlobalTrack,
    Identity,
    IdentityCandidate,
    IdentityCorrection,
    IdentityRevision,
    PersonActivity,
    PrivacyZone,
    StreamAssignment,
    StreamConfig,
    TrackingEvent,
    Tracklet,
)
from app.storage import (
    InMemoryActivityRepository,
    InMemoryAssignmentRepository,
    InMemoryCorrectionRepository,
    InMemoryGalleryRepository,
    InMemoryPrivacyRepository,
    InMemorySettingsRepository,
    InMemoryTrackingRepository,
)

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
BOX = BoundingBox(0, 0, 10, 20)
FRAME_REF = FrameRef(
    minio_key="frames/cam1/2026/01/01/12/00000000000000000042-1.jpg",
    width=640,
    height=480,
    frame_index=42,
    capture_time=NOW,
)


@pytest.fixture
def tracking_repo() -> InMemoryTrackingRepository:
    return InMemoryTrackingRepository()


@pytest.fixture
def gallery_repo() -> InMemoryGalleryRepository:
    return InMemoryGalleryRepository()


@pytest.fixture
def settings_repo() -> InMemorySettingsRepository:
    return InMemorySettingsRepository()


@pytest.fixture
def activity_repo() -> InMemoryActivityRepository:
    return InMemoryActivityRepository()


@pytest.fixture
def assignment_repo() -> InMemoryAssignmentRepository:
    return InMemoryAssignmentRepository()


@pytest.fixture
def correction_repo() -> InMemoryCorrectionRepository:
    return InMemoryCorrectionRepository()


@pytest.fixture
def privacy_repo() -> InMemoryPrivacyRepository:
    return InMemoryPrivacyRepository()


# -----------------------------------------------------------------------
# TrackingRepository
# -----------------------------------------------------------------------


async def test_save_and_get_tracking_event(
    tracking_repo: InMemoryTrackingRepository,
) -> None:
    event = TrackingEvent(
        event_id="event-1",
        camera_id="cam1",
        event_time=NOW,
        frame_index=42,
        frame_ref=FRAME_REF,
    )
    eid = await tracking_repo.save_tracking_event(event)
    assert eid == event.event_id
    fetched = await tracking_repo.get_tracking_event(eid)
    assert fetched is not None
    assert fetched.event_id == event.event_id


async def test_save_detections(tracking_repo: InMemoryTrackingRepository) -> None:
    detections = [
        Detection(
            detection_id="d1",
            camera_id="cam1",
            bbox=BOX,
            embedding=[0.1] * 768,
            capture_time=NOW,
            event_time=NOW,
        )
    ]
    await tracking_repo.save_detections("event-1", detections)
    assert tracking_repo._detections["event-1"][0].detection_id == "d1"


async def test_save_and_get_tracklet(tracking_repo: InMemoryTrackingRepository) -> None:
    t = Tracklet(
        tracklet_id="tracklet-1",
        camera_id="cam1",
        detection_ids=["d1"],
        started_at=NOW,
    )
    await tracking_repo.save_tracklet(t)
    fetched = await tracking_repo.get_tracklet(t.tracklet_id)
    assert fetched is not None
    assert fetched.camera_id == "cam1"


async def test_save_tracklet_merges_growth(tracking_repo: InMemoryTrackingRepository) -> None:
    original = Tracklet(
        tracklet_id="tracklet-merge",
        camera_id="cam1",
        detection_ids=["d1"],
        started_at=NOW,
    )
    updated = Tracklet(
        tracklet_id="tracklet-merge",
        camera_id="cam1",
        detection_ids=["d1", "d2"],
        started_at=NOW,
        ended_at=NOW + timedelta(seconds=5),
        state="terminated",
    )

    await tracking_repo.save_tracklet(original)
    await tracking_repo.save_tracklet(updated)

    fetched = await tracking_repo.get_tracklet("tracklet-merge")
    assert fetched is not None
    assert fetched.detection_ids == ["d1", "d2"]
    assert fetched.state == "terminated"
    assert fetched.ended_at == NOW + timedelta(seconds=5)


async def test_save_global_track(tracking_repo: InMemoryTrackingRepository) -> None:
    track_id = "gt-merge-test"
    t1 = GlobalTrack(
        global_track_id=track_id,
        camera_ids=["cam1"],
        tracklet_ids=["t1"],
        started_at=NOW,
        last_seen_at=NOW,
    )
    t2 = GlobalTrack(
        global_track_id=track_id,
        camera_ids=["cam2"],
        tracklet_ids=["t2"],
        started_at=NOW,
        last_seen_at=NOW + timedelta(seconds=10),
    )
    await tracking_repo.save_global_track(t1)
    await tracking_repo.save_global_track(t2)
    fetched = await tracking_repo.get_global_track(track_id)
    assert fetched is not None
    assert set(fetched.camera_ids) == {"cam1", "cam2"}


async def test_identity_revision(tracking_repo: InMemoryTrackingRepository) -> None:
    rev = IdentityRevision(
        revision_id="revision-1",
        global_track_id="gt1",
        tracklet_ids=["t1"],
        candidates=[IdentityCandidate("i1", "Alice", 0.8)],
        map_identity_id="i1",
        posterior_entropy=0.5,
        revision_time=NOW,
    )
    await tracking_repo.save_identity_revision(rev)
    results = await tracking_repo.list_identity_revisions("gt1")
    assert len(results) == 1
    assert results[0].map_identity_id == "i1"


# -----------------------------------------------------------------------
# GalleryRepository
# -----------------------------------------------------------------------


async def test_upsert_and_get_gallery_entry(
    gallery_repo: InMemoryGalleryRepository,
) -> None:
    identity = Identity(identity_id="i1", display_name="Alice", enrolled_at=NOW)
    entry = GalleryEmbedding(
        gallery_entry_id="g1",
        identity_id="i1",
        embedding=[0.1] * 768,
        seen_at=NOW,
    )
    await gallery_repo.upsert_identity(identity)
    await gallery_repo.upsert_gallery_entry(entry)
    fetched = await gallery_repo.get_gallery_entry("g1")
    assert fetched is not None
    assert fetched.identity_id == "i1"


async def test_search_similar(gallery_repo: InMemoryGalleryRepository) -> None:
    await gallery_repo.upsert_identity(
        Identity(identity_id="i1", display_name="Alice", enrolled_at=NOW)
    )
    await gallery_repo.upsert_identity(
        Identity(identity_id="i2", display_name="Bob", enrolled_at=NOW)
    )
    await gallery_repo.upsert_gallery_entry(
        GalleryEmbedding(
            gallery_entry_id="g1",
            identity_id="i1",
            embedding=[1.0] * 768,
            seen_at=NOW,
        )
    )
    await gallery_repo.upsert_gallery_entry(
        GalleryEmbedding(
            gallery_entry_id="g2",
            identity_id="i2",
            embedding=[-1.0] * 768,
            seen_at=NOW,
        )
    )
    results = await gallery_repo.search_similar([1.0] * 768, limit=2)
    assert len(results) == 2
    assert results[0].identity_id == "i1"


# -----------------------------------------------------------------------
# SettingsRepository
# -----------------------------------------------------------------------


async def test_save_and_get_camera_config(
    settings_repo: InMemorySettingsRepository,
) -> None:
    cfg = CameraConfig(camera_id="c1", name="Kitchen", rtsp_url="rtsp://cam1/stream")
    await settings_repo.save_camera_config(cfg)
    fetched = await settings_repo.get_camera_config("c1")
    assert fetched is not None
    assert fetched.name == "Kitchen"


async def test_save_and_get_stream_config(
    settings_repo: InMemorySettingsRepository,
) -> None:
    cfg = StreamConfig(stream_id="s1", camera_id="c1", frame_rate=5.0)
    await settings_repo.save_stream_config(cfg)
    fetched = await settings_repo.get_stream_config("s1")
    assert fetched is not None
    assert fetched.frame_rate == 5.0


# -----------------------------------------------------------------------
# ActivityRepository
# -----------------------------------------------------------------------


async def test_save_and_get_activity(
    activity_repo: InMemoryActivityRepository,
) -> None:
    act = PersonActivity(
        activity_id="activity-1",
        identity_id="i1",
        activity_type="exit",
        camera_id="c1",
        occurred_at=NOW,
    )
    aid = await activity_repo.save_activity(act)
    assert aid == act.activity_id
    fetched = await activity_repo.get_activity(aid)
    assert fetched is not None
    assert fetched.activity_type == "exit"


async def test_list_activities(activity_repo: InMemoryActivityRepository) -> None:
    await activity_repo.save_activity(
        PersonActivity(
            activity_id="activity-2",
            identity_id="i1",
            activity_type="entry",
            camera_id="c1",
            occurred_at=NOW,
        )
    )
    await activity_repo.save_activity(
        PersonActivity(
            activity_id="activity-3",
            identity_id="i1",
            activity_type="exit",
            camera_id="c2",
            occurred_at=NOW + timedelta(minutes=5),
        )
    )
    results = await activity_repo.list_activities(identity_id="i1")
    assert len(results) == 2
    assert results[0].activity_id == "activity-3"


# -----------------------------------------------------------------------
# AssignmentRepository
# -----------------------------------------------------------------------


async def test_save_and_get_assignment(
    assignment_repo: InMemoryAssignmentRepository,
) -> None:
    a = StreamAssignment(stream_id="s1", room_id="kitchen", zone_id="zone1")
    await assignment_repo.save_assignment(a)
    fetched = await assignment_repo.get_assignment("s1")
    assert fetched is not None
    assert fetched.room_id == "kitchen"


async def test_save_and_list_corrections(
    correction_repo: InMemoryCorrectionRepository,
) -> None:
    correction = IdentityCorrection(
        correction_id="correction-1",
        global_track_id="gt1",
        from_identity_id="i1",
        to_identity_id="i2",
        corrected_at=NOW,
    )
    await correction_repo.save_correction(correction)
    results = await correction_repo.list_corrections("gt1")
    assert len(results) == 1
    assert results[0].to_identity_id == "i2"


async def test_save_and_list_privacy_zones(
    privacy_repo: InMemoryPrivacyRepository,
) -> None:
    zone = PrivacyZone(
        zone_id="zone-1",
        camera_id="cam1",
        name="bedroom-mask",
        polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
    )
    await privacy_repo.save_privacy_zone(zone)
    results = await privacy_repo.list_privacy_zones("cam1")
    assert len(results) == 1
    assert results[0].name == "bedroom-mask"

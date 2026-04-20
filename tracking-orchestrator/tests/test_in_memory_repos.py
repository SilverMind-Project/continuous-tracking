"""Tests for in-memory repository implementations."""

import pytest

from app.domain import (
    CameraConfig,
    Detection,
    DetectionId,
    GalleryEntry,
    GlobalTrack,
    IdentityCandidate,
    IdentityRevision,
    PersonActivity,
    StreamAssignment,
    StreamConfig,
    Tracklet,
    TrackingEvent,
)
from app.storage import (
    InMemoryActivityRepository,
    InMemoryAssignmentRepository,
    InMemoryGalleryRepository,
    InMemorySettingsRepository,
    InMemoryTrackingRepository,
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


# -----------------------------------------------------------------------
# TrackingRepository
# -----------------------------------------------------------------------

def test_save_and_get_tracking_event(tracking_repo: InMemoryTrackingRepository) -> None:
    event = TrackingEvent(camera_id="cam1", frame_index=42)
    eid = tracking_repo.save_tracking_event(event)
    assert eid == event.event_id
    fetched = tracking_repo.get_tracking_event(eid)
    assert fetched is not None
    assert fetched.event_id == event.event_id


def test_save_detections(tracking_repo: InMemoryTrackingRepository) -> None:
    detections = [Detection(detection_id="d1", camera_id="cam1")]
    tracking_repo.save_detections(detections)


def test_save_and_get_tracklet(tracking_repo: InMemoryTrackingRepository) -> None:
    t = Tracklet(camera_id="cam1")
    tracking_repo.save_tracklet(t)
    fetched = tracking_repo.get_tracklet(t.tracklet_id)
    assert fetched is not None
    assert fetched.camera_id == "cam1"


def test_save_global_track(tracking_repo: InMemoryTrackingRepository) -> None:
    t1 = GlobalTrack(camera_ids=["cam1"], tracklet_ids=["t1"])
    t2 = GlobalTrack(camera_ids=["cam2"], tracklet_ids=["t2"])
    tracking_repo.save_global_track(t1)
    tracking_repo.save_global_track(t2)
    fetched = tracking_repo.get_global_track(t1.global_track_id)
    assert fetched is not None
    assert set(fetched.camera_ids) == {"cam1", "cam2"}


def test_identity_revision(tracking_repo: InMemoryTrackingRepository) -> None:
    rev = IdentityRevision(
        global_track_id="gt1",
        candidates=[IdentityCandidate("i1", "Alice", 0.8)],
        map_identity_id="i1",
        posterior_entropy=0.5,
    )
    tracking_repo.save_identity_revision(rev)
    results = tracking_repo.list_identity_revisions("gt1")
    assert len(results) == 1
    assert results[0].map_identity_id == "i1"


# -----------------------------------------------------------------------
# GalleryRepository
# -----------------------------------------------------------------------

def test_upsert_and_get_gallery_entry(gallery_repo: InMemoryGalleryRepository) -> None:
    entry = GalleryEntry(identity_id="i1", display_name="Alice", embedding=[0.1] * 512)
    gallery_repo.upsert_gallery_entry(entry)
    fetched = gallery_repo.get_gallery_entry("i1")
    assert fetched is not None
    assert fetched.display_name == "Alice"


def test_search_similar(gallery_repo: InMemoryGalleryRepository) -> None:
    gallery_repo.upsert_gallery_entry(
        GalleryEntry(identity_id="i1", display_name="Alice", embedding=[1.0] * 512)
    )
    gallery_repo.upsert_gallery_entry(
        GalleryEntry(identity_id="i2", display_name="Bob", embedding=[-1.0] * 512)
    )
    results = gallery_repo.search_similar([1.0] * 512, limit=2)
    assert len(results) == 2
    assert results[0].display_name == "Alice"


# -----------------------------------------------------------------------
# SettingsRepository
# -----------------------------------------------------------------------

def test_save_and_get_camera_config(settings_repo: InMemorySettingsRepository) -> None:
    cfg = CameraConfig(camera_id="c1", name="Kitchen", rtsp_url="rtsp://cam1/stream")
    settings_repo.save_camera_config(cfg)
    fetched = settings_repo.get_camera_config("c1")
    assert fetched is not None
    assert fetched.name == "Kitchen"


def test_save_and_get_stream_config(settings_repo: InMemorySettingsRepository) -> None:
    cfg = StreamConfig(stream_id="s1", camera_id="c1", frame_rate=5.0)
    settings_repo.save_stream_config(cfg)
    fetched = settings_repo.get_stream_config("s1")
    assert fetched is not None
    assert fetched.frame_rate == 5.0


# -----------------------------------------------------------------------
# ActivityRepository
# -----------------------------------------------------------------------

def test_save_and_get_activity(activity_repo: InMemoryActivityRepository) -> None:
    act = PersonActivity(identity_id="i1", activity_type="exit", camera_id="c1")
    aid = activity_repo.save_activity(act)
    assert aid == act.activity_id
    fetched = activity_repo.get_activity(aid)
    assert fetched is not None
    assert fetched.activity_type == "exit"


def test_list_activities(activity_repo: InMemoryActivityRepository) -> None:
    activity_repo.save_activity(PersonActivity(identity_id="i1", activity_type="entry", camera_id="c1"))
    activity_repo.save_activity(PersonActivity(identity_id="i1", activity_type="exit", camera_id="c2"))
    results = activity_repo.list_activities(identity_id="i1")
    assert len(results) == 2


# -----------------------------------------------------------------------
# AssignmentRepository
# -----------------------------------------------------------------------

def test_save_and_get_assignment(assignment_repo: InMemoryAssignmentRepository) -> None:
    a = StreamAssignment(stream_id="s1", room_id="kitchen", zone_id="zone1")
    assignment_repo.save_assignment(a)
    fetched = assignment_repo.get_assignment("s1")
    assert fetched is not None
    assert fetched.room_id == "kitchen"

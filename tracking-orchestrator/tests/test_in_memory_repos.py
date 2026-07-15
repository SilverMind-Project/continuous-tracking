"""Tests for in-memory repository implementations."""

from datetime import UTC, datetime, timedelta

import pytest

from app.domain import (
    BoundingBox,
    CameraConfig,
    GalleryEmbedding,
    Identity,
    IdentityCorrection,
    PersonActivity,
    PrivacyZone,
    StreamAssignment,
    StreamConfig,
)
from app.storage import (
    InMemoryActivityRepository,
    InMemoryAssignmentRepository,
    InMemoryCorrectionRepository,
    InMemoryGalleryRepository,
    InMemoryPrivacyRepository,
    InMemorySettingsRepository,
)

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
BOX = BoundingBox(0, 0, 10, 20)


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
            state="operator_verified",
        )
    )
    await gallery_repo.upsert_gallery_entry(
        GalleryEmbedding(
            gallery_entry_id="g2",
            identity_id="i2",
            embedding=[-1.0] * 768,
            seen_at=NOW,
            state="operator_verified",
        )
    )
    results = await gallery_repo.search_similar([1.0] * 768, limit=2)
    assert len(results) == 2
    assert results[0][0].identity_id == "i1"
    assert results[0][1] > results[1][1]


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


async def test_save_camera_config_is_idempotent(
    settings_repo: InMemorySettingsRepository,
) -> None:
    """Repeated saves with the same id keep a single row (upsert semantics)."""
    await settings_repo.save_camera_config(CameraConfig(camera_id="c1", name="first"))
    await settings_repo.save_camera_config(CameraConfig(camera_id="c1", name="second"))
    cameras = await settings_repo.list_camera_configs()
    assert len(cameras) == 1
    assert cameras[0].name == "second"


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

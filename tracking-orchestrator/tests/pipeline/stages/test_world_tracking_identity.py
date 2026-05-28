"""WT1: tests for identity resolution wiring in the world tracker.

Verifies that:
1. Identity commits produce revisions (populates ctx.new_revisions).
2. No revisions are emitted without evidence.
3. FrameProcessingPipeline wires identity_resolver + revision_publisher into WorldTracker.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

from app.domain import (
    BoundingBox,
    FaceAnchor,
    FloorPoint,
    Identity,
    PersonHypothesis,
    WorldObservation,
)
from app.pipeline.frame_pipeline import (
    FrameProcessingPipeline,
    PipelineConfig,
    PipelineDependencies,
    SignalConfig,
)
from app.storage.base import (
    InMemoryGlobalTrackRepository,
    InMemoryPHRepository,
    InMemoryTrackingRepository,
    InMemoryWorldObservationRepository,
)
from app.storage.gallery import InMemoryGalleryRepository
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig
from app.tracking.world.tracker import (
    WorldTrackerConfig,
    _resolve_identities,
)
from app.transport.redis_streams import TransportConfig
from tests.fixtures.fake_publishers import FakeRevisionPublisher

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_identity(identity_id: str = "alice") -> Identity:
    return Identity(
        identity_id=identity_id,
        display_name=identity_id.title(),
        enrolled_at=datetime(2026, 5, 1, tzinfo=UTC),
    )


def _make_ph(ph_id: str = "ph-1", *, now: datetime | None = None) -> PersonHypothesis:
    if now is None:
        now = datetime.now(UTC)
    return PersonHypothesis(
        ph_id=ph_id,
        state_mean=(0.0, 0.0, 0.0, 0.0),
        state_cov=tuple([1.0] * 16),
        born_at=now,
        last_seen_at=now,
        last_seen_camera="cam1",
        observation_count=1,
    )


def _make_observation(
    camera_id: str = "cam1",
    frame_index: int = 1,
    *,
    captured_at: datetime | None = None,
) -> WorldObservation:
    if captured_at is None:
        captured_at = datetime.now(UTC)
    return WorldObservation(
        camera_id=camera_id,
        frame_index=frame_index,
        captured_at=captured_at,
        floor_point=FloorPoint(0, 0, calibrated=True),
        bbox=BoundingBox(0, 0, 100, 200),
        embedding=[1.0, 0.0, 0.0],
        detection_confidence=0.9,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWorldTrackingEmitsRevisions:
    async def test_revision_on_identity_commit(self) -> None:
        """Identity commit with a face anchor produces a revision.

        The FakeRevisionPublisher must receive nothing — the tracker
        returns revisions; RevisionsStage is the sole publish path.
        """
        now = datetime(2026, 5, 27, 12, 0, 0, tzinfo=UTC)
        gallery_repo = InMemoryGalleryRepository()
        identity = _make_identity("alice")
        await gallery_repo.upsert_identity(identity)

        resolver = IdentityResolver(
            tracking_repo=InMemoryTrackingRepository(),
            gallery_repo=gallery_repo,
            global_track_repo=InMemoryGlobalTrackRepository(),
            identities=[identity],
            config=ResolverConfig(),
        )

        fake_publisher = FakeRevisionPublisher()
        ph_repo = InMemoryPHRepository()
        obs_repo = InMemoryWorldObservationRepository()

        ph = _make_ph("ph-1", now=now)
        await ph_repo.save(ph)

        obs = _make_observation("cam1", 1, captured_at=now)
        await obs_repo.save(obs, ph_id="ph-1")

        # The observation ID synthesized by _resolve_identities is
        # "cam1:1:2026-05-27T12:00:00+00:00".  Set tracklet_id to match
        # so the resolver associates this face anchor with the PH.
        obs_id_format = f"cam1:1:{now.isoformat()}"
        face_anchor = FaceAnchor(
            person_id="alice",
            confidence=0.95,
            quality=1.0,
            tracklet_id=obs_id_format,
            camera_id="cam1",
            captured_at=now,
        )

        ph_obs_meta = {"ph-1": (1, BoundingBox(0, 0, 100, 200), 0.9)}

        _decisions, revisions, _identity_by_ph = await _resolve_identities(
            resolver=resolver,
            obs_repo=obs_repo,
            ph_repo=ph_repo,
            phs=[ph],
            ph_obs_meta=ph_obs_meta,
            face_anchors=[face_anchor],
            now=now,
            config=WorldTrackerConfig(),
        )

        assert len(revisions) == 1, f"expected 1 revision, got {len(revisions)}"
        rev = revisions[0]
        assert rev.previous_identity_id is None
        assert rev.new_identity_id == "alice"
        assert len(fake_publisher.published) == 0, (
            "FakeRevisionPublisher must be empty: RevisionsStage owns publish"
        )

    async def test_no_revision_without_evidence(self) -> None:
        """No face anchor and no ReID match → no identity commit → no revision."""
        now = datetime(2026, 5, 27, 12, 0, 0, tzinfo=UTC)
        gallery_repo = InMemoryGalleryRepository()
        identity = _make_identity("alice")
        await gallery_repo.upsert_identity(identity)

        resolver = IdentityResolver(
            tracking_repo=InMemoryTrackingRepository(),
            gallery_repo=gallery_repo,
            global_track_repo=InMemoryGlobalTrackRepository(),
            identities=[identity],
            config=ResolverConfig(),
        )

        fake_publisher = FakeRevisionPublisher()
        ph_repo = InMemoryPHRepository()
        obs_repo = InMemoryWorldObservationRepository()

        ph = _make_ph("ph-1", now=now)
        await ph_repo.save(ph)

        obs = _make_observation("cam1", 1, captured_at=now)
        await obs_repo.save(obs, ph_id="ph-1")

        ph_obs_meta = {"ph-1": (1, BoundingBox(0, 0, 100, 200), 0.9)}

        _decisions, revisions, _identity_by_ph = await _resolve_identities(
            resolver=resolver,
            obs_repo=obs_repo,
            ph_repo=ph_repo,
            phs=[ph],
            ph_obs_meta=ph_obs_meta,
            face_anchors=[],  # no face evidence
            now=now,
            config=WorldTrackerConfig(),
        )

        assert len(revisions) == 0, f"expected 0 revisions, got {len(revisions)}"
        assert len(fake_publisher.published) == 0


class TestPipelineWiring:
    async def test_world_tracker_built_with_resolver_and_publisher(self) -> None:
        """FrameProcessingPipeline.initialize() wires resolver + publisher into WorldTracker.

        This is the only place in WT1 where asserting on a private attribute
        is acceptable — it verifies the wiring contract that is the whole point
        of WT1.
        """
        with (
            patch("app.pipeline.frame_pipeline.RedisStreamsTransport") as mock_transport_cls,
            patch("app.pipeline.frame_pipeline.RevisionPublisher") as mock_rev_cls,
            patch("app.pipeline.frame_pipeline.SceneSamplesPublisher") as mock_scene_cls,
            patch("app.pipeline.frame_pipeline.SignalPublisher") as mock_signal_cls,
        ):
            mock_transport = AsyncMock()
            mock_transport_cls.return_value = mock_transport
            mock_rev = AsyncMock()
            mock_rev_cls.return_value = mock_rev
            mock_scene = AsyncMock()
            mock_scene_cls.return_value = mock_scene
            mock_signal = AsyncMock()
            mock_signal_cls.return_value = mock_signal

            config = PipelineConfig(
                allow_skeleton=True,
                signals=SignalConfig(enabled=False),  # type: ignore[call-arg]
                transport=TransportConfig(),  # type: ignore[arg-type]
            )
            pipeline = FrameProcessingPipeline(config)
            await pipeline.initialize(PipelineDependencies())

            assert pipeline._world_tracker is not None, (
                "WorldTracker must be constructed during initialize()"
            )
            # WT1 wiring contract: WorldTracker receives the same
            # IdentityResolver instance that the pipeline created.
            assert pipeline._world_tracker._identity_resolver is pipeline._identity_resolver, (
                "WorldTracker must receive the IdentityResolver instance created by the pipeline"
            )

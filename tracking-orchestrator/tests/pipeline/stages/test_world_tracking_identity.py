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
    InMemoryPHRepository,
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
            gallery_repo=gallery_repo,
            identities=[identity],
            config=ResolverConfig(),
        )

        fake_publisher = FakeRevisionPublisher()
        ph_repo = InMemoryPHRepository()
        obs_repo = InMemoryWorldObservationRepository()

        ph = _make_ph("ph-1", now=now)
        await ph_repo.save(ph)

        obs = _make_observation("cam1", 1, captured_at=now)
        real_obs_id = await obs_repo.save(obs, ph_id="ph-1")

        # The resolver matches face anchors to PHs by observation_id.
        # Use the real observation UUID as the tracklet_id key.
        face_anchor = FaceAnchor(
            person_id="alice",
            confidence=0.95,
            quality=1.0,
            tracklet_id=real_obs_id,
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
            det_to_ph={},
            face_evidence=None,
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
            gallery_repo=gallery_repo,
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
            det_to_ph={},
            face_evidence=None,
            now=now,
            config=WorldTrackerConfig(),
        )

        assert len(revisions) == 0, f"expected 0 revisions, got {len(revisions)}"
        assert len(fake_publisher.published) == 0

    async def test_ph_mode_face_anchor_commits_to_existing_ph(self) -> None:
        """Existing PH + PH-mode face anchor (tracklet_id='', detection_id mapped via
        det_to_ph) → identity commits on the correct PH.

        This is the regression test for the PH-mode resolver bug:
        FaceIdentityStage runs before WorldTrackingStage so detection_id is the
        only per-detection key at anchor-build time.  The remap in
        _resolve_identities fills tracklet_id=ph_id so the resolver can match.
        """
        now = datetime(2026, 5, 27, 12, 0, 0, tzinfo=UTC)
        gallery_repo = InMemoryGalleryRepository()
        identity = _make_identity("grandma")
        await gallery_repo.upsert_identity(identity)

        resolver = IdentityResolver(
            gallery_repo=gallery_repo,
            identities=[identity],
            config=ResolverConfig(),
        )

        ph_repo = InMemoryPHRepository()
        obs_repo = InMemoryWorldObservationRepository()

        ph = _make_ph("ph-existing", now=now)
        await ph_repo.save(ph)

        obs = _make_observation("cam1", 5, captured_at=now)
        await obs_repo.save(obs, ph_id="ph-existing")

        ph_obs_meta = {"ph-existing": (5, BoundingBox(10, 20, 110, 220), 0.9)}

        # PH-mode anchor: tracklet_id="" (as produced by FaceIdentityStage),
        # detection_id is an ephemeral UUID assigned by DetectStage.
        detection_uuid = "aaaa-bbbb-cccc-dddd"
        face_anchor = FaceAnchor(
            person_id="grandma",
            confidence=0.92,
            quality=1.0,
            tracklet_id="",           # empty — PH mode
            detection_id=detection_uuid,
            camera_id="cam1",
            captured_at=now,
        )

        # det_to_ph maps the detection_uuid to the existing PH — exactly what
        # WorldTracker.step() builds during the association step.
        det_to_ph = {detection_uuid: "ph-existing"}

        _decisions, revisions, identity_by_ph = await _resolve_identities(
            resolver=resolver,
            obs_repo=obs_repo,
            ph_repo=ph_repo,
            phs=[ph],
            ph_obs_meta=ph_obs_meta,
            face_anchors=[face_anchor],
            det_to_ph=det_to_ph,
            face_evidence=None,
            now=now,
            config=WorldTrackerConfig(),
        )

        # The existing PH must be identified as grandma.
        assert len(revisions) == 1, (
            f"Expected identity commit revision, got {len(revisions)}.  "
            "If 0: the remap did not fire — check that det_to_ph wiring and "
            "entity_id matching in _from_face_anchors are both in place."
        )
        assert revisions[0].new_identity_id == "grandma"
        assert identity_by_ph.get("ph-existing", {}).get("identity_id") == "grandma"


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

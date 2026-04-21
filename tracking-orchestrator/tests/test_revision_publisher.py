"""Tests for RevisionPublisher."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.domain import (
    FaceAnchor,
    GlobalTrack,
    Identity,
    IdentityCandidate,
    IdentityRevision,
)
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig
from app.transport.revision_publisher import RevisionPublisher


def _make_identity(identity_id: str, display_name: str = "") -> Identity:
    return Identity(
        identity_id=identity_id,
        display_name=display_name,
        enrolled_at=datetime.now(UTC),
        is_active=True,
    )


def _make_gt(
    global_track_id: str,
    current_identity_id: str | None = None,
) -> GlobalTrack:
    return GlobalTrack(
        global_track_id=global_track_id,
        camera_ids=["cam-1"],
        tracklet_ids=["t1"],
        current_identity_id=current_identity_id,
        started_at=datetime.now(UTC),
        last_seen_at=datetime.now(UTC),
    )


def _make_revision(
    global_track_id: str = "gt-1",
    identity_id: str = "alice",
) -> IdentityRevision:
    return IdentityRevision(
        revision_id="rev-1",
        global_track_id=global_track_id,
        tracklet_ids=["t1"],
        candidates=[IdentityCandidate(identity_id, "Alice", 0.9)],
        map_identity_id=identity_id,
        posterior_entropy=0.3,
        revision_time=datetime.now(UTC),
    )


class TestRevisionPublisher:
    """Test the RevisionPublisher Redis Streams publisher."""

    @pytest.fixture
    def publisher(self) -> RevisionPublisher:
        return RevisionPublisher(redis_url="redis://localhost:6379/0")

    def test_create_publisher(self, publisher: RevisionPublisher) -> None:
        assert publisher._redis is None
        assert publisher._stream == "tracking.revisions"

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, publisher: RevisionPublisher) -> None:
        """Test connect and disconnect with mocked Redis."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.xgroup_create = AsyncMock()
        mock_redis_module = MagicMock()
        mock_redis_module.from_url = MagicMock(return_value=mock_redis_instance)
        mock_redis_module.ResponseError = Exception
        with patch(
            "app.transport.revision_publisher.redis",
            mock_redis_module,
        ):
            await publisher.connect()
            assert publisher._redis is not None
            await publisher.disconnect()
            assert publisher._redis is None

    @pytest.mark.asyncio
    async def test_publish_many_empty(self, publisher: RevisionPublisher) -> None:
        result = await publisher.publish_many([])
        assert result == []

    @pytest.mark.asyncio
    async def test_publish_many_not_connected(self, publisher: RevisionPublisher) -> None:
        result = await publisher.publish_many([_make_revision()])
        assert result == []

    @pytest.mark.asyncio
    async def test_publish_many_success(self, publisher: RevisionPublisher) -> None:
        """publish_many should call Redis pipeline and return message IDs."""
        mock_redis = AsyncMock()
        mock_redis.pipeline = MagicMock()
        mock_pipe = AsyncMock()
        mock_pipe.xadd = MagicMock()
        mock_pipe.execute = AsyncMock(return_value=["msg-1", "msg-2"])
        mock_redis.pipeline.return_value = mock_pipe
        publisher._redis = mock_redis

        revisions = [
            _make_revision("gt-1", "alice"),
            _make_revision("gt-2", "bob"),
        ]
        result = await publisher.publish_many(revisions)
        assert len(result) == 2
        assert result[0] == "msg-1"
        assert result[1] == "msg-2"
        assert mock_pipe.xadd.call_count == 2

    @pytest.mark.asyncio
    async def test_integration_produces_revisions(self) -> None:
        """Full integration: resolver should produce revisions on identity change."""

        identities = [
            _make_identity("alice", "Alice"),
            _make_identity("bob", "Bob"),
        ]
        resolver = IdentityResolver(
            identities=identities,
            gallery_repo=type(
                "MockGalleryRepo",
                (),
                {
                    "search_similar": AsyncMock(return_value=[]),
                },
            )(),
            global_track_repo=type(
                "MockTrackRepo",
                (),
                {
                    "get": AsyncMock(return_value=None),
                },
            )(),
            tracking_repo=type("MockTrackingRepo", (), {})(),
            config=ResolverConfig(
                commit_prob=0.5,
                prior_weight=0.3,
            ),
        )

        # First: assign alice
        gt = _make_gt("gt-1", current_identity_id=None)
        face_anchor = FaceAnchor(
            person_id="alice",
            confidence=0.9,
            quality=0.8,
            tracklet_id="t1",
        )

        outcome1 = await resolver.resolve(
            global_tracks=[gt],
            new_face_anchors=[face_anchor],
            captured_at=datetime.now(UTC),
        )
        assert outcome1.decisions[0].identity_id == "alice"
        assert len(outcome1.revisions) == 1
        assert outcome1.revisions[0].new_identity_id == "alice"

        # Simulate pipeline applying the decision to the GT
        gt = _make_gt("gt-1", current_identity_id="alice")

        # Second: assign bob -> revision
        face_anchor2 = FaceAnchor(
            person_id="bob",
            confidence=0.9,
            quality=0.8,
            tracklet_id="t1",
        )
        outcome2 = await resolver.resolve(
            global_tracks=[gt],
            new_face_anchors=[face_anchor2],
            captured_at=datetime.now(UTC),
        )
        assert outcome2.decisions[0].identity_id == "bob"
        assert outcome2.decisions[0].revises_previous is True
        assert len(outcome2.revisions) == 1
        rev = outcome2.revisions[0]
        assert rev.previous_identity_id == "alice"
        assert rev.new_identity_id == "bob"

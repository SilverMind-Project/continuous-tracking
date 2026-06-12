"""tests for configurable stream names wired through TransportConfig.

Option A: revisions_stream, signals_stream, scene_samples_stream are
plumbed through TransportConfig and used by the corresponding publishers.
"""

from __future__ import annotations

from app.config import Settings
from app.transport.redis_streams import TransportConfig


def _build_config(env: dict[str, str] | None = None) -> TransportConfig:
    settings = Settings.from_dict(
        {
            "redis": {
                "url": "redis://localhost:6379/0",
                "consumer_group": "test-group",
                "consumer_name": "test-consumer",
                "frames_stream": "test.frames",
                "events_stream": "test.events",
                "revisions_stream": "test.revisions",
                "signals_stream": "test.signals",
                "scene_samples_stream": "test.scene",
                "batch_max_wait_ms": "100",
                "batch_max_size": "8",
                "xack_timeout_ms": "5000",
                "ack_ttl_seconds": "300",
            },
        }
    )
    s = settings
    return TransportConfig(
        redis_url=s.as_str("redis.url"),
        consumer_group=s.as_str("redis.consumer_group"),
        consumer_name=s.as_str("redis.consumer_name"),
        frames_stream=s.as_str("redis.frames_stream"),
        events_stream=s.as_str("redis.events_stream"),
        batch_max_wait_ms=s.as_int("redis.batch_max_wait_ms"),
        batch_max_size=s.as_int("redis.batch_max_size"),
        xack_timeout_ms=s.as_int("redis.xack_timeout_ms"),
        ack_ttl_seconds=s.as_int("redis.ack_ttl_seconds"),
        revisions_stream=s.as_str("redis.revisions_stream"),
        signals_stream=s.as_str("redis.signals_stream"),
        scene_samples_stream=s.as_str("redis.scene_samples_stream"),
    )


class TestStreamNamesFromConfig:
    def test_revisions_stream_configured(self) -> None:
        cfg = _build_config()
        assert cfg.revisions_stream == "test.revisions"

    def test_signals_stream_configured(self) -> None:
        cfg = _build_config()
        assert cfg.signals_stream == "test.signals"

    def test_scene_samples_stream_configured(self) -> None:
        cfg = _build_config()
        assert cfg.scene_samples_stream == "test.scene"

    def test_stream_defaults(self) -> None:
        """Defaults match the class-level _stream_name on each publisher."""
        cfg = TransportConfig()
        assert cfg.revisions_stream == "tracking.revisions"
        assert cfg.signals_stream == "tracking.signals"
        assert cfg.scene_samples_stream == "scene.samples"


class TestRevisionPublisherUsesConfiguredStream:
    def test_publisher_stream_is_configured_value(self) -> None:
        """RevisionPublisher uses the stream from TransportConfig."""
        from app.transport.revision_publisher import RevisionPublisher

        pub = RevisionPublisher(
            redis_url="redis://localhost:6379/0",
            stream="custom.revisions",
        )
        assert pub._stream == "custom.revisions"


class TestSignalPublisherUsesConfiguredStream:
    def test_publisher_stream_is_configured_value(self) -> None:
        from app.transport.signal_publisher import SignalPublisher

        pub = SignalPublisher(
            redis_url="redis://localhost:6379/0",
            stream="custom.signals",
        )
        assert pub._stream == "custom.signals"


class TestSceneSamplesPublisherUsesConfiguredStream:
    def test_publisher_stream_is_configured_value(self) -> None:
        from app.transport.scene_publisher import SceneSamplesPublisher

        pub = SceneSamplesPublisher(
            redis_url="redis://localhost:6379/0",
            stream="custom.scene",
        )
        assert pub._stream == "custom.scene"

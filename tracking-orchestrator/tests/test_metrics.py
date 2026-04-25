"""Smoke tests for the M10 Prometheus metrics module.

The tests build a fresh ``CollectorRegistry`` per case so the global
registry stays clean.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry

from app.observability.metrics import build_metrics


def test_build_metrics_registers_required_series() -> None:
    registry = CollectorRegistry()
    m = build_metrics(registry=registry)
    # Inc one of every counter so the series materialise in the registry.
    m.frames_consumed_total.labels(camera_id="cam-1").inc()
    m.frames_failed_total.labels(camera_id="cam-1", reason="timeout").inc()
    m.tracking_events_published_total.labels(camera_id="cam-1").inc()
    m.tracking_revisions_published_total.labels(reason="initial_assignment").inc()
    m.dementia_signals_published_total.labels(signal_kind="pacing", severity="warning").inc()
    m.scene_samples_published_total.labels(reason="periodic").inc()
    m.proto_messages_emitted_total.labels(stream="tracking.events").inc()
    m.identity_commits_total.labels(source="face").inc()
    m.identity_revisions_total.labels(reason="initial_assignment").inc()
    m.posterior_entropy.observe(0.42)
    m.frame_end_to_end_latency_ms.labels(camera_id="cam-1").observe(120.0)
    m.triton_inference_latency_ms.labels(model="yolo").observe(8.0)
    m.tracklets_active.labels(camera_id="cam-1").set(3)
    m.global_tracks_active.set(2)
    m.gallery_size.set(150)

    metric_names = {metric.name for metric in registry.collect()}
    expected = {
        "cts_frames_consumed",
        "cts_frames_failed",
        "cts_tracking_events_published",
        "cts_tracking_revisions_published",
        "cts_dementia_signals_published",
        "cts_scene_samples_published",
        "cts_proto_messages_emitted",
        "cts_identity_commits",
        "cts_identity_revisions",
        "cts_posterior_entropy_bits",
        "cts_frame_end_to_end_latency_ms",
        "cts_triton_inference_latency_ms",
        "cts_tracklets_active",
        "cts_global_tracks_active",
        "cts_gallery_size",
    }
    missing = expected - metric_names
    assert not missing, f"missing metric series: {missing}"


def test_publish_event_increments_counter(monkeypatch) -> None:
    """Calling RedisStreamsTransport.publish_event bumps the published counter."""
    import asyncio
    from datetime import UTC, datetime
    from unittest.mock import AsyncMock

    from prometheus_client import CollectorRegistry

    from app.observability import metrics as metrics_pkg
    from app.observability.metrics import build_metrics
    from app.transport.redis_streams import RedisStreamsTransport, TransportConfig

    fresh = build_metrics(registry=CollectorRegistry())
    monkeypatch.setattr(metrics_pkg, "metrics", fresh)
    # Patch the metrics handle the transport reads from at import time.
    from app.transport import redis_streams

    monkeypatch.setattr(redis_streams.metrics, "metrics", fresh)

    transport = RedisStreamsTransport(TransportConfig())
    mock_redis = AsyncMock()
    mock_redis.xadd = AsyncMock(return_value="mid-1")
    transport._redis = mock_redis

    async def _go() -> None:
        await transport.publish_event(
            camera_id="cam-1",
            event_time=datetime.now(UTC),
            frame_index=1,
            detection_count=0,
        )

    asyncio.run(_go())

    sample = next(
        s
        for metric in fresh.tracking_events_published_total.collect()
        for s in metric.samples
        if s.labels.get("camera_id") == "cam-1" and s.name.endswith("_total")
    )
    assert sample.value == 1.0

"""Tests for frame acknowledgement on every completion path and phase latency."""

from __future__ import annotations

import time
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest
from prometheus_client import CollectorRegistry

from app.inference.triton_client import TritonGrpcClient
from app.observability import metrics as _metrics_module
from app.pipeline.frame_context import FrameContext
from app.pipeline.frame_pipeline import FrameProcessingPipeline, PipelineConfig
from app.pipeline.stages.base import StageRunner, timed_phase
from app.transport.redis_streams import FrameReady


def _now_ns() -> int:
    return time.time_ns()


def _make_frame(
    camera_id: str = "cam-1",
    frame_index: int = 1,
    capture_time_unix_ns: int | None = None,
) -> FrameReady:
    cap_ns = _now_ns() - 50_000_000 if capture_time_unix_ns is None else capture_time_unix_ns
    return FrameReady(
        camera_id=camera_id,
        minio_key=f"frames/{camera_id}/{frame_index}.jpg",
        frame_index=frame_index,
        capture_time_unix_ns=cap_ns,
        received_time_unix_ns=cap_ns + 10_000_000,
        width=640,
        height=480,
    )


class _StubStage:
    def __init__(self, name: str, *, fail: bool = False) -> None:
        self.name = name
        self._fail = fail

    async def run(self, ctx: FrameContext) -> None:
        if self._fail:
            raise RuntimeError(f"{self.name} failed")

    async def run_batch(self, contexts: list[FrameContext]) -> None:
        if self._fail:
            raise RuntimeError(f"{self.name} batch failed")

    async def run_many(self, contexts: list[FrameContext]) -> None:
        if self._fail:
            raise RuntimeError(f"{self.name} many failed")


def _sample_value(metric: Any, suffix: str, labels: dict[str, str]) -> float:
    for fam in metric.collect():
        for sample in fam.samples:
            if sample.name.endswith(suffix) and all(
                sample.labels.get(k) == v for k, v in labels.items()
            ):
                return float(sample.value)
    return 0.0


class TestFrameAckAndPhaseLatency:
    """Every consumed frame is acknowledged exactly once across all pipeline paths."""

    @pytest.fixture(autouse=True)
    def isolated_metrics(self) -> Generator[None, None, None]:
        registry = CollectorRegistry()
        original = _metrics_module.metrics
        _metrics_module.metrics = _metrics_module.build_metrics(registry=registry)
        yield
        _metrics_module.metrics = original

    def _build_cross_camera_pipeline(
        self,
        *,
        fail_fetch: bool = False,
        fail_detect: bool = False,
        fail_pre_world: bool = False,
        fail_world: bool = False,
        fail_post_world: bool = False,
    ) -> tuple[FrameProcessingPipeline, AsyncMock]:
        pipeline = FrameProcessingPipeline(PipelineConfig(signals_enabled=False))
        transport = AsyncMock()
        pipeline._transport = transport
        pipeline._fetch_stage = _StubStage("fetch", fail=fail_fetch)  # type: ignore[assignment]
        pipeline._detect_stage = _StubStage("detect", fail=fail_detect)  # type: ignore[assignment]
        pipeline._pre_world_runner = StageRunner(
            [_StubStage("inference", fail=fail_pre_world)]  # type: ignore[list-item]
        )
        pipeline._world_tracking_stage = _StubStage(  # type: ignore[assignment]
            "world_tracking", fail=fail_world
        )
        pipeline._post_world_runner = StageRunner(
            [_StubStage("publish", fail=fail_post_world)]  # type: ignore[list-item]
        )
        return pipeline, transport

    @pytest.mark.asyncio
    async def test_fetch_failure_acks_frame_and_increments_failed(self) -> None:
        pipeline, transport = self._build_cross_camera_pipeline(fail_fetch=True)
        frame = _make_frame("cam-1", 10)
        await pipeline._handle_cross_camera_batch([frame])

        transport.ack_frame.assert_awaited_once_with(frame)
        assert (
            _sample_value(
                _metrics_module.metrics.frames_failed_total,
                "_total",
                {"camera_id": "cam-1", "reason": "fetch_error"},
            )
            == 1.0
        )

    @pytest.mark.asyncio
    async def test_detector_failure_acks_all_fetched_frames(self) -> None:
        pipeline, transport = self._build_cross_camera_pipeline(fail_detect=True)
        f1 = _make_frame("cam-1", 1)
        f2 = _make_frame("cam-2", 1)
        await pipeline._handle_cross_camera_batch([f1, f2])

        assert transport.ack_frame.await_count == 2
        assert (
            _sample_value(
                _metrics_module.metrics.frames_failed_total,
                "_total",
                {"camera_id": "cam-1", "reason": "detector_error"},
            )
            == 1.0
        )
        assert (
            _sample_value(
                _metrics_module.metrics.frames_failed_total,
                "_total",
                {"camera_id": "cam-2", "reason": "detector_error"},
            )
            == 1.0
        )

    @pytest.mark.asyncio
    async def test_pre_world_failure_acks_frame(self) -> None:
        pipeline, transport = self._build_cross_camera_pipeline(fail_pre_world=True)
        frame = _make_frame("cam-1", 2)
        await pipeline._handle_cross_camera_batch([frame])

        transport.ack_frame.assert_awaited_once_with(frame)
        assert (
            _sample_value(
                _metrics_module.metrics.frames_failed_total,
                "_total",
                {"camera_id": "cam-1", "reason": "processing_error"},
            )
            == 1.0
        )

    @pytest.mark.asyncio
    async def test_world_tracking_failure_acks_frame(self) -> None:
        pipeline, transport = self._build_cross_camera_pipeline(fail_world=True)
        frame = _make_frame("cam-1", 3)
        await pipeline._handle_cross_camera_batch([frame])

        transport.ack_frame.assert_awaited_once_with(frame)
        assert (
            _sample_value(
                _metrics_module.metrics.frames_failed_total,
                "_total",
                {"camera_id": "cam-1", "reason": "processing_error"},
            )
            == 1.0
        )

    @pytest.mark.asyncio
    async def test_post_world_failure_acks_frame(self) -> None:
        pipeline, transport = self._build_cross_camera_pipeline(fail_post_world=True)
        frame = _make_frame("cam-1", 4)
        await pipeline._handle_cross_camera_batch([frame])

        transport.ack_frame.assert_awaited_once_with(frame)
        assert (
            _sample_value(
                _metrics_module.metrics.frames_failed_total,
                "_total",
                {"camera_id": "cam-1", "reason": "processing_error"},
            )
            == 1.0
        )

    @pytest.mark.asyncio
    async def test_stale_frame_acks_once_and_increments_stale_counter(self) -> None:
        pipeline, transport = self._build_cross_camera_pipeline()
        stale_ns = _now_ns() - int(120 * 1e9)
        frame = _make_frame("cam-1", 5, capture_time_unix_ns=stale_ns)
        await pipeline._handle_cross_camera_batch([frame])

        transport.ack_frame.assert_awaited_once_with(frame)
        assert (
            _sample_value(
                _metrics_module.metrics.frames_dropped_stale_total,
                "_total",
                {"camera_id": "cam-1"},
            )
            == 1.0
        )

    @pytest.mark.asyncio
    async def test_handle_batch_and_post_detect_batch_ack_on_failure(self) -> None:
        pipeline = FrameProcessingPipeline(PipelineConfig(signals_enabled=False))
        transport = AsyncMock()
        pipeline._transport = transport
        pipeline._post_detect_runner = StageRunner(
            [_StubStage("post_detect", fail=True)]  # type: ignore[list-item]
        )

        f1 = _make_frame("cam-1", 6)
        ctx1 = pipeline._init_context(f1)
        await pipeline._process_post_detect_batch([ctx1])
        transport.ack_frame.assert_awaited_once_with(f1)

        transport.ack_frame.reset_mock()

        async def _boom(frame: FrameReady) -> None:
            raise RuntimeError("batch failure")

        pipeline._process_frame = _boom  # type: ignore[assignment, method-assign]
        f2 = _make_frame("cam-1", 7)
        await pipeline._handle_batch("cam-1", [f2])
        transport.ack_frame.assert_awaited_once_with(f2)

    @pytest.mark.asyncio
    async def test_cross_camera_batch_observes_stage_and_e2e_latencies(self) -> None:
        pipeline, transport = self._build_cross_camera_pipeline()
        frame = _make_frame("cam-1", 8)
        await pipeline._handle_cross_camera_batch([frame])

        transport.ack_frame.assert_awaited_once_with(frame)
        stage_hist = _metrics_module.metrics.stage_latency_ms
        assert _sample_value(stage_hist, "_count", {"stage": "fetch", "camera_id": "cam-1"}) == 1.0
        assert _sample_value(stage_hist, "_count", {"stage": "detect", "camera_id": "batch"}) == 1.0
        assert (
            _sample_value(stage_hist, "_count", {"stage": "world_tracking", "camera_id": "batch"})
            == 1.0
        )
        assert (
            _sample_value(stage_hist, "_count", {"stage": "publish", "camera_id": "cam-1"}) == 1.0
        )

        e2e_hist = _metrics_module.metrics.frame_end_to_end_latency_ms
        assert _sample_value(e2e_hist, "_count", {"camera_id": "cam-1"}) == 1.0
        assert _sample_value(e2e_hist, "_sum", {"camera_id": "cam-1"}) > 0.0

    @pytest.mark.asyncio
    async def test_triton_client_observes_inference_latency(self, monkeypatch) -> None:
        from triton_shared.client import TritonGrpcClient as BaseClient

        async def _fake_infer(self, model_name: str, inputs: Any, output_names: list[str]) -> Any:
            return {name: np.zeros((1, 4), dtype=np.float32) for name in output_names}

        monkeypatch.setattr(BaseClient, "infer", _fake_infer)
        client = TritonGrpcClient("localhost:8001")
        out = await client.infer(
            "person-detector",
            [("images", np.zeros((1, 3, 8, 8), dtype=np.float32))],
            ["output0"],
        )
        assert "output0" in out
        triton_hist = _metrics_module.metrics.triton_inference_latency_ms
        assert _sample_value(triton_hist, "_count", {"model": "person-detector"}) == 1.0

    @pytest.mark.asyncio
    async def test_sample_stream_backlog_updates_pending_and_lag_gauges(self) -> None:
        pipeline = FrameProcessingPipeline(PipelineConfig(signals_enabled=False))
        transport = AsyncMock()
        transport.pending_count = AsyncMock(return_value=(17, 42))
        pipeline._transport = transport

        await pipeline._sample_stream_backlog_once()

        assert _metrics_module.metrics.frames_stream_pending._value.get() == 17.0
        assert _metrics_module.metrics.frames_stream_lag._value.get() == 42.0

    def test_timed_phase_records_on_exception(self) -> None:
        with pytest.raises(ValueError, match="boom"), timed_phase("custom_stage", "cam-9"):
            raise ValueError("boom")
        stage_hist = _metrics_module.metrics.stage_latency_ms
        assert (
            _sample_value(stage_hist, "_count", {"stage": "custom_stage", "camera_id": "cam-9"})
            == 1.0
        )

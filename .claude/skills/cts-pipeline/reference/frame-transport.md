# Frame Transport and Stage Timing

## One acknowledgement per consumed frame

Every `FrameReady` message read from `frames.ready` is processed at most once and always acknowledged before `FrameProcessingPipeline` releases it:

- `RedisStreamsTransport` tracks in-flight stream message IDs in `_pending_acks` keyed by `frame_ack_key(frame)` (`(frame.camera_id, int(frame.frame_index), int(frame.capture_time_unix_ns))`), never `id(frame)`.
- `FrameProcessingPipeline._finish_frame(frame, *, failed_reason: str | None = None)` is the single terminal helper for every execution route (direct, per-camera batch, and cross-camera batch). When `failed_reason` is set (`fetch_error`, `detector_error`, or `processing_error`), it increments `cts_frames_failed_total{camera_id, reason}`; when `failed_reason is None`, it observes `cts_frame_end_to_end_latency_ms{camera_id}` from `frame.capture_time`; in both cases it awaits `self._transport.ack_frame(frame)`.
- Every exit path in `_process_frame_direct`, `_handle_batch`, `_handle_cross_camera_batch`, `_process_cross_camera_post_detect_batch`, `_process_post_world_context`, and `_process_post_detect_batch` must reach `_finish_frame` (or `ack_frame` on staleness drop before stage work). Never leave a failed frame unacknowledged in the consumer group PEL.
- On `connect()`, `RedisStreamsTransport` runs `XAUTOCLAIM` in a loop for PEL entries idle longer than `redis.pending_reclaim_min_idle_s` (default 300 s), acknowledges them without reprocessing, and increments `cts_frames_abandoned_total`.
- `FrameProcessingPipeline` samples `RedisStreamsTransport.pending_count()` (`XINFO GROUPS` `pending` and `lag`) every 15 s into `cts_frames_stream_pending` and `cts_frames_stream_lag`.

## Timed phase helper

Every pipeline stage invocation is wrapped in `timed_phase(stage_name, camera_id)` (`app/pipeline/stages/base.py`), which measures wall-clock duration via `time.perf_counter()` and records `cts_pipeline_latency_ms{camera_id, phase}` inside a `finally` block:

- `StageRunner.run` wraps each `await stage.run(ctx)` in `with timed_phase(stage.name, ctx.camera_id):`.
- Direct stage calls outside `StageRunner` in `frame_pipeline.py` (`_fetch_stage.run(ctx)`, `_detect_stage.run_batch(batch_ctxs)` with `camera_id="batch"`, and `_world_tracking_stage.run_many(round_ctxs)` with `camera_id="batch"`) also wrap their call in `timed_phase`.
- `TritonGrpcClient.infer` (`app/inference/triton_client.py`) records per-RPC latency in `cts_triton_inference_latency_ms{model}`.

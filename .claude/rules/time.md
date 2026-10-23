---
paths: ["**/tracking-orchestrator/**/*.py", "**/rtsp-ingress/**/*.go"]
---

# Time

- Timestamps are timezone-aware UTC: `datetime.now(UTC)` (`from datetime import UTC`), Go
  `time.Now().UTC()`. A naive datetime breaks TimescaleDB `timestamptz` columns.
- Pipeline code uses the domain time: `ctx.capture_time` for the physical observation,
  `ctx.event_time` for processing; `datetime.now(UTC)` only for true wall-clock side effects
  (for example `emitted_at`).
- Stillness, dwell, and other durations accumulate elapsed seconds from timestamps, never frame
  counts. Async code never calls `time.sleep()`.

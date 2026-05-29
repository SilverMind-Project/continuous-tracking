"""Record a FrameReady replay fixture from a live dev stack.

Subscribes to ``frames.ready`` Redis stream and captures FrameReady
protobuf messages from the specified cameras for the requested duration.

Usage::

    uv run python -m scripts.record_replay_fixture \\
        --camera-id cam_a --camera-id cam_b \\
        --duration-seconds 30 \\
        --output tests/fixtures/frame_replays/two_cameras_one_room.bin
"""

from __future__ import annotations

import argparse
import asyncio
import os
import struct
import sys
import time

from google.protobuf.message import DecodeError
from structlog import get_logger

logger = get_logger(__name__)


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Record a replay fixture from frames.ready")
    parser.add_argument(
        "--camera-id",
        action="append",
        dest="camera_ids",
        required=True,
        help="Camera ID to capture (repeatable)",
    )
    parser.add_argument(
        "--duration-seconds", type=int, default=30, help="Recording duration in seconds"
    )
    parser.add_argument("--output", required=True, help="Output .bin file path")
    parser.add_argument(
        "--redis-url",
        default=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        help="Redis URL (default: $REDIS_URL or redis://localhost:6379/0)",
    )
    parser.add_argument(
        "--stream", default="frames.ready", help="Redis stream name (default: frames.ready)"
    )
    args = parser.parse_args()

    camera_ids = set(args.camera_ids)
    output_path: str = args.output
    duration: int = args.duration_seconds
    redis_url: str = args.redis_url
    stream: str = args.stream

    logger.info(
        "capture_started",
        output=output_path,
        duration_seconds=duration,
        cameras=list(camera_ids),
        redis=redis_url,
        stream=stream,
    )

    try:
        import redis.asyncio as aioredis
        from redis.exceptions import RedisError
    except ImportError:
        logger.error("redis_not_installed", hint="uv sync --extra dev")
        sys.exit(1)

    client = aioredis.from_url(redis_url, decode_responses=False)
    frame_count = 0
    byte_count = 0
    start_time = time.monotonic()

    with open(output_path, "wb") as out:
        # Subscribe to the stream from the latest message.
        last_id = "$"
        while (time.monotonic() - start_time) < duration:
            try:
                messages = await client.xread({stream: last_id}, count=10, block=1000)
            except RedisError as exc:
                logger.warning("redis_read_error", error=str(exc))
                await asyncio.sleep(1)
                continue

            for _stream_name, entries in messages:
                for msg_id, fields in entries:
                    last_id = msg_id.decode() if isinstance(msg_id, bytes) else msg_id

                    # The frame field in the stream message is keyed "frame".
                    frame_bytes = None
                    for k, v in fields.items():
                        key = k.decode() if isinstance(k, bytes) else k
                        if key == "frame":
                            frame_bytes = v
                            break

                    if frame_bytes is None:
                        continue

                    # Parse the FrameReady proto to check camera_id.
                    try:
                        from app.proto.continuoustracking.v1 import frame_pb2

                        frame_msg = frame_pb2.FrameReady.FromString(frame_bytes)
                    except DecodeError:
                        # Not a valid FrameReady protobuf; skip.
                        continue

                    if frame_msg.camera_id not in camera_ids:
                        continue

                    # Write length-prefixed frame.
                    msg_len = len(frame_bytes)
                    out.write(struct.pack(">I", msg_len))
                    out.write(frame_bytes)
                    frame_count += 1
                    byte_count += 4 + msg_len

            elapsed = time.monotonic() - start_time
            if frame_count > 0 and frame_count % 100 == 0:
                logger.info(
                    "capture_progress",
                    frames=frame_count,
                    kb=int(byte_count / 1024),
                    elapsed_s=int(elapsed),
                )

    await client.aclose()
    elapsed = time.monotonic() - start_time
    logger.info(
        "capture_complete",
        output=output_path,
        total_frames=frame_count,
        total_kb=int(byte_count / 1024),
        elapsed_seconds=round(elapsed, 1),
    )


if __name__ == "__main__":
    asyncio.run(_main())

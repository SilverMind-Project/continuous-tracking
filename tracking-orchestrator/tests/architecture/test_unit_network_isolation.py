"""Guardrail canary: non-integration tests cannot open IP network sockets.

Local developer machines often run Redis/Postgres/MQTT on localhost, which lets
tests with unmocked clients pass locally and fail in CI where no service
containers run during unit tests. The autouse fixture in ``tests/conftest.py``
blocks IP socket connections for all non-integration tests.
"""

from __future__ import annotations

import asyncio
import socket

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.pipeline.frame_pipeline import FrameProcessingPipeline, PipelineConfig


def test_sync_inet_socket_connect_is_blocked() -> None:
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock,
        pytest.raises(RuntimeError, match="outbound network connection"),
    ):
        sock.connect(("127.0.0.1", 6379))


@pytest.mark.asyncio
async def test_async_inet_connection_is_blocked() -> None:
    with pytest.raises(RuntimeError, match="outbound async connection"):
        await asyncio.open_connection("127.0.0.1", 6379)


@pytest.mark.asyncio
async def test_unmocked_pipeline_initialize_fails_loudly_without_redis() -> None:
    pipeline = FrameProcessingPipeline(PipelineConfig(allow_skeleton=True))
    with pytest.raises(RedisConnectionError, match="outbound async connection"):
        await pipeline.initialize()

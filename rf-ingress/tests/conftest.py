"""Shared pytest helpers for rf-ingress unit, conformance, and integration tests."""

from __future__ import annotations

import asyncio
import socket
from pathlib import Path
from typing import Any

import pytest

SCENARIOS_DIR = Path(__file__).resolve().parents[1] / "config" / "scenarios"


@pytest.fixture(autouse=True)
def _block_network_in_unit_tests(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent non-integration tests from opening IP sockets.

    Local dev machines often run Redis/MQTT on localhost, which lets unmocked
    clients pass locally and fail in CI where no service containers run during
    unit tests. Tests that need real sockets must carry
    ``@pytest.mark.integration``.
    """
    if request.node.get_closest_marker("integration") is not None:
        return

    orig_connect = socket.socket.connect
    orig_connect_ex = socket.socket.connect_ex
    orig_create_connection = asyncio.BaseEventLoop.create_connection

    def _guarded_connect(self: socket.socket, address: Any) -> Any:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError(
                f"Unit test attempted outbound network connection to {address!r}. "
                "Mock the client or mark the test @pytest.mark.integration."
            )
        return orig_connect(self, address)

    def _guarded_connect_ex(self: socket.socket, address: Any) -> int:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError(
                f"Unit test attempted outbound network connection to {address!r}. "
                "Mock the client or mark the test @pytest.mark.integration."
            )
        return orig_connect_ex(self, address)

    async def _guarded_create_connection(
        self: asyncio.BaseEventLoop, *args: Any, **kwargs: Any
    ) -> Any:
        sock = kwargs.get("sock")
        if sock is None or sock.family in (socket.AF_INET, socket.AF_INET6):
            host = args[1] if len(args) > 1 else kwargs.get("host")
            port = args[2] if len(args) > 2 else kwargs.get("port")
            raise RuntimeError(
                f"Unit test attempted outbound async connection to ({host!r}, {port!r}). "
                "Mock the client or mark the test @pytest.mark.integration."
            )
        return await orig_create_connection(self, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _guarded_connect_ex)
    monkeypatch.setattr(asyncio.BaseEventLoop, "create_connection", _guarded_create_connection)

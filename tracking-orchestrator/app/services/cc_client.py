"""Shared HTTP client for cognitive-companion API calls.

Centralises auth (X-API-Key header), timeout, and URL construction so
every orchestrator→CC call uses the same pattern.
"""

from __future__ import annotations

from typing import Any

from structlog import get_logger

logger = get_logger(__name__)


class CognitiveCompanionClient:
    """Thin HTTP client for cognitive-companion.

    Wraps httpx with the shared auth header and timeout so callers don't
    duplicate the boilerplate.
    """

    def __init__(self, base_url: str, api_key: str = "") -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    async def get(self, path: str) -> Any:
        """Authenticated GET to *path* (e.g. ``/api/v1/cts/cameras``).

        Raises on HTTP or network error — callers should catch and degrade
        gracefully.
        """
        import httpx

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["X-API-Key"] = self._api_key

        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            resp = await client.get(f"{self._base_url}{path}", headers=headers)
            resp.raise_for_status()
            return resp.json()

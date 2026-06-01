"""Unit tests for CachedCameraTopologyRepository write-through cache.

These tests previously covered WorldTracker._load_topology_edges and
_upsert_topology_edge.  Both methods were removed; the same caching contract
is now satisfied by CachedCameraTopologyRepository (storage/base.py).
"""

from __future__ import annotations

import pytest

from app.domain import CameraTopologyEdge
from app.storage.base import CachedCameraTopologyRepository, InMemoryCameraTopologyRepository


def _make_cached() -> tuple[CachedCameraTopologyRepository, InMemoryCameraTopologyRepository]:
    raw = InMemoryCameraTopologyRepository()
    cached = CachedCameraTopologyRepository(raw)
    return cached, raw


def _edge(from_cam: str, to_cam: str, count: int = 1) -> CameraTopologyEdge:
    return CameraTopologyEdge(
        from_camera=from_cam,
        to_camera=to_cam,
        observation_count=count,
        mean_transit_s=5.0,
        variance_transit_s2=1.0,
    )


@pytest.mark.asyncio
async def test_cache_cold_loads_from_db_on_first_call() -> None:
    """Cold cache fetches from the underlying repo on the first list_edges call."""
    cached, raw = _make_cached()
    await raw.upsert_edge(_edge("cam-a", "cam-b"))

    edges = await cached.list_edges()
    assert len(edges) == 1
    assert edges[0].from_camera == "cam-a"


@pytest.mark.asyncio
async def test_cache_returns_cached_value_on_second_call() -> None:
    """Subsequent list_edges calls skip the DB and return the cached result."""
    cached, raw = _make_cached()
    await raw.upsert_edge(_edge("cam-a", "cam-b"))

    first = await cached.list_edges()
    # Add a second edge directly to the raw repo, bypassing the cache.
    await raw.upsert_edge(_edge("cam-b", "cam-c"))

    second = await cached.list_edges()
    # Cache must not see the bypass write — still 1 edge.
    assert len(second) == 1
    assert second == first


@pytest.mark.asyncio
async def test_upsert_updates_cache_in_place() -> None:
    """upsert_edge writes to the delegate and immediately updates the cache."""
    cached, raw = _make_cached()
    await cached.list_edges()  # prime cache (empty)

    edge = _edge("cam-a", "cam-b")
    await cached.upsert_edge(edge)

    # Cache must reflect the new edge without a fresh DB round-trip.
    edges = await cached.list_edges()
    assert any(e.from_camera == "cam-a" and e.to_camera == "cam-b" for e in edges)

    # Underlying DB must also contain it.
    assert await raw.get_edge("cam-a", "cam-b") is not None


@pytest.mark.asyncio
async def test_upsert_replaces_existing_edge_not_appends() -> None:
    """Upserting an existing edge replaces it; no duplicates accumulate."""
    cached, _raw = _make_cached()
    await cached.upsert_edge(_edge("cam-a", "cam-b", count=1))
    await cached.upsert_edge(_edge("cam-a", "cam-b", count=5))

    edges = await cached.list_edges()
    matching = [e for e in edges if e.from_camera == "cam-a" and e.to_camera == "cam-b"]
    assert len(matching) == 1, "Duplicate edges must not accumulate in cache"
    assert matching[0].observation_count == 5

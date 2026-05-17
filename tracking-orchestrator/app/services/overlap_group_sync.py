"""Fetches camera overlap groups from cognitive-companion at startup.

Overlap groups define cameras that share a physical field of view.  The
cross-camera associator uses this information to fuse tracklets across
group cameras aggressively (no transit-time budget, relaxed appearance
threshold) and to share face-anchor evidence within the same GlobalTrack.
"""

from __future__ import annotations

from typing import Any

from structlog import get_logger

from ..domain import OverlapGroup
from .cc_client import CognitiveCompanionClient

logger = get_logger(__name__)


async def fetch_overlap_groups(client: CognitiveCompanionClient) -> list[OverlapGroup]:
    """Fetch camera overlap groups from CC.

    Returns [] on error so the pipeline degrades gracefully when CC is
    unavailable at startup.
    """
    try:
        raw: list[dict[str, Any]] = await client.get("/api/v1/cts/overlap_groups")
    except Exception:
        logger.warning("Failed to fetch overlap groups from CC", exc_info=True)
        return []

    groups: list[OverlapGroup] = []
    skipped = 0
    for r in raw:
        gid = r.get("id")
        cam_ids = r.get("camera_ids")
        if not isinstance(gid, str) or not gid or not isinstance(cam_ids, list) or len(cam_ids) < 2:
            skipped += 1
            continue
        groups.append(
            OverlapGroup(
                group_id=gid,
                name=str(r.get("name", "")),
                camera_ids=tuple(str(c) for c in cam_ids),
            )
        )
    if skipped:
        logger.warning("Skipped malformed overlap groups", count=skipped)
    logger.info("Fetched camera overlap groups from CC", group_count=len(groups))
    return groups


async def fetch_adjacency_edges(client: CognitiveCompanionClient) -> list[dict[str, Any]]:
    """Fetch persisted camera adjacency edges from CC.

    Returns [] on error so the pipeline degrades gracefully when CC is
    unavailable at startup.  Each dict has keys ``from``, ``to``,
    ``min_transit_s``, ``max_transit_s``, ``overlap`` matching the
    orchestrator's AdjacencyEdgeIn wire format.
    """
    try:
        data: dict[str, Any] = await client.get("/api/v1/cts/calibration/adjacency")
    except Exception:
        logger.warning("Failed to fetch adjacency edges from CC", exc_info=True)
        return []

    edges: list[dict[str, Any]] = data.get("edges", [])
    logger.info("Fetched camera adjacency edges from CC", edge_count=len(edges))
    return edges

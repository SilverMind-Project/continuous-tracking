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

    Raises when CC is unavailable so startup does not silently run with an
    empty cross-camera topology.
    """
    try:
        raw: list[dict[str, Any]] = await client.get("/api/v1/cts/overlap_groups")
    except Exception as exc:
        logger.exception("overlap_groups_fetch_failed")
        raise RuntimeError("Failed to fetch overlap groups from CC") from exc

    if not isinstance(raw, list):
        raise TypeError("CC overlap_groups response must be a list")

    groups: list[OverlapGroup] = []
    skipped = 0
    for r in raw:
        gid = r.get("id")
        cam_ids = r.get("camera_ids")
        if (
            not isinstance(gid, (str, int))
            or not gid
            or not isinstance(cam_ids, list)
            or len(cam_ids) < 2
        ):
            skipped += 1
            continue
        gid = str(gid)
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

    Raises when CC is unavailable so startup does not silently run with an
    empty camera-transition graph. Each dict has keys ``from``, ``to``,
    ``min_transit_s``, ``max_transit_s``, ``overlap`` matching the
    orchestrator's AdjacencyEdgeIn wire format.
    """
    try:
        data: dict[str, Any] = await client.get("/api/v1/cts/calibration/adjacency")
    except Exception as exc:
        logger.exception("adjacency_edges_fetch_failed")
        raise RuntimeError("Failed to fetch adjacency edges from CC") from exc

    edges = data["edges"]
    if not isinstance(edges, list):
        raise TypeError("CC adjacency response field edges must be a list")
    logger.info("Fetched camera adjacency edges from CC", edge_count=len(edges))
    return edges

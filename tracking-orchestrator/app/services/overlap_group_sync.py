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

logger = get_logger(__name__)


async def fetch_overlap_groups(cc_url: str, api_key: str = "") -> list[OverlapGroup]:
    """Fetch camera overlap groups from CC.

    Returns [] on error so the pipeline degrades gracefully when CC is
    unavailable at startup.
    """
    if not cc_url:
        return []
    try:
        import httpx
    except ImportError:
        logger.error("httpx not installed; cannot fetch overlap groups from CC")
        return []

    url = cc_url.rstrip("/") + "/api/v1/cts/overlap_groups"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            raw: list[dict[str, Any]] = resp.json()
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

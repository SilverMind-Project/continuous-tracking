"""Fetches camera overlap groups from cognitive-companion.

Loaded once at startup and then on every CC config sync poll
(:class:`app.services.cc_config_sync.CCConfigSyncService`), so a group the operator
registers takes effect without a restart.

Overlap groups define cameras that share a physical field of view.  The
cross-camera associator uses this information to fuse tracklets across
group cameras aggressively (no transit-time budget, relaxed appearance
threshold) and to share face-anchor evidence within the same GlobalTrack.
"""

from __future__ import annotations

from typing import Final

from structlog import get_logger

from app.domain import OverlapGroup
from app.services.cc_client import CognitiveCompanionClient

logger = get_logger(__name__)

OVERLAP_GROUPS_PATH: Final = "/api/v1/cts/overlap_groups"


async def fetch_overlap_groups(client: CognitiveCompanionClient) -> list[OverlapGroup]:
    """Fetch camera overlap groups from CC.

    Raises when CC is unavailable so startup does not silently run with an
    empty cross-camera topology.
    """
    try:
        raw: object = await client.get(OVERLAP_GROUPS_PATH)
    except Exception as exc:
        logger.exception("overlap_groups_fetch_failed")
        raise RuntimeError("Failed to fetch overlap groups from CC") from exc

    groups = parse_overlap_groups(raw)
    logger.info("Fetched camera overlap groups from CC", group_count=len(groups))
    return groups


def parse_overlap_groups(raw: object) -> list[OverlapGroup]:
    """Validate CC's overlap-group list; skip malformed entries with one warning."""
    if not isinstance(raw, list):
        raise TypeError("CC overlap_groups response must be a list")

    groups: list[OverlapGroup] = []
    skipped = 0
    for r in raw:
        if not isinstance(r, dict):
            skipped += 1
            continue
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
    return groups

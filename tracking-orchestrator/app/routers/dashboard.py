"""Internal dashboard and keyframe endpoints consumed by the CC BFF.

These endpoints are NOT exposed publicly.  The CC backend is the only
authorized caller.

Routes:
    GET /internal/dashboard/signals
    GET /internal/dashboard/trajectory
    GET /internal/dashboard/dwell_summary
    GET /internal/keyframes
    GET /internal/keyframes/{keyframe_id}/bboxes
    PUT /internal/bboxes/{annotation_id}/override
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from app.storage.base import (
    BboxAnnotationRepository,
    DementiaSignalRepository,
    InMemoryBboxAnnotationRepository,
    InMemoryDementiaSignalRepository,
    InMemoryKeyframeRepository,
    InMemoryTrajectoryRepository,
    KeyframeRepository,
    TrajectoryRepository,
)

router = APIRouter(tags=["dashboard-internal"])

# ---------------------------------------------------------------------------
# Module-level repository singletons (replaced in tests via dependency override)
# ---------------------------------------------------------------------------

_signal_repo: DementiaSignalRepository = InMemoryDementiaSignalRepository()
_trajectory_repo: TrajectoryRepository = InMemoryTrajectoryRepository()
_keyframe_repo: KeyframeRepository = InMemoryKeyframeRepository()
_bbox_repo: BboxAnnotationRepository = InMemoryBboxAnnotationRepository()


def get_signal_repo() -> DementiaSignalRepository:
    return _signal_repo


def get_trajectory_repo() -> TrajectoryRepository:
    return _trajectory_repo


def get_keyframe_repo() -> KeyframeRepository:
    return _keyframe_repo


def get_bbox_repo() -> BboxAnnotationRepository:
    return _bbox_repo


def set_repos(
    signal: DementiaSignalRepository,
    trajectory: TrajectoryRepository,
    keyframe: KeyframeRepository,
) -> None:
    """Wire production repositories at startup (called from main.py lifespan)."""
    global _signal_repo, _trajectory_repo, _keyframe_repo
    _signal_repo = signal
    _trajectory_repo = trajectory
    _keyframe_repo = keyframe


def set_bbox_repo(bbox: BboxAnnotationRepository) -> None:
    """Wire the production bbox annotation repository."""
    global _bbox_repo
    _bbox_repo = bbox


# ---------------------------------------------------------------------------
# GET /internal/dashboard/signals
# ---------------------------------------------------------------------------


@router.get("/internal/dashboard/signals")
async def get_signals(
    person_id: str | None = Query(None, description="Filter by identity ID"),
    window_hours: int = Query(24, ge=1, le=720, description="Lookback window in hours"),
    signal_kind: str | None = Query(None, description="Filter by signal kind"),
    limit: int = Query(200, ge=1, le=1000),
    repo: DementiaSignalRepository = Depends(get_signal_repo),
) -> dict[str, Any]:
    """Return recent dementia signals for the dashboard."""
    after = datetime.now(UTC) - timedelta(hours=window_hours)
    signals = await repo.list_signals(
        identity_id=person_id,
        signal_kind=signal_kind,
        after=after,
        limit=limit,
    )
    return {
        "signals": [_signal_to_dict(s) for s in signals],
        "count": len(signals),
        "window_hours": window_hours,
    }


# ---------------------------------------------------------------------------
# GET /internal/dashboard/trajectory
# ---------------------------------------------------------------------------


@router.get("/internal/dashboard/trajectory")
async def get_trajectory(
    person_id: str = Query(..., description="Identity ID (required)"),
    start: str | None = Query(None, description="ISO-8601 start time"),
    end: str | None = Query(None, description="ISO-8601 end time"),
    limit: int = Query(500, ge=1, le=5000),
    repo: TrajectoryRepository = Depends(get_trajectory_repo),
) -> dict[str, Any]:
    """Return trajectory points for floor-plan overlay."""
    after: datetime | None = None
    if start:
        try:
            after = datetime.fromisoformat(start)
            if after.tzinfo is None:
                after = after.replace(tzinfo=UTC)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Invalid start timestamp: {start!r}",
            ) from exc

    points = await repo.list_trajectory_points(
        identity_id=person_id,
        after=after,
        limit=limit,
    )

    # Apply end filter in-memory (not worth a DB round-trip for a simple bound).
    if end:
        try:
            end_dt = datetime.fromisoformat(end)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=UTC)
            points = [p for p in points if p.observed_at <= end_dt]
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Invalid end timestamp: {end!r}",
            ) from exc

    return {
        "person_id": person_id,
        "points": [_point_to_dict(p) for p in points],
        "count": len(points),
    }


# ---------------------------------------------------------------------------
# GET /internal/dashboard/dwell_summary
# ---------------------------------------------------------------------------


@router.get("/internal/dashboard/dwell_summary")
async def get_dwell_summary(
    person_id: str = Query(..., description="Identity ID (required)"),
    date: str | None = Query(None, description="ISO-8601 date (YYYY-MM-DD); defaults to today"),
    repo: TrajectoryRepository = Depends(get_trajectory_repo),
) -> dict[str, Any]:
    """Return room dwell aggregation (time-in-room) for one day."""
    if date:
        try:
            day = datetime.fromisoformat(date)
            day = day.replace(tzinfo=UTC) if day.tzinfo is None else day.astimezone(UTC)
            day = day.replace(hour=0, minute=0, second=0, microsecond=0)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Invalid date: {date!r}",
            ) from exc
    else:
        now = datetime.now(UTC)
        day = now.replace(hour=0, minute=0, second=0, microsecond=0)

    day_end = day + timedelta(days=1)

    dwells = await repo.list_room_dwells(
        identity_id=person_id,
        after=day,
        limit=1000,
    )
    # Filter to the requested day.
    dwells = [d for d in dwells if d.entered_at < day_end]

    # Aggregate seconds per room.
    by_room: dict[str, int] = {}
    for dwell in dwells:
        dur = dwell.duration_seconds or 0
        by_room[dwell.room_name] = by_room.get(dwell.room_name, 0) + dur

    total = sum(by_room.values())
    rooms = [
        {"room_name": room, "duration_seconds": secs, "fraction": secs / total if total else 0.0}
        for room, secs in sorted(by_room.items(), key=lambda x: x[1], reverse=True)
    ]

    return {
        "person_id": person_id,
        "date": day.date().isoformat(),
        "total_seconds": total,
        "rooms": rooms,
    }


# ---------------------------------------------------------------------------
# GET /internal/keyframes
# ---------------------------------------------------------------------------


@router.get("/internal/keyframes")
async def list_keyframes(
    person_id: str | None = Query(None, description="Filter by identity ID"),
    tag_reason: str | None = Query(None, description="Filter by tag reason"),
    signal_type: str | None = Query(None, description="Filter by annotations.signal_type"),
    global_track_id: str | None = Query(None, description="Filter by global track"),
    strategy: str | None = Query(None, pattern="^lifecycle$"),
    after: str | None = Query(None, description="ISO-8601 start time"),
    limit: int = Query(100, ge=1, le=500),
    repo: KeyframeRepository = Depends(get_keyframe_repo),
) -> dict[str, Any]:
    """Return tagged keyframes for review."""
    after_dt: datetime | None = None
    if after:
        try:
            after_dt = datetime.fromisoformat(after)
            if after_dt.tzinfo is None:
                after_dt = after_dt.replace(tzinfo=UTC)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Invalid after timestamp: {after!r}",
            ) from exc

    fetch_limit = 500 if strategy == "lifecycle" else limit
    keyframes = await repo.list_keyframes(
        global_track_id=global_track_id,
        after=after_dt,
        limit=fetch_limit,
    )

    # Apply person_id and tag_reason filters in-memory (KeyframeRepository
    # protocol does not expose these filters to keep the interface minimal).
    if person_id:
        keyframes = [k for k in keyframes if k.annotations.get("identity_id") == person_id]
    if tag_reason:
        keyframes = [k for k in keyframes if k.tag_reason == tag_reason]
    if signal_type:
        keyframes = [k for k in keyframes if k.annotations.get("signal_type") == signal_type]
    if strategy == "lifecycle" and len(keyframes) > 3:
        ordered = sorted(keyframes, key=lambda k: k.captured_at)
        midpoint = ordered[len(ordered) // 2]
        keyframes = [ordered[0], midpoint, ordered[-1]]
    else:
        keyframes = keyframes[:limit]

    return {
        "keyframes": [_keyframe_to_dict(k) for k in keyframes],
        "count": len(keyframes),
    }


@router.get("/internal/keyframes/{sample_id}")
async def get_keyframe(
    sample_id: str,
    repo: KeyframeRepository = Depends(get_keyframe_repo),
) -> dict[str, Any]:
    """Return a single tagged keyframe by ID."""
    keyframe = await repo.get_keyframe(sample_id)
    if keyframe is not None:
        return _keyframe_to_dict(keyframe)
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"code": "keyframe.not_found", "message": f"Keyframe {sample_id} not found."},
    )


@router.post("/internal/keyframes/{sample_id}/retain", status_code=status.HTTP_200_OK)
async def retain_keyframe(
    sample_id: str,
    repo: KeyframeRepository = Depends(get_keyframe_repo),
) -> dict[str, Any]:
    """Extend a keyframe's retention past the normal window."""
    expires_at = datetime.now(UTC) + timedelta(days=365)
    if await repo.update_retention(sample_id, expires_at):
        return {
            "retained": True,
            "sample_id": sample_id,
            "expires_at": expires_at.isoformat(),
        }
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"code": "keyframe.not_found", "message": f"Keyframe {sample_id} not found."},
    )


# ---------------------------------------------------------------------------
# GET /internal/keyframes/{keyframe_id}/bboxes
# ---------------------------------------------------------------------------


@router.get("/internal/keyframes/{keyframe_id}/bboxes")
async def get_keyframe_bboxes(
    keyframe_id: str,
    repo: BboxAnnotationRepository = Depends(get_bbox_repo),
) -> dict[str, Any]:
    """Return YOLO bounding-box annotations for a tagged keyframe."""
    bboxes = await repo.get_bbox_annotations_for_keyframe(keyframe_id)
    return {"bboxes": [_bbox_to_dict(b) for b in bboxes], "count": len(bboxes)}


# ---------------------------------------------------------------------------
# PUT /internal/bboxes/{annotation_id}/override
# ---------------------------------------------------------------------------


class BboxOverrideBody(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    override_by: str = "caregiver"


@router.put("/internal/bboxes/{annotation_id}/override")
async def override_bbox(
    annotation_id: str,
    body: BboxOverrideBody,
    repo: BboxAnnotationRepository = Depends(get_bbox_repo),
) -> dict[str, Any]:
    """Persist a user-drawn bounding-box override.

    Returns the full updated annotation.  Raises 404 when the annotation
    does not exist.
    """
    await repo.save_override_bbox(
        annotation_id=annotation_id,
        x1=body.x1,
        y1=body.y1,
        x2=body.x2,
        y2=body.y2,
        override_by=body.override_by,
    )
    updated = await repo.get_annotation_by_id(annotation_id)
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "bbox_annotation.not_found",
                "message": f"Bbox annotation {annotation_id} not found.",
            },
        )
    return _bbox_to_dict(updated)


# ---------------------------------------------------------------------------
# PUT /internal/bboxes/{annotation_id}/tag
# ---------------------------------------------------------------------------


class BboxTagBody(BaseModel):
    identity_id: str | None
    tagged_by: str = "caregiver"


@router.put("/internal/bboxes/{annotation_id}/tag")
async def tag_bbox(
    annotation_id: str,
    body: BboxTagBody,
    repo: BboxAnnotationRepository = Depends(get_bbox_repo),
) -> dict[str, Any]:
    """Set or clear the identity_id on a single bbox annotation."""
    existing = await repo.get_annotation_by_id(annotation_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "bbox_annotation.not_found",
                "message": f"Bbox annotation {annotation_id} not found.",
            },
        )
    await repo.tag_annotation(annotation_id, body.identity_id)
    updated = await repo.get_annotation_by_id(annotation_id)
    return _bbox_to_dict(updated)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# DELETE /internal/bboxes/{annotation_id}
# ---------------------------------------------------------------------------


@router.delete("/internal/bboxes/{annotation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_bbox(
    annotation_id: str,
    repo: BboxAnnotationRepository = Depends(get_bbox_repo),
) -> None:
    """Delete a single bbox annotation by ID."""
    existing = await repo.get_annotation_by_id(annotation_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "bbox_annotation.not_found",
                "message": f"Bbox annotation {annotation_id} not found.",
            },
        )
    await repo.delete_annotation(annotation_id)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _signal_to_dict(s: Any) -> dict[str, Any]:
    return {
        "signal_id": s.signal_id,
        "identity_id": s.identity_id,
        "signal_kind": s.signal_kind,
        "severity": s.severity,
        "value": s.value,
        "baseline": s.baseline,
        "z_score": s.z_score,
        "window_start": s.window_start.isoformat(),
        "window_end": s.window_end.isoformat(),
        "context": s.context,
        "emitted_at": s.emitted_at.isoformat(),
    }


def _point_to_dict(p: Any) -> dict[str, Any]:
    return {
        "identity_id": p.identity_id,
        "global_track_id": p.global_track_id,
        "observed_at": p.observed_at.isoformat(),
        "room_name": p.room_name,
        "ground_x": p.ground_x,
        "ground_y": p.ground_y,
        "posture": p.posture,
        "motion_energy": p.motion_energy,
        "identity_confidence": p.identity_confidence,
    }


def _keyframe_to_dict(k: Any) -> dict[str, Any]:
    annotations = k.annotations if isinstance(k.annotations, dict) else {}
    return {
        "sample_id": k.keyframe_id,
        "keyframe_id": k.keyframe_id,
        "tracklet_id": k.tracklet_id,
        "global_track_id": k.global_track_id,
        "camera_id": k.camera_id,
        "minio_key": k.minio_key,
        "captured_at": k.captured_at.isoformat(),
        "annotations": annotations,
        "tag_reason": k.tag_reason,
        "expires_at": k.expires_at.isoformat(),
        "person_id": annotations.get("identity_id") or None,
        "signal_type": annotations.get("signal_type") or None,
        "severity": annotations.get("severity") or None,
    }


def _bbox_to_dict(b: Any) -> dict[str, Any]:
    return {
        "id": getattr(b, "id", None),
        "keyframe_id": b.keyframe_id,
        "tracklet_id": b.tracklet_id,
        "camera_id": b.camera_id,
        "x1": b.x1,
        "y1": b.y1,
        "x2": b.x2,
        "y2": b.y2,
        "detection_confidence": b.detection_confidence,
        "frame_width": b.frame_width,
        "frame_height": b.frame_height,
        "identity_id": b.identity_id,
        "created_at": b.created_at.isoformat(),
        "override_x1": getattr(b, "override_x1", None),
        "override_y1": getattr(b, "override_y1", None),
        "override_x2": getattr(b, "override_x2", None),
        "override_y2": getattr(b, "override_y2", None),
        "override_by": getattr(b, "override_by", None),
        "override_at": b.override_at.isoformat() if getattr(b, "override_at", None) else None,
    }

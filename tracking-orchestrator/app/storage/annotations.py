"""Bounding-box annotation storage."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Protocol

from ..domain import BboxAnnotation


class BboxAnnotationRepository(Protocol):
    """Persist YOLO bounding-box annotations per keyframe."""

    async def save_bbox_annotations(self, annotations: list[BboxAnnotation]) -> None: ...

    async def get_bbox_annotations_for_keyframe(self, keyframe_id: str) -> list[BboxAnnotation]: ...

    async def get_bbox_annotations_for_tracklet(self, tracklet_id: str) -> list[BboxAnnotation]: ...

    async def update_identity_id(self, tracklet_id: str, identity_id: str) -> None:
        """Called by IdentityRewriter when an identity is revised."""
        ...

    async def save_override_bbox(
        self,
        annotation_id: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        override_by: str,
    ) -> None:
        """Persist a user-drawn bbox override (written by M4 frontend path)."""
        ...

    async def get_annotation_by_id(self, annotation_id: str) -> BboxAnnotation | None:
        """Return a single bbox annotation by its UUID, or None."""
        ...

    async def tag_annotation(self, annotation_id: str, identity_id: str | None) -> None:
        """Set the identity_id on a single bbox annotation by its UUID."""
        ...

    async def delete_annotation(self, annotation_id: str) -> None:
        """Delete a single bbox annotation by its UUID."""
        ...


class InMemoryBboxAnnotationRepository:
    """In-memory store for bbox annotations."""

    def __init__(self) -> None:
        self._rows: dict[str, BboxAnnotation] = {}
        self._by_keyframe: dict[str, list[str]] = {}
        self._by_tracklet: dict[str, list[str]] = {}

    async def save_bbox_annotations(self, annotations: list[BboxAnnotation]) -> None:
        for ann in annotations:
            ann_id = str(uuid.uuid4())
            ann_with_id = BboxAnnotation(**{**ann.__dict__, "id": ann_id})
            self._rows[ann_id] = ann_with_id
            self._by_keyframe.setdefault(ann.keyframe_id, []).append(ann_id)
            self._by_tracklet.setdefault(ann.tracklet_id, []).append(ann_id)

    async def get_bbox_annotations_for_keyframe(self, keyframe_id: str) -> list[BboxAnnotation]:
        return [self._rows[i] for i in self._by_keyframe.get(keyframe_id, [])]

    async def get_bbox_annotations_for_tracklet(self, tracklet_id: str) -> list[BboxAnnotation]:
        return [self._rows[i] for i in self._by_tracklet.get(tracklet_id, [])]

    async def get_annotation_by_id(self, annotation_id: str) -> BboxAnnotation | None:
        return self._rows.get(annotation_id)

    async def update_identity_id(self, tracklet_id: str, identity_id: str) -> None:
        for ann_id, ann in list(self._rows.items()):
            if ann.tracklet_id == tracklet_id:
                self._rows[ann_id] = BboxAnnotation(**{**ann.__dict__, "identity_id": identity_id})

    async def save_override_bbox(
        self,
        annotation_id: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        override_by: str,
    ) -> None:
        if annotation_id in self._rows:
            ann = self._rows[annotation_id]
            self._rows[annotation_id] = BboxAnnotation(
                **{
                    **ann.__dict__,
                    "override_x1": x1,
                    "override_y1": y1,
                    "override_x2": x2,
                    "override_y2": y2,
                    "override_by": override_by,
                    "override_at": datetime.now(UTC),
                }
            )

    async def tag_annotation(self, annotation_id: str, identity_id: str | None) -> None:
        if annotation_id in self._rows:
            ann = self._rows[annotation_id]
            self._rows[annotation_id] = BboxAnnotation(
                **{**ann.__dict__, "identity_id": identity_id}
            )

    async def delete_annotation(self, annotation_id: str) -> None:
        ann = self._rows.pop(annotation_id, None)
        if ann is not None:
            if ann.keyframe_id in self._by_keyframe:
                self._by_keyframe[ann.keyframe_id] = [
                    i for i in self._by_keyframe[ann.keyframe_id] if i != annotation_id
                ]
            if ann.tracklet_id in self._by_tracklet:
                self._by_tracklet[ann.tracklet_id] = [
                    i for i in self._by_tracklet[ann.tracklet_id] if i != annotation_id
                ]

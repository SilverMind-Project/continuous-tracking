"""Bounding-box annotation storage."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from ..domain import BboxAnnotation


class BboxBatchOperation:
    """A single operation in a bbox annotation batch."""

    def __init__(
        self,
        op: Literal["create", "update", "delete"],
        annotation_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.op = op
        self.annotation_id = annotation_id
        self.data = data or {}


class BboxAnnotationRepository(Protocol):
    """Persist YOLO bounding-box annotations per keyframe."""

    async def save_bbox_annotations(self, annotations: list[BboxAnnotation]) -> None: ...

    async def get_bbox_annotations_for_keyframe(self, keyframe_id: str) -> list[BboxAnnotation]: ...

    async def get_bbox_annotations_for_keyframes(
        self, keyframe_ids: list[str]
    ) -> list[BboxAnnotation]:
        """Batch fetch annotations for many keyframes in one query (M07 read model)."""
        ...

    async def get_bbox_annotations_for_ph(self, ph_id: str) -> list[BboxAnnotation]: ...

    async def update_identity_id(self, ph_id: str, identity_id: str) -> None:
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
        """Persist a user-drawn bbox override."""
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

    # --- Idempotent delete + batch ops ---

    async def delete_annotation_if_exists(self, annotation_id: str) -> bool:
        """Delete by ID. Returns True if a row was deleted, False if none existed."""
        ...

    def transaction(self) -> AsyncIterator[None]:
        """Async context manager for batch operations. Yield, then commit/rollback."""
        ...

    async def apply_bbox_batch(
        self,
        keyframe_id: str,
        operations: list[BboxBatchOperation],
    ) -> list[dict[str, Any]]:
        """Apply a batch of create/update/delete operations atomically."""
        ...

    async def delete_annotations_below_confidence(self, threshold: float, since: datetime) -> int:
        """Drop annotations with detection_confidence below *threshold*. Returns count."""
        ...


class InMemoryBboxAnnotationRepository:
    """In-memory store for bbox annotations."""

    def __init__(self) -> None:
        self._rows: dict[str, BboxAnnotation] = {}
        self._by_keyframe: dict[str, list[str]] = {}
        self._by_ph: dict[str, list[str]] = {}

    async def save_bbox_annotations(self, annotations: list[BboxAnnotation]) -> None:
        for ann in annotations:
            ann_id = str(uuid.uuid4())
            ann_with_id = BboxAnnotation(**{**ann.__dict__, "id": ann_id})
            self._rows[ann_id] = ann_with_id
            self._by_keyframe.setdefault(ann.keyframe_id, []).append(ann_id)
            self._by_ph.setdefault(ann.ph_id, []).append(ann_id)

    async def get_bbox_annotations_for_keyframe(self, keyframe_id: str) -> list[BboxAnnotation]:
        return [self._rows[i] for i in self._by_keyframe.get(keyframe_id, [])]

    async def get_bbox_annotations_for_keyframes(
        self, keyframe_ids: list[str]
    ) -> list[BboxAnnotation]:
        result: list[BboxAnnotation] = []
        for kf_id in keyframe_ids:
            result.extend(self._rows[i] for i in self._by_keyframe.get(kf_id, []))
        return result

    async def get_bbox_annotations_for_ph(self, ph_id: str) -> list[BboxAnnotation]:
        return [self._rows[i] for i in self._by_ph.get(ph_id, [])]

    async def get_annotation_by_id(self, annotation_id: str) -> BboxAnnotation | None:
        return self._rows.get(annotation_id)

    async def update_identity_id(self, ph_id: str, identity_id: str) -> None:
        for ann_id, ann in list(self._rows.items()):
            if ann.ph_id == ph_id:
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
            if ann.ph_id in self._by_ph:
                self._by_ph[ann.ph_id] = [i for i in self._by_ph[ann.ph_id] if i != annotation_id]

    # --- Batch methods ---

    async def delete_annotation_if_exists(self, annotation_id: str) -> bool:
        existed = annotation_id in self._rows
        await self.delete_annotation(annotation_id)
        return existed

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        # InMemory: no-op (operations are already atomic in dict).
        yield

    async def apply_bbox_batch(
        self,
        keyframe_id: str,
        operations: list[BboxBatchOperation],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for op in operations:
            if op.op == "delete" and op.annotation_id:
                await self.delete_annotation(op.annotation_id)
                results.append({"op": "delete", "annotation_id": op.annotation_id, "ok": True})
            elif op.op == "update" and op.annotation_id and op.data:
                if op.annotation_id in self._rows:
                    ann = self._rows[op.annotation_id]
                    d = op.data
                    patch: dict[str, Any] = {}
                    for key in (
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                        "detection_confidence",
                        "frame_width",
                        "frame_height",
                        "identity_id",
                    ):
                        if key in d:
                            patch[key] = d[key]
                    self._rows[op.annotation_id] = BboxAnnotation(
                        **{
                            **ann.__dict__,
                            **patch,
                        }
                    )
                    results.append({"op": "update", "annotation_id": op.annotation_id, "ok": True})
                else:
                    results.append(
                        {
                            "op": "update",
                            "annotation_id": op.annotation_id,
                            "ok": False,
                            "error": "not_found",
                        }
                    )
            elif op.op == "create" and op.data:
                new_id = str(uuid.uuid4())
                ann = BboxAnnotation(
                    keyframe_id=keyframe_id,
                    ph_id=str(op.data.get("ph_id") or ""),
                    camera_id=str(op.data.get("camera_id") or ""),
                    x1=float(op.data.get("x1", 0)),
                    y1=float(op.data.get("y1", 0)),
                    x2=float(op.data.get("x2", 0)),
                    y2=float(op.data.get("y2", 0)),
                    detection_confidence=float(op.data.get("detection_confidence", 0.5)),
                    frame_width=int(op.data.get("frame_width", 0)),
                    frame_height=int(op.data.get("frame_height", 0)),
                    identity_id=op.data.get("identity_id"),
                    created_at=datetime.now(UTC),
                    id=new_id,
                )
                self._rows[new_id] = ann
                self._by_keyframe.setdefault(keyframe_id, []).append(new_id)
                if ann.ph_id:
                    self._by_ph.setdefault(ann.ph_id, []).append(new_id)
                results.append({"op": "create", "annotation_id": new_id, "ok": True})
        return results

    async def delete_annotations_below_confidence(self, threshold: float, since: datetime) -> int:
        count = 0
        for ann_id, ann in list(self._rows.items()):
            if ann.detection_confidence < threshold and ann.created_at >= since:
                await self.delete_annotation(ann_id)
                count += 1
        return count

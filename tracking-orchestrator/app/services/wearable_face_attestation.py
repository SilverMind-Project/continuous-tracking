"""Offers faces of wearable-attested PHs to person-identification-service's attested queue.

When a wearable association is attested, the faces the face stage emitted for that PH's
detections are faces of the tag's member, and person-identification-service still holds them
in a short in-memory buffer keyed by detection id. This service batches those detection ids per
association segment and grade (one call per ``batch_interval_s`` of capture time) and asks
person-identification-service to keep them as ``sensor_attested`` calibration candidates. They
stay operator-gated there: an audit sample decides whether they may ever calibrate, and only an
operator promotes one into recognition.

A face is offered only when its gate outcome was ``emitted`` (anchors in the live list); a face
dropped by head-region attribution can belong to a neighbour. A detection on which any face,
live or shadow-only, was recognized as someone else is never offered.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol

from structlog import get_logger

from app.domain import FaceAnchor, WearableEvidence
from app.inference.face_id_client import FaceAttestationResult
from app.observability import metrics as _metrics
from app.observability.metrics import WearableFaceAttestationResult

logger = get_logger(__name__)


class FaceAttestationClient(Protocol):
    """The person-identification-service call this service needs."""

    async def attest_faces(
        self,
        *,
        association_id: str,
        identity_id: str,
        attestation_grade: str,
        detection_ids: Sequence[str],
    ) -> FaceAttestationResult | None: ...


@dataclass
class _Batch:
    identity_id: str
    opened_at: datetime
    detection_ids: list[str] = field(default_factory=list)


class WearableFaceAttestationService:
    """Batches emitted faces of attested PHs and sends them for operator-gated attestation."""

    def __init__(
        self,
        client: FaceAttestationClient,
        *,
        batch_interval_s: float = 10.0,
        max_batch: int = 64,
    ) -> None:
        self._client = client
        self._interval = timedelta(seconds=batch_interval_s)
        self._max_batch = max_batch
        # (segment_id, grade) -> pending detection ids. A grade change opens a new batch, so a
        # face collected while the association was room grade is never sent as trajectory.
        self._batches: dict[tuple[str, str], _Batch] = {}

    async def offer_round(
        self,
        *,
        evidence_by_ph: Mapping[str, WearableEvidence],
        det_to_ph: Mapping[str, str],
        emitted_faces: Iterable[FaceAnchor],
        shadow_faces: Iterable[FaceAnchor],
        captured_at: datetime,
    ) -> None:
        """Queue this round's eligible detections, then send every batch that is due.

        Call it every round, also when nothing is attested: due batches are sent here only.
        """
        emitted_by_det: dict[str, list[FaceAnchor]] = {}
        for face in emitted_faces:
            if face.detection_id:
                emitted_by_det.setdefault(face.detection_id, []).append(face)
        others_by_det: dict[str, list[FaceAnchor]] = {}
        for face in shadow_faces:
            if face.detection_id:
                others_by_det.setdefault(face.detection_id, []).append(face)

        counter = _metrics.metrics.cts_wearable_face_attestations_total
        for det_id, ph_id in det_to_ph.items():
            evidence = evidence_by_ph.get(ph_id)
            faces = emitted_by_det.get(det_id)
            if evidence is None or not evidence.attested or not faces or not evidence.segment_id:
                continue
            if any(
                face.recognition_state == "recognized"
                and face.person_id is not None
                and face.person_id != evidence.identity_id
                for face in (*faces, *others_by_det.get(det_id, ()))
            ):
                counter.labels(result=WearableFaceAttestationResult.EXCLUDED_CONFLICT).inc()
                continue
            key = (evidence.segment_id, str(evidence.attestation_grade))
            batch = self._batches.setdefault(
                key, _Batch(identity_id=evidence.identity_id, opened_at=captured_at)
            )
            if det_id not in batch.detection_ids:
                batch.detection_ids.append(det_id)
                counter.labels(result=WearableFaceAttestationResult.QUEUED).inc()

        for key in [
            k
            for k, b in self._batches.items()
            if captured_at - b.opened_at >= self._interval
            or len(b.detection_ids) >= self._max_batch
        ]:
            await self._send(key, self._batches.pop(key))

    async def purge_identity(self, identity_id: str) -> int:
        """Drop pending batches of a purged identity."""
        doomed = [k for k, b in self._batches.items() if b.identity_id == identity_id]
        for key in doomed:
            del self._batches[key]
        return len(doomed)

    async def _send(self, key: tuple[str, str], batch: _Batch) -> None:
        segment_id, grade = key
        counter = _metrics.metrics.cts_wearable_face_attestations_total
        result = await self._client.attest_faces(
            association_id=segment_id,
            identity_id=batch.identity_id,
            attestation_grade=grade,
            detection_ids=batch.detection_ids,
        )
        if result is None:
            counter.labels(result=WearableFaceAttestationResult.FAILED).inc(
                len(batch.detection_ids)
            )
            return
        counter.labels(result=WearableFaceAttestationResult.SENT).inc(len(batch.detection_ids))
        counter.labels(result=WearableFaceAttestationResult.MOVED).inc(result.moved)
        counter.labels(result=WearableFaceAttestationResult.BUFFER_MISS).inc(result.missed)
        logger.debug(
            "wearable_face_attestation_sent",
            segment_id=segment_id,
            grade=grade,
            sent=len(batch.detection_ids),
            moved=result.moved,
            missed=result.missed,
        )

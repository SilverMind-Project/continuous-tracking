"""Face templates: several faces of one Person Hypothesis matched together by person-id.

``FaceTemplateService`` is the one object ``WorldTrackingStage`` talks to, once before and once
after each tracker round:

- ``pending_anchors(now)``: template results that arrived since the last round, as face anchors
  for the identity resolver only (never association), each tied to its PH by ``tracklet_id``.
- ``after_round(...)``: feeds the faces associated in this round into the per-PH buffer, resets a
  PH whose identity changed, evicts old faces, and starts a template request for every PH whose
  best faces changed, has enough of them, and is off cooldown.

Requests run in background tasks with bounded concurrency (``FaceTemplateMatcher``), so a slow
person-id never delays a frame; a result applies on a later round, once its PH is observed. A
result requested before a reset of its PH (identity change, operator split, merge, correction,
purge) is discarded. Embeddings stay in memory and only travel to person-id.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from structlog import get_logger

from app.domain import FaceAnchor
from app.inference.evidence import FaceEvidence
from app.inference.face_id_client import (
    TEMPLATE_PROFILE,
    TemplateMatchResult,
    TemplateMember,
)
from app.observability import metrics as _metrics
from app.tracking.identity.face_template_buffer import (
    BufferAdd,
    BufferedFace,
    FaceTemplateBuffer,
    FaceTemplatePolicy,
)

logger = get_logger(__name__)


class TemplateMatchClient(Protocol):
    async def match_template(
        self, members: Sequence[TemplateMember], embedding_model_version: str
    ) -> TemplateMatchResult | None: ...


@dataclass(frozen=True)
class CompletedTemplate:
    """A template result waiting for its PH's next resolver round."""

    ph_id: str
    ticket: int
    members: tuple[BufferedFace, ...]
    result: TemplateMatchResult


class FaceTemplateMatcher:
    """Runs template requests in background tasks; at most one in flight per PH."""

    def __init__(self, client: TemplateMatchClient, policy: FaceTemplatePolicy) -> None:
        self._client = client
        self._policy = policy
        self._slots = asyncio.Semaphore(policy.max_concurrency)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._completed: dict[str, CompletedTemplate] = {}

    def submit(self, ph_id: str, ticket: int, members: tuple[BufferedFace, ...]) -> bool:
        """Start a request; False when one is already in flight for the PH or the limit is hit."""
        if ph_id in self._tasks:
            return False
        if len(self._tasks) >= self._policy.max_in_flight:
            _metrics.metrics.cts_face_template_requests_total.labels(
                result="dropped_in_flight_limit"
            ).inc()
            return False
        task = asyncio.create_task(self._run(ph_id, ticket, members))
        self._tasks[ph_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(ph_id, None))
        return True

    async def _run(self, ph_id: str, ticket: int, members: tuple[BufferedFace, ...]) -> None:
        started = time.monotonic()
        async with self._slots:
            result = await self._client.match_template(
                [TemplateMember(embedding=f.embedding, quality=f.quality) for f in members],
                members[0].embedding_model_version,
            )
        _metrics.metrics.cts_face_template_latency_seconds.observe(time.monotonic() - started)
        _metrics.metrics.cts_face_template_requests_total.labels(
            result=result.recognition_state if result is not None else "error"
        ).inc()
        if result is not None:
            self._completed[ph_id] = CompletedTemplate(ph_id, ticket, members, result)

    def completed(self) -> list[CompletedTemplate]:
        return list(self._completed.values())

    def discard(self, ph_id: str, ticket: int) -> None:
        """Forget the PH's result if it is still the one with ``ticket``."""
        current = self._completed.get(ph_id)
        if current is not None and current.ticket == ticket:
            del self._completed[ph_id]

    def in_flight(self) -> int:
        return len(self._tasks)

    async def drain(self) -> None:
        """Wait for every request in flight (tests and shutdown)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks.values()), return_exceptions=True)

    async def stop(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()
        await asyncio.gather(*list(self._tasks.values()), return_exceptions=True)
        self._tasks.clear()
        self._completed.clear()


@dataclass(frozen=True)
class TemplateAnchor:
    anchor: FaceAnchor
    evidence: FaceEvidence | None
    # Why a recognized template cannot grant ArcFace authority; None when it may.
    authority_blocked: str | None


def template_anchor(
    completed: CompletedTemplate,
    *,
    shadow_only: bool,
    expected_quality_version: str,
) -> TemplateAnchor | None:
    """The face anchor (and evidence record) a template result becomes; None if it names no one.

    The anchor is tied to its PH by ``tracklet_id``; ``detection_id`` is the newest member's, for
    provenance. Its calibrated confidence is kept only from a template-profile artifact fitted on
    the expected quality scorer. It may grant authority only if every member face person-id
    recognized on its own names the template's identity.
    """
    result = completed.result
    if result.recognition_state == "recognized":
        person_id: str | None = result.person_id
    elif result.recognition_state == "candidate":
        person_id = result.best_candidate_id
    else:
        person_id = None
    if not person_id or person_id == "unknown":
        return None

    calibrated: float | None = None
    if (
        result.recognition_state == "recognized"
        and result.calibration_profile == TEMPLATE_PROFILE
        and result.quality_version == expected_quality_version
    ):
        calibrated = result.calibrated_confidence

    blocked: str | None = None
    if result.recognition_state == "recognized":
        member_ids = {f.single_frame_person_id for f in completed.members}
        if any(m is not None and m != person_id for m in member_ids):
            blocked = "member_disagreement"
        elif calibrated is None:
            blocked = "uncalibrated"

    newest = max(completed.members, key=lambda f: f.captured_at)
    anchor = FaceAnchor(
        person_id=person_id,
        match_score=result.raw_similarity,
        quality=1.0,
        tracklet_id=completed.ph_id,
        detection_id=newest.detection_id,
        camera_id=newest.camera_id,
        captured_at=newest.captured_at,
        recognition_state=result.recognition_state,
        similarity=result.raw_similarity,
        # Medians, so a template is never scored above its typical member.
        yaw_deg=statistics.median(f.yaw_deg for f in completed.members),
        det_score=statistics.median(f.det_score for f in completed.members),
        calibrated_confidence=calibrated,
        face_quality=result.quality.score,
        face_quality_version=f"{result.aggregator_version}/{result.quality_version}",
        shadow_only=shadow_only,
        aggregation="template",
        authority_eligible=blocked != "member_disagreement",
    )
    evidence = (
        None
        if shadow_only
        else FaceEvidence(
            person_id=person_id,
            match_score=result.raw_similarity,
            tracklet_id=completed.ph_id,
            detection_id=newest.detection_id,
            camera_id=newest.camera_id,
            source="template",
            quality=1.0,
            model_version=result.aggregator_version,
            captured_at=newest.captured_at,
            recognition_state=result.recognition_state,
            similarity=result.raw_similarity,
            yaw_deg=anchor.yaw_deg,
            calibrated_confidence=calibrated,
        )
    )
    return TemplateAnchor(anchor, evidence, blocked)


class FaceTemplateService:
    """Per-round coordination of the buffer, the matcher, and template anchors."""

    def __init__(
        self,
        policy: FaceTemplatePolicy,
        client: TemplateMatchClient,
        *,
        expected_quality_version: str,
    ) -> None:
        if policy.mode == "off":
            raise ValueError("FaceTemplateService is not built when face_templates.mode is off")
        self._policy = policy
        self._buffer = FaceTemplateBuffer(policy)
        self._matcher = FaceTemplateMatcher(client, policy)
        self._expected_quality_version = expected_quality_version
        # PH -> (ticket, authority-block reason) of the template offered to this round's resolver.
        self._offered: dict[str, tuple[int, str | None]] = {}

    @property
    def policy(self) -> FaceTemplatePolicy:
        return self._policy

    def pending_anchors(self, now: datetime) -> tuple[list[FaceAnchor], list[FaceEvidence]]:
        """Template anchors for this round's resolver, and evidence records for live ones.

        Stale results (requested before a reset of their PH) and expired ones are dropped here;
        the rest stay pending until their PH is resolved in a round (``after_round``).
        """
        anchors: list[FaceAnchor] = []
        evidence: list[FaceEvidence] = []
        self._offered = {}
        horizon = now - timedelta(seconds=self._policy.max_age_s)
        for completed in self._matcher.completed():
            reason: str | None = None
            if not self._buffer.is_current(completed.ph_id, completed.ticket):
                reason = "stale_after_reset"
            elif max(f.captured_at for f in completed.members) < horizon:
                reason = "expired"
            built = None
            if reason is None:
                built = template_anchor(
                    completed,
                    shadow_only=self._policy.mode == "shadow",
                    expected_quality_version=self._expected_quality_version,
                )
                if built is None:
                    reason = "not_identifying"
            if reason is not None or built is None:
                _metrics.metrics.cts_face_template_results_discarded_total.labels(
                    reason=reason or "not_identifying"
                ).inc()
                self._matcher.discard(completed.ph_id, completed.ticket)
                continue
            anchors.append(built.anchor)
            if built.evidence is not None:
                evidence.append(built.evidence)
            self._offered[completed.ph_id] = (completed.ticket, built.authority_blocked)
        return anchors, evidence

    def after_round(
        self,
        *,
        faces: Iterable[BufferedFace],
        det_to_ph: Mapping[str, str],
        committed: Mapping[str, str | None],
        now: datetime,
    ) -> None:
        """Feed this round's associated faces, reset changed PHs, evict, and trigger requests.

        ``committed`` maps every PH the resolver decided this round to its identity (None for
        Unknown); a template offered to one of them has now been applied and is consumed.
        """
        for ph_id, identity_id in committed.items():
            self._buffer.observe_identity(ph_id, identity_id, now)
            offered = self._offered.pop(ph_id, None)
            if offered is not None:
                ticket, blocked = offered
                self._matcher.discard(ph_id, ticket)
                if blocked is not None:
                    _metrics.metrics.cts_face_template_authority_blocked_total.labels(
                        reason=blocked
                    ).inc()

        changed: set[str] = set()
        for face in faces:
            face_ph_id = det_to_ph.get(face.detection_id)
            if face_ph_id is None:
                continue
            outcome = self._buffer.add(face_ph_id, face)
            if outcome is BufferAdd.OUTLIER:
                _metrics.metrics.cts_face_template_outliers_total.inc()
            elif outcome is BufferAdd.KEPT:
                changed.add(face_ph_id)
        self._buffer.evict_older_than(now)

        for ph_id in sorted(changed):
            if self._buffer.should_trigger(ph_id, now):
                self._matcher.submit(ph_id, self._buffer.ticket(), self._buffer.snapshot(ph_id))
        for ph_id in self._buffer.ph_ids():
            _metrics.metrics.cts_face_template_buffer_size.observe(self._buffer.size(ph_id))

    def reset_ph(self, ph_id: str) -> None:
        """Drop a PH's faces and any template computed from them (operator split, merge, fix).

        Called from request handlers, outside any tracker round, so the reset is stamped with the
        wall clock.
        """
        self._buffer.reset(ph_id, datetime.now(UTC))
        logger.debug("face_template_reset", ph_id=ph_id)

    def reset_identity(self, identity_id: str) -> None:
        """Drop every PH's faces tied to a purged identity."""
        self._buffer.reset_identity(identity_id, datetime.now(UTC))

    @property
    def buffer(self) -> FaceTemplateBuffer:
        return self._buffer

    @property
    def matcher(self) -> FaceTemplateMatcher:
        return self._matcher

    async def stop(self) -> None:
        await self._matcher.stop()

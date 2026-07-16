"""M07 keyframe read model: group trigger rows into physical-frame cards.

``KeyframeStage`` writes one ``tagged_keyframes`` row per *triggering* Person
Hypothesis, and persists the whole frame's bbox set (one row per visible
person) under each of those trigger rows. So a single physical source frame
with ``K`` triggers and ``N`` visible people yields ``K`` keyframe rows and
``K * N`` ``keyframe_bbox_annotations`` rows.

This read model collapses that back to one card per physical frame. The
physical-frame identity is the immutable source tuple
``(camera_id, minio_key, captured_at)`` hashed to a deterministic
``physical_frame_id``; presigned URLs never participate. Existing trigger rows
remain audit records and are exposed as ``triggers``.

For every deduplicated bbox the card carries explicit, server-computed
provenance: the inferred identity (raw inference, immutable), the effective
identity (M06 operator/inferred revision ranges applied), authority, decision
source, calibrated confidence, conflict, revision, and pending-ReID state. The
frontend never derives any of these.

Composition is a pure function over already-fetched lists so the in-memory and
Postgres repositories share identical grouping/dedup/effective-identity logic
and the query count stays bounded (one keyframe window query plus a constant
number of batched provenance lookups keyed by the page's PH set).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from ..domain import (
    BboxAnnotation,
    IdentityProvenanceDecision,
    IdentityRevisionRange,
    TaggedKeyframe,
)
from ..storage.annotations import BboxAnnotationRepository
from ..storage.base import IdentityDecisionRepositoryProtocol, KeyframeRepository
from ..storage.corrections import IdentityCorrectionRepositoryProtocol
from ..storage.gallery import GalleryRepository
from ..tracking.identity.types import IdentityAuthority

# Stable namespace for deriving ``physical_frame_id`` from the source tuple.
_PHYSICAL_FRAME_NS = uuid.UUID("6f3d2c1a-0b4e-5a6f-9c8d-1e2f3a4b5c6d")

# Bounded authority vocabulary (F9/M07). A stored row outside this set
# (empty string, or a pre-M07 identity id on the ArcFace-authority path)
# composes to ``none`` at read time — display tolerance only, never persisted.
_KNOWN_AUTHORITIES = frozenset(a.value for a in IdentityAuthority)


def _normalize_authority(authority: str) -> str:
    return authority if authority in _KNOWN_AUTHORITIES else IdentityAuthority.NONE.value


def physical_frame_id(camera_id: str, minio_key: str, captured_at: datetime) -> str:
    """Deterministic ID for one physical source frame.

    Keyed only on immutable source attributes; two trigger rows that reference
    the same ``(camera_id, minio_key, captured_at)`` map to the same card.
    """
    key = f"{camera_id}\x00{minio_key}\x00{captured_at.isoformat()}"
    return str(uuid.uuid5(_PHYSICAL_FRAME_NS, key))


# ---------------------------------------------------------------------------
# Read-model value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KeyframeTrigger:
    """One audit trigger row that contributed to a physical-frame card."""

    keyframe_id: str
    ph_id: str
    tag_reason: str


@dataclass(frozen=True)
class BboxIdentityView:
    """One deduplicated bbox annotation with full effective-identity provenance."""

    bbox_id: str | None
    ph_id: str
    x1: float
    y1: float
    x2: float
    y2: float
    detection_confidence: float
    frame_width: int
    frame_height: int
    inferred_identity_id: str | None
    effective_identity_id: str | None
    authority: str
    decision_source: str
    calibrated_confidence: float | None
    conflict: bool
    conflict_kind: str | None
    revision_id: str | None
    pending_review: bool
    decision_id: str | None
    override_x1: float | None = None
    override_y1: float | None = None
    override_x2: float | None = None
    override_y2: float | None = None


@dataclass(frozen=True)
class PhysicalFrameCard:
    """One card per physical source frame with every visible bbox identity."""

    physical_frame_id: str
    camera_id: str
    minio_key: str
    captured_at: datetime
    frame_width: int
    frame_height: int
    triggers: list[KeyframeTrigger]
    trigger_reasons: list[str]
    bboxes: list[BboxIdentityView]
    unknown_count: int
    conflict_count: int
    pending_review_count: int


@dataclass(frozen=True)
class PhysicalFramePage:
    """A page of grouped physical-frame cards plus the total matching count.

    ``truncated`` is True when the keyframe scan hit its window cap, so ``total``
    counts only frames within the most recent window rather than all history.
    """

    frames: list[PhysicalFrameCard]
    total: int
    truncated: bool = False


@dataclass(frozen=True)
class KeyframeReadFilters:
    """Server-side filters applied before grouped-frame pagination."""

    camera_id: str | None = None
    tag_reason: str | None = None
    after: datetime | None = None
    before: datetime | None = None
    effective_identity_id: str | None = None
    explicit_unknown: bool = False
    authority: str | None = None
    decision_source: str | None = None
    conflict_only: bool = False
    pending_review_only: bool = False


# ---------------------------------------------------------------------------
# Bounded batch reads the service depends on (Protocol + InMemory + Postgres)
# ---------------------------------------------------------------------------


class KeyframeReadRepositories(Protocol):
    """Bounded batch reads required to compose physical-frame cards.

    Each method runs at most one query; the service calls each once per page so
    the total query count is independent of frame/bbox cardinality.
    """

    async def list_keyframes_for_read_model(
        self,
        *,
        camera_id: str | None,
        tag_reason: str | None,
        after: datetime | None,
        before: datetime | None,
        limit: int,
    ) -> list[TaggedKeyframe]: ...

    async def get_bbox_annotations_for_keyframes(
        self, keyframe_ids: list[str]
    ) -> list[BboxAnnotation]: ...

    async def decisions_for_phs(
        self, ph_ids: list[str], at_or_before: datetime
    ) -> dict[str, list[IdentityProvenanceDecision]]: ...

    async def live_ranges_for_phs(
        self, ph_ids: list[str]
    ) -> dict[str, list[IdentityRevisionRange]]: ...

    async def phs_with_pending_reid(self, ph_ids: list[str]) -> set[str]: ...


# Internal cap on the keyframe window scanned for one page. Grouping collapses
# trigger rows, so this is intentionally generous; admin review is recency-first.
_READ_WINDOW_CAP = 5000

# Pixel rounding for deduplicating bboxes that carry no PH id.
_DEDUP_ROUND = 1


class KeyframeReadModelService:
    """Compose paginated physical-frame cards from bounded batch reads."""

    def __init__(self, repos: KeyframeReadRepositories) -> None:
        self._repos = repos

    async def list_physical_frames(
        self,
        *,
        filters: KeyframeReadFilters | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> PhysicalFramePage:
        filters = filters or KeyframeReadFilters()

        keyframes = await self._repos.list_keyframes_for_read_model(
            camera_id=filters.camera_id,
            tag_reason=filters.tag_reason,
            after=filters.after,
            before=filters.before,
            limit=_READ_WINDOW_CAP,
        )
        if not keyframes:
            return PhysicalFramePage(frames=[], total=0)

        truncated = len(keyframes) >= _READ_WINDOW_CAP

        keyframe_ids = [k.keyframe_id for k in keyframes]
        bboxes = await self._repos.get_bbox_annotations_for_keyframes(keyframe_ids)

        ph_ids = sorted(
            {b.ph_id for b in bboxes if b.ph_id} | {k.ph_id for k in keyframes if k.ph_id}
        )
        page_max_time = max(k.captured_at for k in keyframes)
        # Decisions are persisted only at identity change points, so a held PH's
        # applicable decision may predate this page's window. Fetch each PH's
        # decisions up to the newest frame time (no lower bound) and let the
        # composer pick the latest at or before each frame's capture time --
        # never the page's newest decision. Change-gated writes keep this set
        # small per PH.
        decisions = await self._repos.decisions_for_phs(ph_ids, page_max_time)
        ranges = await self._repos.live_ranges_for_phs(ph_ids)
        pending = await self._repos.phs_with_pending_reid(ph_ids)

        cards = _compose_cards(
            keyframes=keyframes,
            bboxes=bboxes,
            decisions=decisions,
            ranges=ranges,
            pending=pending,
        )
        cards = [c for c in cards if _frame_matches(c, filters)]
        cards.sort(key=lambda c: (c.captured_at, c.physical_frame_id), reverse=True)

        total = len(cards)
        page = cards[offset : offset + limit]
        return PhysicalFramePage(frames=page, total=total, truncated=truncated)


# ---------------------------------------------------------------------------
# Pure composition (shared by every repository implementation)
# ---------------------------------------------------------------------------


@dataclass
class _FrameGroup:
    camera_id: str
    minio_key: str
    captured_at: datetime
    triggers: list[KeyframeTrigger] = field(default_factory=list)
    keyframe_ids: list[str] = field(default_factory=list)


def _compose_cards(
    *,
    keyframes: list[TaggedKeyframe],
    bboxes: list[BboxAnnotation],
    decisions: dict[str, list[IdentityProvenanceDecision]],
    ranges: dict[str, list[IdentityRevisionRange]],
    pending: set[str],
) -> list[PhysicalFrameCard]:
    groups: dict[str, _FrameGroup] = {}
    for kf in keyframes:
        fid = physical_frame_id(kf.camera_id, kf.minio_key, kf.captured_at)
        group = groups.get(fid)
        if group is None:
            group = _FrameGroup(
                camera_id=kf.camera_id,
                minio_key=kf.minio_key,
                captured_at=kf.captured_at,
            )
            groups[fid] = group
        group.keyframe_ids.append(kf.keyframe_id)
        group.triggers.append(
            KeyframeTrigger(keyframe_id=kf.keyframe_id, ph_id=kf.ph_id, tag_reason=kf.tag_reason)
        )

    bboxes_by_keyframe: dict[str, list[BboxAnnotation]] = {}
    for b in bboxes:
        bboxes_by_keyframe.setdefault(b.keyframe_id, []).append(b)

    cards: list[PhysicalFrameCard] = []
    for fid, group in groups.items():
        # Union every bbox across the frame's trigger rows, then deduplicate by
        # PH identity (the same physical person box repeats once per trigger).
        # Distinct people that merely overlap are never merged.
        seen: set[object] = set()
        deduped: list[BboxAnnotation] = []
        for kf_id in group.keyframe_ids:
            for b in bboxes_by_keyframe.get(kf_id, []):
                key: object = (
                    ("ph", b.ph_id)
                    if b.ph_id
                    else (
                        "box",
                        round(b.x1, _DEDUP_ROUND),
                        round(b.y1, _DEDUP_ROUND),
                        round(b.x2, _DEDUP_ROUND),
                        round(b.y2, _DEDUP_ROUND),
                    )
                )
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(b)

        views = [_bbox_view(b, group.captured_at, decisions, ranges, pending) for b in deduped]
        frame_w = next((b.frame_width for b in deduped if b.frame_width), 0)
        frame_h = next((b.frame_height for b in deduped if b.frame_height), 0)
        unknown = sum(1 for v in views if not v.effective_identity_id)
        conflict = sum(1 for v in views if v.conflict)
        pending_n = sum(1 for v in views if v.pending_review)
        cards.append(
            PhysicalFrameCard(
                physical_frame_id=fid,
                camera_id=group.camera_id,
                minio_key=group.minio_key,
                captured_at=group.captured_at,
                frame_width=frame_w,
                frame_height=frame_h,
                triggers=group.triggers,
                trigger_reasons=sorted({t.tag_reason for t in group.triggers}),
                bboxes=views,
                unknown_count=unknown,
                conflict_count=conflict,
                pending_review_count=pending_n,
            )
        )
    return cards


def _latest_decision_at_or_before(
    decisions: list[IdentityProvenanceDecision], at: datetime
) -> IdentityProvenanceDecision | None:
    """The most recent decision with ``captured_at <= at`` (per-frame join)."""
    candidates = [d for d in decisions if d.captured_at <= at]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.captured_at)


def _bbox_view(
    b: BboxAnnotation,
    captured_at: datetime,
    decisions: dict[str, list[IdentityProvenanceDecision]],
    ranges: dict[str, list[IdentityRevisionRange]],
    pending: set[str],
) -> BboxIdentityView:
    decision = (
        _latest_decision_at_or_before(decisions.get(b.ph_id, []), captured_at) if b.ph_id else None
    )
    inferred = decision.inferred_identity_id if decision else b.identity_id

    eff_id, eff_authority, revision_id = _effective_identity(ranges.get(b.ph_id, []), captured_at)
    if eff_id is None and revision_id is None:
        # No covering revision range: effective identity is raw inference.
        effective_identity_id = inferred
        authority = _normalize_authority(
            decision.authority if decision else IdentityAuthority.NONE.value
        )
    else:
        effective_identity_id = eff_id
        # eff_authority is a RevisionAuthority ("operator" | "inferred") -- a distinct
        # vocabulary from the decision's IdentityAuthority (F9). Pass it through verbatim;
        # the read-time tolerance below applies only to the decision-authority fallback.
        authority = eff_authority or _normalize_authority(
            decision.authority if decision else IdentityAuthority.NONE.value
        )

    # Operator authority presents as ``Verified`` (no fabricated confidence).
    calibrated = (
        None if authority == "operator" else (decision.top_probability if decision else None)
    )
    conflict_kind = decision.conflict_kind if decision else None
    return BboxIdentityView(
        bbox_id=b.id,
        ph_id=b.ph_id,
        x1=b.x1,
        y1=b.y1,
        x2=b.x2,
        y2=b.y2,
        detection_confidence=b.detection_confidence,
        frame_width=b.frame_width,
        frame_height=b.frame_height,
        inferred_identity_id=inferred,
        effective_identity_id=effective_identity_id,
        authority=authority,
        decision_source=decision.decision_source if decision else "unknown",
        calibrated_confidence=calibrated,
        conflict=bool(conflict_kind),
        conflict_kind=conflict_kind,
        revision_id=revision_id,
        pending_review=b.ph_id in pending if b.ph_id else False,
        decision_id=decision.decision_id if decision else None,
        override_x1=b.override_x1,
        override_y1=b.override_y1,
        override_x2=b.override_x2,
        override_y2=b.override_y2,
    )


def _effective_identity(
    ranges: list[IdentityRevisionRange], at: datetime
) -> tuple[str | None, str | None, str | None]:
    """Resolve effective identity at ``at`` from live revision ranges.

    Mirrors :meth:`IdentityCorrectionRepositoryProtocol.effective_identity`:
    operator ranges win over inferred ranges, latest-created breaks ties.
    Returns ``(identity_id, authority, range_id)`` or ``(None, None, None)``
    when no range covers ``at``.
    """
    covering = [r for r in ranges if r.range_start <= at <= r.range_end]
    if not covering:
        return (None, None, None)
    operator = [r for r in covering if r.authority == "operator"]
    pool = operator if operator else covering
    winner = max(pool, key=lambda r: r.created_at)
    return (winner.effective_identity_id, winner.authority, winner.range_id)


def _frame_matches(card: PhysicalFrameCard, filters: KeyframeReadFilters) -> bool:
    """Frame matches an identity/authority filter when any bbox matches.

    The whole bbox set is still returned for context; the filter only decides
    whether the frame appears on the page.
    """
    eid = filters.effective_identity_id
    if eid is not None and not any(v.effective_identity_id == eid for v in card.bboxes):
        return False
    if filters.explicit_unknown and card.unknown_count == 0:
        return False
    auth = filters.authority
    if auth is not None and not any(v.authority == auth for v in card.bboxes):
        return False
    src = filters.decision_source
    if src is not None and not any(v.decision_source == src for v in card.bboxes):
        return False
    if filters.conflict_only and card.conflict_count == 0:
        return False
    return not (filters.pending_review_only and card.pending_review_count == 0)


class KeyframeReadRepositoryBundle:
    """Adapt the five resource repositories to :class:`KeyframeReadRepositories`.

    Each method delegates to one repository's bounded batch query, keeping the
    page's total query count constant regardless of how many frames or bboxes
    it contains.
    """

    def __init__(
        self,
        *,
        keyframe_repo: KeyframeRepository,
        bbox_repo: BboxAnnotationRepository,
        decision_repo: IdentityDecisionRepositoryProtocol,
        correction_repo: IdentityCorrectionRepositoryProtocol,
        gallery_repo: GalleryRepository,
    ) -> None:
        self._keyframe_repo = keyframe_repo
        self._bbox_repo = bbox_repo
        self._decision_repo = decision_repo
        self._correction_repo = correction_repo
        self._gallery_repo = gallery_repo

    async def list_keyframes_for_read_model(
        self,
        *,
        camera_id: str | None,
        tag_reason: str | None,
        after: datetime | None,
        before: datetime | None,
        limit: int,
    ) -> list[TaggedKeyframe]:
        return await self._keyframe_repo.list_for_read_model(
            camera_id=camera_id,
            tag_reason=tag_reason,
            after=after,
            before=before,
            limit=limit,
        )

    async def get_bbox_annotations_for_keyframes(
        self, keyframe_ids: list[str]
    ) -> list[BboxAnnotation]:
        return await self._bbox_repo.get_bbox_annotations_for_keyframes(keyframe_ids)

    async def decisions_for_phs(
        self, ph_ids: list[str], at_or_before: datetime
    ) -> dict[str, list[IdentityProvenanceDecision]]:
        return await self._decision_repo.decisions_for_phs(ph_ids, at_or_before)

    async def live_ranges_for_phs(
        self, ph_ids: list[str]
    ) -> dict[str, list[IdentityRevisionRange]]:
        return await self._correction_repo.live_ranges_for_phs(ph_ids)

    async def phs_with_pending_reid(self, ph_ids: list[str]) -> set[str]:
        return await self._gallery_repo.phs_with_pending_reid(ph_ids)

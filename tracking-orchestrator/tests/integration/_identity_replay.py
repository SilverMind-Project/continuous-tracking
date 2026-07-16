"""M12 identity replay runner and golden evaluator.

Release-gate machinery for the identity-integrity program. The pieces here turn
a sequence of per-frame resolver decisions into effective-identity records and
score those records against golden expectations.

The single most important correctness property: effective identity is read
through ``IdentityCorrectionRepositoryProtocol.effective_identity(ph_id, at)``,
which overlays live operator revision ranges (operator-wins, latest-first) on
inference. It is **never** read from ``identity_decisions.effective_identity_id``,
which is written ``= inferred`` at decision time and is never updated by an
operator correction. A consumer that reads the column would silently ignore
every correction and report false swaps (see the program architecture notes).

An authoritative swap -- the release gate -- is a bbox whose authority is
operator or qualifying direct ArcFace whose household identity flips between
adjacent authoritative frames without an intervening operator correction
explaining the change. Sub-threshold / ReID / temporal-prior wobble is not a
swap, and an operator correction deliberately changing identity is not a swap.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from app.domain import WorldFrameSnapshot
    from app.storage.corrections import IdentityCorrectionRepositoryProtocol

# decision_source emitted by the resolver when a qualifying calibrated direct
# ArcFace anchor pre-empts the posterior (app/tracking/identity_resolver.py).
ARCFACE_AUTHORITY_SOURCE = "arcface_authority"
# Authority tags carried on a FrameRecord's effective identity.
AUTH_OPERATOR = "operator"
AUTH_ARCFACE = "arcface"
AUTH_INFERRED = "inferred"
AUTH_NONE = ""
UNKNOWN = ""  # empty household identity == Unknown


@dataclass(frozen=True)
class DecisionRow:
    """One per-PH resolver decision for one frame.

    Mirrors the identity-bearing fields of ``WorldFrameSnapshot`` so a runner can
    populate it from a live ``WorldTracker.step`` result or from a synthetic
    fixture. ``inferred_identity_id`` is ``""`` for Unknown. ``authority`` is the
    resolver's bounded ``IdentityAuthority`` vocabulary value (M07/F9) --
    ``"direct_face"`` on a qualifying direct-ArcFace commit, never an identity
    id -- exactly as the resolver writes it.
    """

    ph_id: str
    captured_at: datetime
    frame_index: int
    inferred_identity_id: str = ""
    decision_source: str = ""
    authority: str = ""
    conflict: str = ""
    bbox: tuple[float, float, float, float] | None = None


def decision_row_from_snapshot(snap: WorldFrameSnapshot) -> DecisionRow:
    """Adapt a live ``WorldTracker.step`` snapshot into a runner input row.

    The snapshot carries the resolver's per-frame identity provenance directly
    (``inferred_identity_id``, ``authority``, ``decision_source``, ``conflict``),
    so the real-tracker replay path and the synthetic-fixture path share one
    evaluator.
    """
    bbox = (
        (snap.bbox.x_min, snap.bbox.y_min, snap.bbox.x_max, snap.bbox.y_max)
        if snap.bbox is not None
        else None
    )
    return DecisionRow(
        ph_id=snap.ph_id,
        captured_at=snap.captured_at,
        frame_index=snap.frame_index,
        inferred_identity_id=snap.inferred_identity_id,
        decision_source=snap.decision_source,
        authority=snap.authority,
        conflict=snap.conflict,
        bbox=bbox,
    )


@dataclass(frozen=True)
class FrameRecord:
    """A decision after the live operator-correction overlay is applied."""

    ph_id: str
    captured_at: datetime
    frame_index: int
    inferred_identity_id: str
    effective_identity_id: str
    effective_authority: str
    decision_source: str
    conflict: str
    is_authoritative: bool
    bbox: tuple[float, float, float, float] | None

    def to_jsonl(self) -> str:
        d = asdict(self)
        d["captured_at"] = self.captured_at.isoformat()
        return json.dumps(d, sort_keys=True)


async def build_records(
    rows: Iterable[DecisionRow],
    correction_repo: IdentityCorrectionRepositoryProtocol,
) -> list[FrameRecord]:
    """Overlay live operator corrections onto raw decisions.

    For every row, ``effective_identity(ph_id, captured_at)`` decides the
    effective identity. A covering operator range wins; otherwise the row falls
    back to its raw inferred decision. Reading the correction repository here --
    not a stored decision column -- is what makes the evaluator honour operator
    corrections.
    """
    records: list[FrameRecord] = []
    for row in rows:
        eff_id, range_auth = await correction_repo.effective_identity(row.ph_id, row.captured_at)
        if range_auth == "operator":
            # An operator range is authoritative and overrides inference.
            effective = eff_id or UNKNOWN
            authority = AUTH_OPERATOR
        else:
            # No covering range, or an inferred range. An inferred range only
            # restates inference over a span; it never changes who held
            # authority, so the raw decision still owns it. (record_inferred_range
            # is currently unwired, so range_auth == "inferred" does not occur in
            # the live pipeline; this branch handles it defensively so an inferred
            # range can never mask a direct-ArcFace authority and zero the gate.)
            effective = (
                eff_id
                if (range_auth == "inferred" and eff_id is not None)
                else row.inferred_identity_id
            )
            if row.authority == "direct_face":
                authority = AUTH_ARCFACE
            elif effective:
                authority = AUTH_INFERRED
            else:
                authority = AUTH_NONE
        is_authoritative = authority in (AUTH_OPERATOR, AUTH_ARCFACE) and effective != UNKNOWN
        records.append(
            FrameRecord(
                ph_id=row.ph_id,
                captured_at=row.captured_at,
                frame_index=row.frame_index,
                inferred_identity_id=row.inferred_identity_id,
                effective_identity_id=effective,
                effective_authority=authority,
                decision_source=row.decision_source,
                conflict=row.conflict,
                is_authoritative=is_authoritative,
                bbox=row.bbox,
            )
        )
    return records


@dataclass(frozen=True)
class AuthoritativeSwap:
    """One authoritative household-identity flip on a single PH."""

    ph_id: str
    from_identity: str
    to_identity: str
    at: datetime
    to_authority: str


@dataclass
class EvaluationReport:
    """Aggregate replay verdict. ``authoritative_swaps`` is the release gate."""

    frames: int = 0
    authoritative_swaps: list[AuthoritativeSwap] = field(default_factory=list)
    unknown_frames: int = 0
    unknown_after_known: int = 0
    unknown_durations_s: list[float] = field(default_factory=list)
    distinct_phs: int = 0
    duplicate_active_frames: int = 0
    authoritative_frames: int = 0
    source_attribution_complete: bool = True
    identity_accuracy: float | None = None
    # Count of known identities carried by more than one PH across the replay
    # (an over-segmentation / handoff-fragmentation signal, distinct from the
    # per-instant duplicate_active_frames).
    fragmented_identities: int = 0
    # Deferred: requires golden correction-boundary annotations the synthetic
    # fixtures do not yet carry. Stays None until a golden set provides them.
    correction_boundary_accuracy: float | None = None

    @property
    def swap_count(self) -> int:
        return len(self.authoritative_swaps)

    @property
    def unknown_rate(self) -> float:
        return self.unknown_frames / self.frames if self.frames else 0.0


def _by_ph(records: list[FrameRecord]) -> dict[str, list[FrameRecord]]:
    grouped: dict[str, list[FrameRecord]] = {}
    for rec in records:
        grouped.setdefault(rec.ph_id, []).append(rec)
    for recs in grouped.values():
        recs.sort(key=lambda r: (r.captured_at, r.frame_index))
    return grouped


def detect_authoritative_swaps(records: list[FrameRecord]) -> list[AuthoritativeSwap]:
    """Return every authoritative household-identity flip.

    Walk each PH's authoritative, known frames in capture order. A change from
    one household identity to another is a swap unless the later frame's
    authority is operator (a deliberate, authorised correction). A machine
    (direct-ArcFace) flip -- including one that overrides an operator range -- is
    a swap.
    """
    swaps: list[AuthoritativeSwap] = []
    for ph_id, recs in _by_ph(records).items():
        prev_identity: str | None = None
        for rec in recs:
            if not rec.is_authoritative or rec.effective_identity_id == UNKNOWN:
                continue
            cur = rec.effective_identity_id
            if (
                prev_identity is not None
                and cur != prev_identity
                and rec.effective_authority != AUTH_OPERATOR
            ):
                swaps.append(
                    AuthoritativeSwap(
                        ph_id=ph_id,
                        from_identity=prev_identity,
                        to_identity=cur,
                        at=rec.captured_at,
                        to_authority=rec.effective_authority,
                    )
                )
            prev_identity = cur
    return swaps


def _duplicate_active_frames(records: list[FrameRecord]) -> int:
    """Count capture instants where two distinct PHs hold the same known identity."""
    by_time: dict[datetime, list[FrameRecord]] = {}
    for rec in records:
        by_time.setdefault(rec.captured_at, []).append(rec)
    dup = 0
    for recs in by_time.values():
        seen: dict[str, set[str]] = {}
        for rec in recs:
            if rec.effective_identity_id != UNKNOWN:
                seen.setdefault(rec.effective_identity_id, set()).add(rec.ph_id)
        if any(len(phs) > 1 for phs in seen.values()):
            dup += 1
    return dup


def evaluate(
    records: list[FrameRecord],
    *,
    golden: dict[str, dict[int, str]] | None = None,
) -> EvaluationReport:
    """Score a record stream.

    ``golden`` is an optional ``{ph_id: {frame_index: expected_identity}}`` map.
    When present it yields ``identity_accuracy`` over annotated frames.
    """
    report = EvaluationReport(frames=len(records))
    report.authoritative_swaps = detect_authoritative_swaps(records)
    report.unknown_frames = sum(1 for r in records if r.effective_identity_id == UNKNOWN)
    report.authoritative_frames = sum(1 for r in records if r.is_authoritative)
    report.duplicate_active_frames = _duplicate_active_frames(records)
    report.distinct_phs = len({r.ph_id for r in records})

    # Fragmentation: a known identity carried by more than one PH across the
    # whole replay (the person was split/handed between PHs).
    phs_per_identity: dict[str, set[str]] = {}
    for rec in records:
        if rec.effective_identity_id != UNKNOWN:
            phs_per_identity.setdefault(rec.effective_identity_id, set()).add(rec.ph_id)
    report.fragmented_identities = sum(1 for phs in phs_per_identity.values() if len(phs) > 1)

    # Source attribution: every authoritative/known decision must name a source.
    report.source_attribution_complete = all(
        r.decision_source != "" for r in records if r.effective_identity_id != UNKNOWN
    )

    # Per-PH unknown-after-known transitions and unknown-span durations.
    for recs in _by_ph(records).values():
        prev_known = False
        span_start: datetime | None = None
        prev_at: datetime | None = None
        for rec in recs:
            is_unknown = rec.effective_identity_id == UNKNOWN
            if is_unknown:
                if prev_known:
                    report.unknown_after_known += 1
                if span_start is None:
                    span_start = rec.captured_at
            else:
                if span_start is not None and prev_at is not None:
                    report.unknown_durations_s.append((prev_at - span_start).total_seconds())
                span_start = None
            prev_known = not is_unknown
            prev_at = rec.captured_at
        if span_start is not None and prev_at is not None:
            report.unknown_durations_s.append((prev_at - span_start).total_seconds())

    if golden is not None:
        annotated = 0
        correct = 0
        for rec in records:
            expected = golden.get(rec.ph_id, {}).get(rec.frame_index)
            if expected is None:
                continue
            annotated += 1
            if rec.effective_identity_id == expected:
                correct += 1
        report.identity_accuracy = (correct / annotated) if annotated else None

    return report


def write_jsonl(records: list[FrameRecord], path: str) -> None:
    """Write the per-frame records as JSON lines (machine-readable runner output)."""
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.to_jsonl() + "\n")

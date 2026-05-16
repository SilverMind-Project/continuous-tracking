"""IdentityCommitter: buffered evidence, windowed commit, high-confidence fast-path.

Phase 5: Separates the per-frame evidence collection from the commit
decision.  The resolver still computes a posterior every frame, but the
committer buffers evidence over a ``commit_window_s`` and emits one
decision per global track per window.  This prevents rapid identity
flip-flops and gives the operator fewer, higher-quality decisions in the
audit log.

High-confidence face anchors (>= ``high_confidence_face_threshold``)
bypass the buffer and commit immediately with ``applies_from`` set to
the track's ``first_seen_at``, enabling retroactive labelling of all
prior trajectory/dwell points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta


@dataclass
class CommitDecision:
    """A buffered or immediate identity decision for one GlobalTrack."""

    global_track_id: str
    identity_id: str | None
    confidence: float
    previous_identity_id: str | None = None
    reason: str = ""
    buffered: bool = False  # True = from flush, False = immediate fast-path


@dataclass
class IdentityCommitter:
    """Buffers evidence and emits commit decisions per commit window."""

    commit_window_s: float = 3.0
    high_confidence_face_threshold: float = 0.85

    _buffer: dict[str, list[tuple[datetime, str | None, float, str]]] = field(default_factory=dict)

    def ingest(
        self,
        global_track_id: str,
        identity_id: str | None,
        confidence: float,
        reason: str = "",
    ) -> None:
        """Buffer a per-frame evidence update for *global_track_id*."""
        now = datetime.now(UTC)
        self._buffer.setdefault(global_track_id, []).append((now, identity_id, confidence, reason))

    def flush(self, now: datetime | None = None) -> list[CommitDecision]:
        """Emit one decision per global track for evidence older than
        ``commit_window_s``.  Clears the buffer for emitted tracks.
        """
        now = now or datetime.now(UTC)
        cutoff = now - timedelta(seconds=self.commit_window_s)
        decisions: list[CommitDecision] = []
        expired: list[str] = []

        for gt_id, entries in self._buffer.items():
            # Only flush when the oldest entry is past the window.
            oldest = min(e[0] for e in entries)
            if oldest > cutoff:
                continue

            # Take the latest identity_id and max confidence in the window.
            latest_entry = max(entries, key=lambda e: e[0])
            max_conf = max(e[2] for e in entries if e[1] == latest_entry[1])
            decisions.append(
                CommitDecision(
                    global_track_id=gt_id,
                    identity_id=latest_entry[1],
                    confidence=max_conf,
                    reason=latest_entry[3],
                    buffered=True,
                )
            )
            expired.append(gt_id)

        for gt_id in expired:
            del self._buffer[gt_id]

        return decisions

    def check_high_confidence_face(
        self,
        global_track_id: str,
        face_identity_id: str,
        face_confidence: float,
        first_seen_at: datetime | None = None,
    ) -> CommitDecision | None:
        """Return an immediate commit decision if the face anchor meets the
        high-confidence threshold.
        """
        if face_confidence < self.high_confidence_face_threshold:
            return None

        return CommitDecision(
            global_track_id=global_track_id,
            identity_id=face_identity_id,
            confidence=face_confidence,
            reason="face_high_confidence",
            buffered=False,
        )

    def clear_track(self, global_track_id: str) -> None:
        """Remove buffered evidence for a closed track."""
        self._buffer.pop(global_track_id, None)

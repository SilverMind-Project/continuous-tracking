"""In-memory ring buffer of recent member-bound wearable fixes.

Epoch: rolling time window bounded by ``max_age_s`` (default 30 s) and
``max_per_tag`` fixes per tag. Fixes are maintained in ascending
``measured_at`` order per tag so micro-batched bursts arriving slightly later
than continuous low-rate fixes are still ordered by physical measurement time.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from app.domain import WearableFix

# Allow small forward clock skew between wearable transmitters and camera frames.
_MAX_FUTURE_SKEW_S: float = 2.0


class WearableFixBuffer:
    """Bounded per-tag buffer implementing :class:`WearableFixSource`."""

    def __init__(
        self,
        *,
        max_age_s: float = 30.0,
        max_per_tag: int = 128,
    ) -> None:
        if max_age_s <= 0.0:
            raise ValueError("max_age_s must be positive")
        if max_per_tag < 1:
            raise ValueError("max_per_tag must be >= 1")
        self._max_age_s = max_age_s
        self._max_per_tag = max_per_tag
        self._by_tag: dict[str, list[WearableFix]] = {}

    def add(self, fix: WearableFix) -> None:
        """Insert *fix* into its tag buffer in ascending ``measured_at`` order."""
        entries = self._by_tag.setdefault(fix.tag_id, [])
        # Deduplicate exact replay of (measured_at, epoch_seq) for the same tag.
        for existing in entries:
            if existing.measured_at == fix.measured_at and existing.epoch_seq == fix.epoch_seq:
                return
        # Insert keeping ascending measured_at order.
        idx = len(entries)
        while idx > 0 and entries[idx - 1].measured_at > fix.measured_at:
            idx -= 1
        entries.insert(idx, fix)

        # Bound per-tag length by keeping the newest fixes.
        if len(entries) > self._max_per_tag:
            del entries[: len(entries) - self._max_per_tag]

        # Prune entries older than max_age_s relative to the newest fix on this tag.
        newest = entries[-1].measured_at
        cutoff = newest - timedelta(seconds=self._max_age_s)
        first_valid = 0
        while first_valid < len(entries) and entries[first_valid].measured_at < cutoff:
            first_valid += 1
        if first_valid > 0:
            del entries[:first_valid]

    def recent(self, *, now: datetime, max_age_s: float) -> tuple[WearableFix, ...]:
        """Return fixes with ``measured_at`` within ``[now - max_age_s, now + skew]``."""
        retain_cutoff = now - timedelta(seconds=max(self._max_age_s, max_age_s))
        query_cutoff = now - timedelta(seconds=max_age_s)
        future_limit = now + timedelta(seconds=_MAX_FUTURE_SKEW_S)

        result: list[WearableFix] = []
        empty_tags: list[str] = []
        for tag_id, entries in self._by_tag.items():
            # Evict fixes older than retention cutoff.
            first_keep = 0
            while first_keep < len(entries) and entries[first_keep].measured_at < retain_cutoff:
                first_keep += 1
            if first_keep > 0:
                del entries[:first_keep]
            if not entries:
                empty_tags.append(tag_id)
                continue
            for fix in entries:
                if query_cutoff <= fix.measured_at <= future_limit:
                    result.append(fix)

        for tag_id in empty_tags:
            del self._by_tag[tag_id]

        result.sort(key=lambda f: (f.measured_at, f.tag_id, f.epoch_seq))
        return tuple(result)

    def purge_identity(self, identity_id: str) -> int:
        """Drop all buffered fixes bound to *identity_id* and return removed count."""
        removed = 0
        empty_tags: list[str] = []
        for tag_id, entries in self._by_tag.items():
            kept = [f for f in entries if f.member_id != identity_id]
            removed += len(entries) - len(kept)
            if kept:
                self._by_tag[tag_id] = kept
            else:
                empty_tags.append(tag_id)
        for tag_id in empty_tags:
            del self._by_tag[tag_id]
        return removed

    def recent_count(self, *, now: datetime, max_age_s: float) -> int:
        """Return the number of currently buffered fixes within *max_age_s* of *now*."""
        return len(self.recent(now=now, max_age_s=max_age_s))

    def clear(self) -> None:
        """Clear all buffered fixes."""
        self._by_tag.clear()

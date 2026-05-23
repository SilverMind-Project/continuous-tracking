"""Gallery governance: entry state machine and promotion logic.

Gallery entries transition through explicit states:

- ``unlabeled`` — Body embedding, no identity label (default on creation).
- ``quarantined_identity`` — Tentative identity label, not yet confirmed.
- ``promoted_identity`` — Identity confirmed by direct face or operator.
- ``operator_verified`` — Manually verified, immune to auto-revision.
- ``rejected`` — Contaminated entry, excluded from gallery search.

Only promoted or operator_verified entries may vote for a known identity
with full weight in ReID likelihood scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from structlog import get_logger

from ...storage.base import GalleryRepository

logger = get_logger(__name__)

GalleryEntryState = Literal[
    "unlabeled",
    "quarantined_identity",
    "promoted_identity",
    "operator_verified",
    "rejected",
]


@dataclass(frozen=True)
class GalleryGovernanceConfig:
    """Configuration for gallery governance."""

    # Seconds a quarantined entry must survive without revision before promotion.
    quarantine_duration_s: float = 30.0

    # Weight multiplier for quarantined entries in ReID scoring.
    quarantined_weight: float = 0.3

    # Weight multiplier for unlabeled entries (track continuity only).
    unlabeled_weight: float = 0.0  # cannot vote for known identities


@dataclass
class GalleryGovernanceService:
    """Manages gallery entry state transitions.

    Wraps a ``GalleryRepository`` to enforce governance rules on every
    state change.
    """

    _gallery: GalleryRepository
    _config: GalleryGovernanceConfig = field(default_factory=GalleryGovernanceConfig)

    async def promote_to_quarantined(
        self,
        tracklet_ids: set[str],
        identity_id: str,
        evidence_source: str = "direct_face",
    ) -> None:
        """Quarantine gallery entries for *tracklet_ids* under *identity_id*.

        Only ``direct_face`` or ``operator`` evidence may promote to quarantined.
        Association hints and temporal prior are not sufficient.
        """
        if evidence_source not in ("direct_face", "operator"):
            logger.debug(
                "gallery_quarantine_skipped_insufficient_source",
                identity_id=identity_id,
                source=evidence_source,
                tracklet_count=len(tracklet_ids),
            )
            return

        await self._gallery.update_identity_for_tracklets(
            tracklet_ids=tracklet_ids,
            identity_id=identity_id,
        )
        logger.info(
            "gallery_entries_quarantined",
            identity_id=identity_id,
            tracklet_count=len(tracklet_ids),
            source=evidence_source,
        )

    async def promote_to_verified(
        self,
        tracklet_ids: set[str],
        identity_id: str,
    ) -> None:
        """Operator-verify gallery entries.  Immune to automatic revision."""
        await self._gallery.update_identity_for_tracklets(
            tracklet_ids=tracklet_ids,
            identity_id=identity_id,
        )

    async def reject_entries(
        self,
        tracklet_ids: set[str],
        reason: str = "revision",
    ) -> None:
        """Reject gallery entries (contaminated by false identity commit)."""
        # Clear the identity label — rejected entries return to unlabeled state.
        await self._gallery.update_identity_for_tracklets(
            tracklet_ids=tracklet_ids,
            identity_id="",
        )
        logger.info(
            "gallery_entries_rejected",
            tracklet_count=len(tracklet_ids),
            reason=reason,
        )

    def get_voting_weight(self, entry_state: GalleryEntryState) -> float:
        """Return the voting weight for a gallery entry in a given state."""
        weights = {
            "unlabeled": self._config.unlabeled_weight,
            "quarantined_identity": self._config.quarantined_weight,
            "promoted_identity": 1.0,
            "operator_verified": 1.0,
            "rejected": 0.0,
        }
        return weights.get(entry_state, 0.0)

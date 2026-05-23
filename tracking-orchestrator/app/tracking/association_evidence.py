"""Typed association evidence consumed by the cross-camera solver.

AssociationCandidate and AssociationDecision replace ad-hoc scoring with
inspectable, testable evidence that carries an explicit reject reason
when a candidate pair is not viable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

AssociationAction = Literal["extend", "create", "merge_global_tracks", "reject"]


@dataclass(frozen=True)
class AssociationCandidate:
    """A candidate pair for associating a tracklet with a global track.

    Populated by the candidate-generation phase.  The solver consumes
    these and produces ``AssociationDecision`` records.
    """

    source_tracklet_id: str
    target_global_track_id: str
    appearance_sim: float | None
    floor_distance_m: float | None
    temporal_feasible: bool
    overlap_group_id: str | None
    identity_conflict: bool = False
    do_not_fuse: bool = False
    score: float = 0.0
    reject_reason: str | None = None

    @property
    def is_rejected(self) -> bool:
        return self.reject_reason is not None


@dataclass(frozen=True)
class AssociationDecision:
    """The solver's decision for one tracklet.

    *action* is one of:
    - ``"extend"`` — attach the tracklet to an existing global track.
    - ``"create"`` — create a new global track for this tracklet.
    - ``"merge_global_tracks"`` — merge two existing global tracks.
    - ``"reject"`` — no legal association exists; create a new track.
    """

    tracklet_id: str
    global_track_id: str | None  # None for "create" and "reject"
    action: AssociationAction
    reason: str
    score: float | None = None


@dataclass
class AssociationResult:
    """Complete output of one association frame."""

    decisions: list[AssociationDecision] = field(default_factory=list)
    candidates_generated: int = 0
    candidates_rejected: dict[str, int] = field(default_factory=dict)
    score_histogram: list[float] = field(default_factory=list)

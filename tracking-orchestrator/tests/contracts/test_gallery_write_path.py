"""M04: ReIDCandidateStage is the only pipeline write path into reid_gallery.

Static contract test (no fixture, no live DB): greps the source of every
module under ``app/tracking/world/`` and ``app/pipeline/`` for the mutation
call names, asserting only the sanctioned call sites remain.
``app/routers/gallery.py`` (manual enrollment) is the one approved exception.
"""

from __future__ import annotations

from pathlib import Path

_APP = Path(__file__).parents[2] / "app"

# Call names that mutate reid_gallery directly. A module outside the
# approved list must never call these -- gallery writes are governed either
# through ReIDCandidateStage.create_review_candidate (creation) or
# ReIDReviewService's apply_review_action/compensate_review (M09 review).
_MUTATION_CALLS = ("upsert_gallery_entry", "create_review_candidate")

# The only approved call site within the scanned roots (tracking/world,
# pipeline): the one pipeline stage that calls create_review_candidate.
# app/routers/gallery.py (manual crop enrollment, an operator action) calls
# upsert_gallery_entry but lives outside both scanned roots.
_APPROVED_FILES = {
    "pipeline/stages/reid_candidates.py",
}


def _iter_py_files(*roots: str) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        files.extend((_APP / root).rglob("*.py"))
    return files


def test_no_unsanctioned_gallery_writes_in_tracking_or_pipeline() -> None:
    offenders: list[str] = []
    for path in _iter_py_files("tracking/world", "pipeline"):
        rel = str(path.relative_to(_APP))
        if rel in _APPROVED_FILES:
            continue
        text = path.read_text()
        for call in _MUTATION_CALLS:
            # Match a call, not a def (the Protocol/ABC declarations live in
            # the approved storage files, not under tracking/world or pipeline).
            if f".{call}(" in text or f"def {call}(" in text:
                offenders.append(f"{rel}: {call}")
    assert offenders == [], f"Unsanctioned reid_gallery write call sites: {offenders}"


def test_tracker_has_no_gallery_seeding_method() -> None:
    """Regression guard for F3: the deleted `_seed_multiview_gallery` (which
    never checked face_anchor.person_id == identity_id) must not reappear."""
    tracker_src = (_APP / "tracking/world/tracker.py").read_text()
    assert "_seed_multiview_gallery" not in tracker_src
    assert "upsert_gallery_entry" not in tracker_src

"""FaceTemplateBuffer: per-PH best faces, quality floor, age eviction, outliers, resets, tickets."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from app.inference.face_id_client import FaceQualityReading
from app.tracking.identity.face_template_buffer import (
    BufferAdd,
    BufferedFace,
    FaceTemplateBuffer,
    FaceTemplatePolicy,
)

_T0 = datetime(2026, 11, 12, 9, 0, tzinfo=UTC)
_POLICY = FaceTemplatePolicy(mode="shadow", top_k=4, min_faces=3)


def _unit(seed: int) -> np.ndarray:
    v = np.random.RandomState(seed).randn(512)
    return (v / np.linalg.norm(v)).astype(np.float32)


_ALICE = _unit(1)
_BOB = _unit(2)


def _alice_face(seed: int) -> np.ndarray:
    """One glimpse of alice: her direction plus noise (cosine about 0.7 to her)."""
    v = _ALICE + _unit(100 + seed)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _face(
    n: int,
    score: float,
    *,
    seconds: float = 0.0,
    embedding: np.ndarray | None = None,
    recognized_as: str | None = None,
) -> BufferedFace:
    return BufferedFace(
        detection_id=f"det-{n}",
        camera_id="cam-1",
        captured_at=_T0 + timedelta(seconds=seconds),
        embedding=embedding if embedding is not None else _alice_face(n),
        quality=FaceQualityReading(
            score=score, resolution=0.4, sharpness=0.5, detection=0.9, pitch=1.0, version="v"
        ),
        det_score=0.8,
        yaw_deg=10.0,
        single_frame_person_id=recognized_as,
        embedding_model_version="buffalo_l",
    )


def _ids(buffer: FaceTemplateBuffer, ph_id: str = "ph-1") -> list[str]:
    return [f.detection_id for f in buffer.snapshot(ph_id)]


def test_keeps_the_top_k_faces_by_quality() -> None:
    buffer = FaceTemplateBuffer(_POLICY)
    for n, score in enumerate([0.3, 0.9, 0.5, 0.7, 0.2, 0.8]):
        buffer.add("ph-1", _face(n, score))

    assert _ids(buffer) == ["det-1", "det-5", "det-3", "det-2"]
    assert buffer.add("ph-1", _face(9, 0.25)) is BufferAdd.NOT_KEPT
    assert buffer.add("ph-1", _face(10, 0.95)) is BufferAdd.KEPT
    assert _ids(buffer)[0] == "det-10"


def test_faces_below_the_quality_floor_never_enter() -> None:
    buffer = FaceTemplateBuffer(_POLICY)
    assert buffer.add("ph-1", _face(1, 0.14)) is BufferAdd.LOW_QUALITY
    assert buffer.add("ph-1", _face(2, 0.15)) is BufferAdd.KEPT
    assert _ids(buffer) == ["det-2"]


def test_faces_older_than_max_age_leave_and_an_empty_ph_is_forgotten() -> None:
    buffer = FaceTemplateBuffer(_POLICY)
    buffer.add("ph-1", _face(1, 0.9, seconds=0.0))
    buffer.add("ph-1", _face(2, 0.5, seconds=10.0))

    buffer.evict_older_than(_T0 + timedelta(seconds=14.0))
    assert _ids(buffer) == ["det-1", "det-2"]
    buffer.evict_older_than(_T0 + timedelta(seconds=16.0))
    assert _ids(buffer) == ["det-2"]
    buffer.evict_older_than(_T0 + timedelta(seconds=26.0))
    assert buffer.ph_ids() == ()


def test_a_face_unlike_the_buffer_is_rejected_as_an_outlier() -> None:
    buffer = FaceTemplateBuffer(_POLICY)
    buffer.add("ph-1", _face(1, 0.6))
    buffer.add("ph-1", _face(2, 0.6))

    # Bob's face mixed into alice's PH by an association swap: nearly orthogonal to her faces.
    assert buffer.add("ph-1", _face(3, 0.99, embedding=_BOB)) is BufferAdd.OUTLIER
    assert _ids(buffer) == ["det-1", "det-2"]


def test_an_identity_change_resets_but_unknown_in_between_does_not() -> None:
    buffer = FaceTemplateBuffer(_POLICY)
    buffer.add("ph-1", _face(1, 0.6))
    assert buffer.observe_identity("ph-1", "alice", _T0) is False
    assert buffer.observe_identity("ph-1", None, _T0) is False
    assert buffer.observe_identity("ph-1", "alice", _T0) is False
    assert _ids(buffer) == ["det-1"]

    assert buffer.observe_identity("ph-1", "bob", _T0) is True
    assert _ids(buffer) == []


def test_a_reset_makes_every_earlier_ticket_stale() -> None:
    buffer = FaceTemplateBuffer(_POLICY)
    before = buffer.ticket()
    buffer.reset("ph-1", _T0)
    after = buffer.ticket()

    assert buffer.is_current("ph-1", before) is False
    assert buffer.is_current("ph-1", after) is True
    assert buffer.is_current("ph-2", before) is True  # other PHs are unaffected


def test_reset_identity_drops_phs_committed_to_it_or_holding_its_faces() -> None:
    buffer = FaceTemplateBuffer(_POLICY)
    buffer.add("ph-committed", _face(1, 0.6))
    buffer.observe_identity("ph-committed", "alice", _T0)
    buffer.add("ph-face", _face(2, 0.6, recognized_as="alice"))
    buffer.add("ph-other", _face(3, 0.6, recognized_as="bob"))

    assert buffer.reset_identity("alice", _T0) == ["ph-committed", "ph-face"]
    assert buffer.ph_ids() == ("ph-other",)


def test_a_trigger_needs_min_faces_and_respects_the_cooldown() -> None:
    buffer = FaceTemplateBuffer(_POLICY)
    buffer.add("ph-1", _face(1, 0.6))
    buffer.add("ph-1", _face(2, 0.6))
    assert buffer.should_trigger("ph-1", _T0) is False
    buffer.add("ph-1", _face(3, 0.6))

    assert buffer.should_trigger("ph-1", _T0) is True
    assert buffer.should_trigger("ph-1", _T0 + timedelta(seconds=1.9)) is False
    assert buffer.should_trigger("ph-1", _T0 + timedelta(seconds=2.0)) is True


def test_a_closed_and_revived_ph_keeps_its_faces_within_max_age() -> None:
    # Closing a PH is not a reset: only age, identity change, and operator edits clear faces.
    buffer = FaceTemplateBuffer(_POLICY)
    buffer.add("ph-1", _face(1, 0.6, seconds=0.0))
    buffer.evict_older_than(_T0 + timedelta(seconds=8.0))  # rounds while the PH is closed
    buffer.add("ph-1", _face(2, 0.6, seconds=9.0))  # revived with the same ph_id

    assert _ids(buffer) == ["det-2", "det-1"]


def test_embeddings_never_reach_repr() -> None:
    assert "embedding=" not in repr(_face(1, 0.6))


@pytest.mark.parametrize(
    "changes",
    [
        {"mode": "on"},
        {"min_faces": 1},
        {"min_faces": 5, "top_k": 4},
        {"top_k": 17},
        {"min_face_quality": 1.5},
        {"max_age_s": 0.0},
        {"max_concurrency": 4, "max_in_flight": 2},
    ],
)
def test_policy_validation(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="face_templates"):
        FaceTemplatePolicy(**changes)  # type: ignore[arg-type]

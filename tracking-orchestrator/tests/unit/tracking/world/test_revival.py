"""Table-driven tests for select_revival_candidate (same-camera + cross-camera)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.domain import (
    BoundingBox,
    CameraTopologyEdge,
    FaceAnchor,
    FloorPoint,
    OrientationBin,
    PersonHypothesis,
    ViewPrototype,
    WorldObservation,
)
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.revival import select_revival_candidate

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)


def _obs(
    *,
    camera_id: str = "cam-a",
    floor_x: int = 5000,
    floor_y: int = 3000,
    embedding: list[float] | None = None,
    face_anchor: FaceAnchor | None = None,
    calibrated: bool = True,
) -> WorldObservation:
    return WorldObservation(
        camera_id=camera_id,
        frame_index=0,
        captured_at=NOW,
        floor_point=FloorPoint(x_mm=floor_x, y_mm=floor_y, calibrated=calibrated),
        bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=200),
        embedding=embedding or [0.1] * 768,
        detection_confidence=0.9,
        face_anchor=face_anchor,
        detection_id="det-1",
        quality=0.7,
    )


_DEFAULT_CLOSED_AT = NOW - timedelta(seconds=3)


def _closed_ph(
    *,
    ph_id: str = "ph-closed-1",
    camera_id: str = "cam-a",
    state_x: float = 5.0,
    state_y: float = 3.0,
    closed_at: datetime | None = _DEFAULT_CLOSED_AT,
    gallery_mean: list[float] | None = None,
    identity_id: str | None = "alice",
    obs_count: int = 10,
    view_prototypes: tuple[ViewPrototype, ...] = (),
) -> PersonHypothesis:
    # Use a sentinel default so explicit None (not closed) is preserved.
    actual_closed_at: datetime | None = (
        _DEFAULT_CLOSED_AT if closed_at is _DEFAULT_CLOSED_AT else closed_at
    )
    return PersonHypothesis(
        ph_id=ph_id,
        state_mean=(state_x, state_y, 0.0, 0.0),
        state_cov=tuple([0.1] * 16),
        born_at=NOW - timedelta(seconds=30),
        last_seen_at=NOW - timedelta(seconds=5),
        last_seen_camera=camera_id,
        observation_count=obs_count,
        current_identity_id=identity_id,
        current_identity_committed_at=NOW - timedelta(seconds=10),
        gallery_mean=gallery_mean or [0.1] * 768,
        active_cameras=frozenset([camera_id]),
        closed_at=actual_closed_at,
        mean_quality=0.7,
        view_prototypes=view_prototypes,
    )


def _config(**overrides: object) -> WorldTrackerConfig:
    kw: dict[str, object] = {
        "enable_ph_revival": True,
        "revive_max_age_s": 30.0,
        "revive_max_distance_m": 2.0,
        "revive_appearance_min_sim": 0.55,
        "face_conflict_threshold": 0.70,
    }
    kw.update(overrides)
    return WorldTrackerConfig(**kw)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Positive tests
# ---------------------------------------------------------------------------


def test_same_camera_within_all_gates_selected() -> None:
    """A same-camera closed PH within all gates is selected."""
    closed = _closed_ph()
    obs = _obs()
    result = select_revival_candidate(obs, [closed], NOW, _config())
    assert result is not None
    assert result.ph_id == "ph-closed-1"


def test_two_candidates_highest_similarity_chosen() -> None:
    """Among two passing candidates, the one with highest similarity wins."""
    # Closed PH 1 has embedding close to observation.
    closed1 = _closed_ph(
        ph_id="ph-1",
        gallery_mean=[0.099] * 768,  # close but not identical
    )
    # Closed PH 2 has embedding farther from observation.
    closed2 = _closed_ph(
        ph_id="ph-2",
        gallery_mean=[0.05] * 768,  # farther
    )
    obs = _obs(embedding=[0.1] * 768)
    result = select_revival_candidate(obs, [closed1, closed2], NOW, _config())
    assert result is not None
    assert result.ph_id == "ph-1"


def test_missing_embeddings_falls_back_to_space_time() -> None:
    """When either embedding is missing, select on space and time only."""
    closed = _closed_ph(gallery_mean=None)
    obs = _obs(embedding=None)
    result = select_revival_candidate(obs, [closed], NOW, _config())
    assert result is not None
    assert result.ph_id == "ph-closed-1"


# ---------------------------------------------------------------------------
# Negative tests
# ---------------------------------------------------------------------------


def test_cross_camera_rejected_when_disabled() -> None:
    """A different-camera closed PH is rejected when cross-camera revival is disabled."""
    closed = _closed_ph(camera_id="cam-b")
    obs = _obs(camera_id="cam-a")
    result = select_revival_candidate(obs, [closed], NOW, _config())
    assert result is None


def test_too_old_rejected() -> None:
    """Closed PH older than revive_max_age_s is rejected."""
    closed = _closed_ph(closed_at=NOW - timedelta(seconds=60))
    obs = _obs()
    result = select_revival_candidate(obs, [closed], NOW, _config())
    assert result is None


def test_too_far_rejected() -> None:
    """Closed PH beyond revive_max_distance_m is rejected."""
    closed = _closed_ph(state_x=20.0, state_y=30.0)  # far from obs at (5,3)
    obs = _obs()
    result = select_revival_candidate(obs, [closed], NOW, _config())
    assert result is None


def test_appearance_below_threshold_rejected() -> None:
    """Closed PH with cosine similarity below threshold is rejected."""
    # Build orthogonal embeddings (cosine ≈ 0).
    closed = _closed_ph(gallery_mean=[1.0] + [0.0] * 767)
    obs = _obs(embedding=[0.0] + [1.0] * 767)
    result = select_revival_candidate(obs, [closed], NOW, _config())
    assert result is None


def test_face_conflict_rejects_revival() -> None:
    """A face anchor naming a different identity at high confidence rejects."""
    closed = _closed_ph(identity_id="alice")
    obs = _obs(
        face_anchor=FaceAnchor(
            person_id="bob",
            confidence=0.85,
            quality=0.9,
            detection_id="det-1",
        )
    )
    result = select_revival_candidate(obs, [closed], NOW, _config())
    assert result is None


def test_face_conflict_below_threshold_allows_revival() -> None:
    """A face anchor below face_conflict_threshold does not block revival."""
    closed = _closed_ph(identity_id="alice")
    obs = _obs(
        face_anchor=FaceAnchor(
            person_id="bob",
            confidence=0.60,  # below default 0.70 threshold
            quality=0.9,
            detection_id="det-1",
        )
    )
    result = select_revival_candidate(obs, [closed], NOW, _config())
    assert result is not None


def test_candidate_face_does_not_block_revival() -> None:
    """A candidate (grey-zone) face anchor must NOT block revival.

    Only recognized anchors assert identity strongly enough to prevent revival.
    Candidate and unrecognized anchors are weak corroborating evidence.
    """
    closed = _closed_ph(identity_id="alice")
    obs = _obs(
        face_anchor=FaceAnchor(
            person_id="bob",
            confidence=0.85,
            quality=0.9,
            detection_id="det-1",
            recognition_state="candidate",  # grey-zone, not a hard assertion
        )
    )
    result = select_revival_candidate(obs, [closed], NOW, _config())
    assert result is not None, "Candidate face must not block revival"


def test_unrecognized_face_does_not_block_revival() -> None:
    """An unrecognized face anchor (face present, below unknown_threshold)
    must NOT block revival.
    """
    closed = _closed_ph(identity_id="alice")
    obs = _obs(
        face_anchor=FaceAnchor(
            person_id="unknown",
            confidence=0.70,
            quality=0.9,
            detection_id="det-1",
            recognition_state="unrecognized",
        )
    )
    result = select_revival_candidate(obs, [closed], NOW, _config())
    assert result is not None, "Unrecognized face must not block revival"


def test_empty_closed_list_returns_none() -> None:
    """Empty closed PH list returns None."""
    result = select_revival_candidate(_obs(), [], NOW, _config())
    assert result is None


def test_no_closed_at_returns_none() -> None:
    """A 'closed' PH with closed_at=None is skipped."""
    closed = _closed_ph(closed_at=None)  # not actually closed
    result = select_revival_candidate(_obs(), [closed], NOW, _config())
    assert result is None


# ---------------------------------------------------------------------------
# Cross-camera revival tests
# ---------------------------------------------------------------------------


def _topology_edge(
    from_camera: str = "cam-b",
    to_camera: str = "cam-a",
    obs_count: int = 10,
    mean_s: float = 3.0,
    var_s2: float = 1.0,
) -> CameraTopologyEdge:
    return CameraTopologyEdge(
        from_camera=from_camera,
        to_camera=to_camera,
        observation_count=obs_count,
        mean_transit_s=mean_s,
        variance_transit_s2=var_s2,
    )


def test_cross_camera_accepted_when_enabled() -> None:
    """A different-camera PH can be revived when cross-camera is enabled."""
    closed = _closed_ph(camera_id="cam-b", closed_at=NOW - timedelta(seconds=3))
    obs = _obs(camera_id="cam-a")
    edge = _topology_edge()
    cfg = _config()
    result = select_revival_candidate(
        obs,
        [closed],
        NOW,
        cfg,
        enable_cross_camera=True,
        topology_edges=[edge],
    )
    assert result is not None
    assert result.ph_id == "ph-closed-1"


def test_cross_camera_implausible_transit_rejected() -> None:
    """Cross-camera revival with implausible transit time is rejected."""
    # Edge has mean 3s, but closed_at was 100s ago → way outside.
    closed = _closed_ph(camera_id="cam-b", closed_at=NOW - timedelta(seconds=100))
    obs = _obs(camera_id="cam-a")
    edge = _topology_edge(mean_s=3.0, var_s2=0.5)
    cfg = _config(cross_camera_min_plausibility=0.05)
    result = select_revival_candidate(
        obs,
        [closed],
        NOW,
        cfg,
        enable_cross_camera=True,
        topology_edges=[edge],
    )
    assert result is None


def test_cross_camera_low_appearance_rejected() -> None:
    """Cross-camera revival with low appearance similarity is rejected."""
    closed = _closed_ph(
        camera_id="cam-b",
        gallery_mean=[1.0] + [0.0] * 767,
        view_prototypes=(
            ViewPrototype(
                orientation=OrientationBin.FRONT,
                embedding=(1.0,) + (0.0,) * 767,
                count=5,
            ),
        ),
    )
    obs = _obs(camera_id="cam-a", embedding=[0.0] + [1.0] * 767)
    edge = _topology_edge()
    cfg = _config(cross_camera_revive_appearance_min_sim=0.60)
    result = select_revival_candidate(
        obs,
        [closed],
        NOW,
        cfg,
        enable_cross_camera=True,
        topology_edges=[edge],
    )
    assert result is None


def test_cross_camera_view_prototypes_used() -> None:
    """Cross-camera revival uses max-over-view-prototypes for appearance."""
    closed = _closed_ph(
        camera_id="cam-b",
        view_prototypes=(
            ViewPrototype(
                orientation=OrientationBin.BACK,
                embedding=(0.1,) * 768,  # high sim to obs
                count=5,
            ),
        ),
    )
    obs = _obs(camera_id="cam-a", embedding=[0.1] * 768)
    edge = _topology_edge()
    result = select_revival_candidate(
        obs,
        [closed],
        NOW,
        _config(cross_camera_revive_appearance_min_sim=0.60),
        enable_cross_camera=True,
        topology_edges=[edge],
    )
    assert result is not None


def test_cross_camera_face_conflict_blocked() -> None:
    """Cross-camera revival is blocked by face conflict (same as same-camera)."""
    closed = _closed_ph(camera_id="cam-b", identity_id="alice")
    obs = _obs(
        camera_id="cam-a",
        face_anchor=FaceAnchor(
            person_id="bob",
            confidence=0.85,
            quality=0.9,
            detection_id="det-1",
        ),
    )
    edge = _topology_edge()
    result = select_revival_candidate(
        obs,
        [closed],
        NOW,
        _config(),
        enable_cross_camera=True,
        topology_edges=[edge],
    )
    assert result is None


def test_same_camera_unchanged_with_cross_camera_enabled() -> None:
    """Same-camera behavior is unchanged when cross-camera flag is on."""
    closed = _closed_ph(camera_id="cam-a")
    obs = _obs(camera_id="cam-a")
    result = select_revival_candidate(
        obs,
        [closed],
        NOW,
        _config(),
        enable_cross_camera=True,
        topology_edges=[],
    )
    assert result is not None
    assert result.ph_id == "ph-closed-1"

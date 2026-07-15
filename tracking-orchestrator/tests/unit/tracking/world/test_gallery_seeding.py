"""Multi-view gallery seeding through WorldTracker.step.

Locks the ordering fix: seeding runs AFTER identity resolution (step 9) using
the COMMITTED identity, so the frame on which a face first commits actually
seeds the gallery. The previous code seeded inside step 4 with the
pre-resolution identity (None on the commit frame), so the gallery never
seeded in practice.

Pure unit test: all-InMemory repos, no Postgres, no Triton.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.domain import (
    BoundingBox,
    FaceAnchor,
    FloorPoint,
    GalleryEmbedding,
    Identity,
    WorldObservation,
)
from app.storage.base import (
    InMemoryGalleryRepository,
    InMemoryPHRepository,
    InMemoryWorldObservationRepository,
)
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig
from app.tracking.orientation import OrientationBin
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.tracker import WorldTracker

BASE_TIME = datetime(2026, 6, 2, 9, 0, 0, tzinfo=UTC)
_EMB = tuple(0.05 * (i % 7) for i in range(768))
# Two orthogonal body embeddings (cosine 0): a populated gallery for one must
# never resolve onto the other via the always-on baseline ReID query.
_GRANDMA_BODY = tuple(1.0 if i < 384 else 0.0 for i in range(768))
_STRANGER_BODY = tuple(0.0 if i < 384 else 1.0 for i in range(768))


async def _make_tracker(
    *, multiview: bool = False
) -> tuple[WorldTracker, InMemoryGalleryRepository]:
    gallery = InMemoryGalleryRepository()
    await gallery.upsert_identity(
        Identity(identity_id="grandma", display_name="grandma", enrolled_at=BASE_TIME)
    )
    resolver = IdentityResolver(
        gallery_repo=gallery,
        config=ResolverConfig(enable_multiview_gallery=multiview),
    )
    tracker = WorldTracker(
        ph_repo=InMemoryPHRepository(),
        obs_repo=InMemoryWorldObservationRepository(),
        config=WorldTrackerConfig(enable_multiview_association=multiview),
        identity_resolver=resolver,
        gallery_repo=gallery,
    )
    return tracker, gallery


async def _seed_grandma_body(gallery: InMemoryGalleryRepository) -> None:
    """Pre-seed grandma's gallery with FRONT+BACK body prototypes.

    state="operator_verified" so the resolver's multiview search_similar
    query (verified-only by default, M03) can see these prototypes.
    """
    for i, orient in enumerate((OrientationBin.FRONT, OrientationBin.BACK)):
        await gallery.upsert_gallery_entry(
            GalleryEmbedding(
                gallery_entry_id=f"g{i}",
                identity_id="grandma",
                embedding=_GRANDMA_BODY,
                seen_at=BASE_TIME,
                quality=0.9,
                face_confirmed=True,
                camera_id="cam01",
                orientation=int(orient),
                state="operator_verified",
            )
        )


def _body_obs(
    *, detection_id: str, embedding: tuple[float, ...], frame_index: int
) -> WorldObservation:
    """A no-face observation carrying only a body embedding (drives ReID)."""
    return WorldObservation(
        camera_id="cam01",
        frame_index=frame_index,
        captured_at=BASE_TIME,
        floor_point=FloorPoint(x_mm=1000.0, y_mm=1000.0, calibrated=True),
        bbox=BoundingBox(x_min=0.4, y_min=0.3, x_max=0.5, y_max=0.7),
        embedding=embedding,
        detection_confidence=0.9,
        height_estimate_m=1.65,
        face_anchor=None,
        detection_id=detection_id,
        quality=0.8,
        orientation=OrientationBin.FRONT,
        orientation_confidence=0.9,
    )


def _observation(
    *,
    detection_id: str,
    recognition_state: str,
    confidence: float,
    orientation: OrientationBin,
    orientation_confidence: float,
) -> WorldObservation:
    face = FaceAnchor(
        person_id="grandma",
        confidence=confidence,
        quality=0.9,
        detection_id=detection_id,
        camera_id="cam01",
        captured_at=BASE_TIME,
        recognition_state=recognition_state,
        similarity=confidence,
        yaw_deg=5.0,
    )
    return WorldObservation(
        camera_id="cam01",
        frame_index=0,
        captured_at=BASE_TIME,
        floor_point=FloorPoint(x_mm=1000.0, y_mm=1000.0, calibrated=True),
        bbox=BoundingBox(x_min=0.4, y_min=0.3, x_max=0.5, y_max=0.7),
        embedding=_EMB,
        detection_confidence=0.9,
        height_estimate_m=1.65,
        face_anchor=face,
        detection_id=detection_id,
        quality=0.8,
        orientation=orientation,
        orientation_confidence=orientation_confidence,
    )


@pytest.mark.asyncio
async def test_recognized_face_seeds_gallery_with_committed_identity() -> None:
    """A recognized face that commits grandma seeds a gallery entry tagged with
    the observation orientation, even though identity commits in step 9 after
    the (former) step-4 seeding point."""
    tracker, gallery = await _make_tracker()

    obs = _observation(
        detection_id="d1",
        recognition_state="recognized",
        confidence=0.95,
        orientation=OrientationBin.FRONT,
        orientation_confidence=0.9,
    )
    result = await tracker.step(observations=[obs], now=BASE_TIME, face_anchors=[obs.face_anchor])

    # The PH committed to grandma this frame.
    assert any(d.identity_id == "grandma" for d in result.identity_decisions)

    entries = await gallery.list_gallery_entries(
        identity_id="grandma",
        active_only=False,
        # Diagnostic read of the raw seed write (_seed_multiview_gallery leaves
        # entries pending_review by default -- F4/M04), not a resolver vote.
        states=None,
    )
    assert len(entries) == 1
    assert entries[0].orientation == int(OrientationBin.FRONT)
    assert entries[0].face_confirmed is True


@pytest.mark.asyncio
async def test_candidate_face_does_not_seed_gallery() -> None:
    """A grey-zone candidate face must never seed the shared gallery (poisoning
    guard): seeding requires recognition_state == 'recognized'."""
    tracker, gallery = await _make_tracker()

    obs = _observation(
        detection_id="d1",
        recognition_state="candidate",
        confidence=0.33,
        orientation=OrientationBin.FRONT,
        orientation_confidence=0.9,
    )
    await tracker.step(observations=[obs], now=BASE_TIME, face_anchors=[obs.face_anchor])

    entries = await gallery.list_gallery_entries(
        identity_id="grandma",
        active_only=False,
        # Diagnostic read of the raw seed write (_seed_multiview_gallery leaves
        # entries pending_review by default -- F4/M04), not a resolver vote.
        states=None,
    )
    assert entries == []


@pytest.mark.asyncio
async def test_multiview_gallery_does_not_leak_identity_to_stranger() -> None:
    """Clinical guardrail with the multiview gallery query live. With only grandma
    enrolled and a stranger whose body embedding is
    orthogonal to grandma's prototypes (similarity ~0), the resolver must keep
    the stranger UNKNOWN. Before the UNKNOWN-complement fix this leaked: the
    single-identity likelihood normalized to grandma=1.0 and committed."""
    tracker, gallery = await _make_tracker(multiview=True)
    await _seed_grandma_body(gallery)

    for k in range(6):
        result = await tracker.step(
            observations=[_body_obs(detection_id=f"s{k}", embedding=_STRANGER_BODY, frame_index=k)],
            now=BASE_TIME,
            face_anchors=None,
        )
        assert all(d.identity_id != "grandma" for d in result.identity_decisions)

    open_phs = await tracker._ph_repo.list_open()
    assert open_phs, "stranger PH should exist"
    assert all(ph.current_identity_id != "grandma" for ph in open_phs)


@pytest.mark.asyncio
async def test_multiview_gallery_still_commits_matching_identity() -> None:
    """Positive control: the UNKNOWN-complement fix must not over-suppress. A PH
    whose body matches grandma's gallery prototypes (high similarity) still
    commits to grandma via the live multiview query. Without this, the guardrail
    test above would pass vacuously by never committing anyone."""
    tracker, gallery = await _make_tracker(multiview=True)
    await _seed_grandma_body(gallery)

    committed = False
    for k in range(6):
        result = await tracker.step(
            observations=[_body_obs(detection_id=f"d{k}", embedding=_GRANDMA_BODY, frame_index=k)],
            now=BASE_TIME,
            face_anchors=None,
        )
        if any(d.identity_id == "grandma" for d in result.identity_decisions):
            committed = True

    assert committed, "matching body should commit grandma via multiview gallery"
    open_phs = await tracker._ph_repo.list_open()
    assert any(ph.current_identity_id == "grandma" for ph in open_phs)


@pytest.mark.asyncio
async def test_multiview_back_view_retrieval_commits_via_max_over_views() -> None:
    """End-to-end front->back retrieval. Grandma's gallery holds
    a BACK prototype distinct from her front. A back-facing PH (no face, body
    matches the BACK entry) commits grandma via max-over-views, where a single
    front-mean query would miss. Proves turn-around re-ID through the tracker."""
    grandma_front = tuple(1.0 if i < 384 else 0.0 for i in range(768))
    grandma_back = tuple(0.0 if i < 384 else 1.0 for i in range(768))

    tracker, gallery = await _make_tracker(multiview=True)
    # Seed both a FRONT and a (distinct) BACK prototype for grandma.
    # state="operator_verified" so search_similar's verified-only default
    # (M03) can see them.
    await gallery.upsert_gallery_entry(
        GalleryEmbedding(
            gallery_entry_id="gf",
            identity_id="grandma",
            embedding=grandma_front,
            seen_at=BASE_TIME,
            quality=0.9,
            face_confirmed=True,
            camera_id="cam01",
            orientation=int(OrientationBin.FRONT),
            state="operator_verified",
        )
    )
    await gallery.upsert_gallery_entry(
        GalleryEmbedding(
            gallery_entry_id="gb",
            identity_id="grandma",
            embedding=grandma_back,
            seen_at=BASE_TIME,
            quality=0.9,
            face_confirmed=True,
            camera_id="cam01",
            orientation=int(OrientationBin.BACK),
            state="operator_verified",
        )
    )

    committed = False
    for k in range(6):
        obs = _body_obs(detection_id=f"b{k}", embedding=grandma_back, frame_index=k)
        result = await tracker.step(observations=[obs], now=BASE_TIME, face_anchors=None)
        if any(d.identity_id == "grandma" for d in result.identity_decisions):
            committed = True

    assert committed, "back-facing body matching grandma's BACK prototype must commit grandma"


@pytest.mark.asyncio
async def test_unknown_orientation_does_not_seed_gallery() -> None:
    """Even a recognized face does not seed when the orientation is UNKNOWN:
    an untagged prototype is useless for the max-over-views query."""
    tracker, gallery = await _make_tracker()

    obs = _observation(
        detection_id="d1",
        recognition_state="recognized",
        confidence=0.95,
        orientation=OrientationBin.UNKNOWN,
        orientation_confidence=0.9,
    )
    await tracker.step(observations=[obs], now=BASE_TIME, face_anchors=[obs.face_anchor])

    entries = await gallery.list_gallery_entries(
        identity_id="grandma",
        active_only=False,
        # Diagnostic read of the raw seed write (_seed_multiview_gallery leaves
        # entries pending_review by default -- F4/M04), not a resolver vote.
        states=None,
    )
    assert entries == []

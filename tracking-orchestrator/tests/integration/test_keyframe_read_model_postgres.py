"""Postgres parity for the M07 keyframe read model batch reads.

Seeds equivalent data into Postgres (via SQL) and the in-memory repositories
(via domain objects), then runs the same ``KeyframeReadModelService`` over both
bundles and asserts the composed physical-frame cards match. This exercises all
five new batch reads (`list_for_read_model`, `get_bbox_annotations_for_keyframes`,
`decisions_for_phs`, `live_ranges_for_phs`, `phs_with_pending_reid`) plus the
shared composer through one comparison.

Marked @pytest.mark.integration; CI selects this marker against a testcontainer.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from app.domain import (
    BboxAnnotation,
    GalleryEmbedding,
    IdentityProvenanceDecision,
    IdentityRevisionRange,
    TaggedKeyframe,
)
from app.services.keyframe_read_model import (
    KeyframeReadModelService,
    KeyframeReadRepositoryBundle,
)
from app.storage.base import (
    InMemoryBboxAnnotationRepository,
    InMemoryIdentityDecisionRepository,
    InMemoryKeyframeRepository,
)
from app.storage.corrections import InMemoryIdentityCorrectionRepository
from app.storage.gallery import InMemoryGalleryRepository
from app.storage.postgres.bbox_annotations import PostgresBboxAnnotationRepository
from app.storage.postgres.correction_repo import PostgresIdentityCorrectionRepository
from app.storage.postgres.gallery_repo import PostgresGalleryRepository
from app.storage.postgres.identity_decision_repo import (
    PostgresIdentityDecisionRepository,
)
from app.storage.postgres.keyframe_repo import PostgresKeyframeRepository

pytestmark = pytest.mark.integration

_T0 = datetime(2026, 6, 20, 12, 0, 0, tzinfo=UTC)
_CAM = "cam-a"
_KEY = "frames/cam-a/0001-0.jpg"


async def _seed_ph(conn: Any, ph_id: str) -> None:
    await conn.execute(
        """
        INSERT INTO continuous_tracking.person_hypotheses
            (ph_id, born_at, last_seen_at, last_seen_camera, observation_count,
             state_mean, state_cov, active_cameras, mean_quality, metadata)
        VALUES ($1::uuid, $2, $2, $3, 5, $4, $5, $6, 0.9, '{}'::jsonb)
        """,
        ph_id,
        _T0,
        _CAM,
        [1.0, 2.0, 0.0, 0.0],
        [0.1] * 16,
        [_CAM],
    )


@pytest.mark.asyncio
async def test_postgres_matches_inmemory_composition(db_pool: Any) -> None:
    ph_alpha, ph_beta = str(uuid.uuid4()), str(uuid.uuid4())
    kf_a, kf_b = str(uuid.uuid4()), str(uuid.uuid4())
    dec_alpha = str(uuid.uuid4())
    range_id, rev_id = str(uuid.uuid4()), str(uuid.uuid4())

    # -- Postgres seed --
    async with db_pool.acquire() as conn:
        await _seed_ph(conn, ph_alpha)
        await _seed_ph(conn, ph_beta)
        # Two triggers, one physical frame.
        for kf_id, ph, reason in (
            (kf_a, ph_alpha, "identity_changed"),
            (kf_b, ph_beta, "periodic"),
        ):
            await conn.execute(
                """
                INSERT INTO continuous_tracking.tagged_keyframes
                    (id, ph_id, camera_id, minio_key, captured_at, annotations,
                     tag_reason, expires_at)
                VALUES ($1::uuid, $2::uuid, $3, $4, $5, '{}'::jsonb, $6, $7)
                """,
                kf_id,
                ph,
                _CAM,
                _KEY,
                _T0,
                reason,
                _T0 + timedelta(days=1),
            )
            for ph_box, x in ((ph_alpha, 10), (ph_beta, 150)):
                await conn.execute(
                    """
                    INSERT INTO continuous_tracking.keyframe_bbox_annotations
                        (keyframe_id, ph_id, camera_id, x1, y1, x2, y2,
                         detection_confidence, frame_width, frame_height, identity_id)
                    VALUES ($1, $2::uuid, $3, $4, 10, $5, 210, 0.9, 1920, 1080, $6)
                    """,
                    kf_id,
                    ph_box,
                    _CAM,
                    float(x),
                    float(x + 100),
                    "amma" if ph_box == ph_alpha else "grandma",
                )
        await conn.execute(
            """
            INSERT INTO continuous_tracking.identity_decisions
                (decision_id, ph_id, captured_at, authority, decision_source,
                 inferred_identity_id, effective_identity_id, top_probability,
                 resolver_version, diagnostics)
            VALUES ($1::uuid, $2::uuid, $3, 'arcface_authority', 'face',
                    'amma', 'amma', 0.8, 'r1', '{}'::jsonb)
            """,
            dec_alpha,
            ph_alpha,
            _T0,
        )
        await conn.execute(
            """
            INSERT INTO continuous_tracking.identity_revision_ranges
                (range_id, revision_id, ph_id, effective_identity_id, authority,
                 range_start, range_end, created_at)
            VALUES ($1::uuid, $2::uuid, $3::uuid, 'grandma', 'operator', $4, $5, $6)
            """,
            range_id,
            rev_id,
            ph_alpha,
            _T0 - timedelta(minutes=1),
            _T0 + timedelta(minutes=1),
            _T0,
        )
        await conn.execute(
            """
            INSERT INTO continuous_tracking.reid_gallery (id, ph_id, state)
            VALUES ($1::uuid, $2::uuid, 'pending_review')
            """,
            str(uuid.uuid4()),
            ph_beta,
        )

    pg_bundle = KeyframeReadRepositoryBundle(
        keyframe_repo=PostgresKeyframeRepository(db_pool),
        bbox_repo=PostgresBboxAnnotationRepository(db_pool),
        decision_repo=PostgresIdentityDecisionRepository(db_pool),
        correction_repo=PostgresIdentityCorrectionRepository(db_pool),
        gallery_repo=PostgresGalleryRepository(db_pool),
    )
    pg_card = (await KeyframeReadModelService(pg_bundle).list_physical_frames()).frames[0]

    # -- In-memory seed with equivalent domain objects --
    km, bm = InMemoryKeyframeRepository(), InMemoryBboxAnnotationRepository()
    dm = InMemoryIdentityDecisionRepository()
    cm = InMemoryIdentityCorrectionRepository()
    gm = InMemoryGalleryRepository()
    for kf_id, ph, reason in (
        (kf_a, ph_alpha, "identity_changed"),
        (kf_b, ph_beta, "periodic"),
    ):
        await km.save_keyframe(
            TaggedKeyframe(
                keyframe_id=kf_id,
                ph_id=ph,
                camera_id=_CAM,
                minio_key=_KEY,
                captured_at=_T0,
                annotations={},
                tag_reason=reason,  # type: ignore[arg-type]
                expires_at=_T0 + timedelta(days=1),
            )
        )
        await bm.save_bbox_annotations(
            [
                BboxAnnotation(
                    keyframe_id=kf_id,
                    ph_id=ph_box,
                    camera_id=_CAM,
                    x1=float(x),
                    y1=10,
                    x2=float(x + 100),
                    y2=210,
                    detection_confidence=0.9,
                    frame_width=1920,
                    frame_height=1080,
                    identity_id="amma" if ph_box == ph_alpha else "grandma",
                )
                for ph_box, x in ((ph_alpha, 10), (ph_beta, 150))
            ]
        )
    await dm.save(
        IdentityProvenanceDecision(
            decision_id=dec_alpha,
            ph_id=ph_alpha,
            captured_at=_T0,
            authority="arcface_authority",
            decision_source="face",
            diagnostics={},
            inferred_identity_id="amma",
            effective_identity_id="amma",
            top_probability=0.8,
        )
    )
    await cm.save_range(
        IdentityRevisionRange(
            range_id=range_id,
            revision_id=rev_id,
            ph_id=ph_alpha,
            authority="operator",
            range_start=_T0 - timedelta(minutes=1),
            range_end=_T0 + timedelta(minutes=1),
            effective_identity_id="grandma",
        )
    )
    await gm.upsert_gallery_entry(
        GalleryEmbedding(
            gallery_entry_id=str(uuid.uuid4()),
            identity_id="grandma",
            embedding=[0.1, 0.2],
            seen_at=_T0,
            origin_tracklet_id=ph_beta,
            state="pending_review",
        )
    )
    mem_bundle = KeyframeReadRepositoryBundle(
        keyframe_repo=km,
        bbox_repo=bm,
        decision_repo=dm,
        correction_repo=cm,
        gallery_repo=gm,
    )
    mem_card = (await KeyframeReadModelService(mem_bundle).list_physical_frames()).frames[0]

    # Parity on the load-bearing per-bbox provenance and counts.
    def _norm(card: Any) -> str:
        return json.dumps(
            {
                "triggers": sorted(card.trigger_reasons),
                "unknown": card.unknown_count,
                "conflict": card.conflict_count,
                "pending": card.pending_review_count,
                "bboxes": sorted(
                    [
                        {
                            "ph_id": b.ph_id,
                            "inferred": b.inferred_identity_id,
                            "effective": b.effective_identity_id,
                            "authority": b.authority,
                            "source": b.decision_source,
                            "pending": b.pending_review,
                        }
                        for b in card.bboxes
                    ],
                    key=lambda d: d["ph_id"],
                ),
            },
            sort_keys=True,
        )

    assert _norm(pg_card) == _norm(mem_card)
    # Operator range overrides inference; ph-beta flagged pending.
    alpha_box = next(b for b in pg_card.bboxes if b.ph_id == ph_alpha)
    assert alpha_box.inferred_identity_id == "amma"
    assert alpha_box.effective_identity_id == "grandma"
    assert pg_card.pending_review_count == 1

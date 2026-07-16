"""M07 bounded authority vocabulary unit tests (finding F9).

Covers every producer path in ``IdentityResolver._commit`` and the duplicate-active
guard's ``_block_decision``: each must emit a member of ``IdentityAuthority``, never
an identity id. Before this milestone, the ArcFace-authority path set ``authority``
to the matched identity id and the normal Bayesian path left it ``""``.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from app.domain import FaceAnchor, GlobalTrack, Identity, IdentityDecision, PosteriorDist
from app.storage.gallery import InMemoryGalleryRepository
from app.tracking.identity.types import IdentityAuthority
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig

_KNOWN_AUTHORITIES = frozenset(a.value for a in IdentityAuthority)
_NOW = datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC)


def _make_identity(identity_id: str) -> Identity:
    return Identity(identity_id=identity_id, display_name=identity_id, enrolled_at=_NOW)


def _make_resolver(
    *,
    identities: list[Identity] | None = None,
    config: ResolverConfig | None = None,
    gallery_repo: InMemoryGalleryRepository | None = None,
) -> IdentityResolver:
    return IdentityResolver(
        gallery_repo=gallery_repo or InMemoryGalleryRepository(),
        identities=identities or [],
        config=config or ResolverConfig(),
    )


def _face_anchor(
    person_id: str,
    *,
    confidence: float = 0.92,
    calibrated_confidence: float | None = None,
    tracklet_id: str = "t1",
) -> FaceAnchor:
    return FaceAnchor(
        person_id=person_id,
        confidence=confidence,
        quality=0.9,
        tracklet_id=tracklet_id,
        calibrated_confidence=calibrated_confidence,
    )


class _FakePH:
    """Minimal IdentityResolvableEntity for direct ``_commit`` calls."""

    def __init__(
        self,
        entity_id: str = "ph-1",
        current_identity_id: str | None = None,
        current_identity_committed_at: datetime | None = None,
        last_independent_identity_evidence_at: datetime | None = None,
    ) -> None:
        self.entity_id = entity_id
        self.current_identity_id = current_identity_id
        self.current_identity_committed_at = current_identity_committed_at
        self.last_independent_identity_evidence_at = last_independent_identity_evidence_at
        self.last_seen_at = _NOW
        self.started_at = _NOW

    @property
    def observation_ids(self) -> list[str]:
        return ["obs-1"]

    @property
    def camera_ids(self) -> list[str]:
        return ["cam-1"]

    @property
    def view_prototypes(self) -> tuple:
        return ()


# ---------------------------------------------------------------------------
# 1. Each producer path emits the mapped vocabulary value
# ---------------------------------------------------------------------------


class TestProducerPathsEmitVocabulary:
    def test_arcface_authority_commit_emits_direct_face(self) -> None:
        resolver = _make_resolver()
        entity = _FakePH()
        decision = resolver._commit(
            entity,
            PosteriorDist({"alice": 0.9, "UNKNOWN": 0.1}),
            PosteriorDist({"alice": 0.9}),
            PosteriorDist({}),
            _NOW,
            arcface_authority="alice",
        )
        assert decision.identity_id == "alice"
        assert decision.authority == "direct_face"

    def test_arcface_authority_conflict_emits_none(self) -> None:
        resolver = _make_resolver()
        entity = _FakePH()
        decision = resolver._commit(
            entity,
            PosteriorDist({"alice": 0.5, "bob": 0.5}),
            PosteriorDist({"alice": 0.5, "bob": 0.5}),
            PosteriorDist({}),
            _NOW,
            arcface_authority="CONFLICT",
        )
        assert decision.identity_id is None
        assert decision.authority == "none"

    def test_posterior_commit_face_led_emits_posterior(self) -> None:
        resolver = _make_resolver()
        entity = _FakePH()
        decision = resolver._commit(
            entity,
            PosteriorDist({"alice": 0.90, "UNKNOWN": 0.10}),
            PosteriorDist({"alice": 0.90}),
            PosteriorDist({}),
            _NOW,
            evidence_identity_ids=frozenset({"alice"}),
        )
        assert decision.identity_id == "alice"
        assert decision.decision_source == "face"
        assert decision.authority == "posterior"

    def test_posterior_commit_reid_led_emits_posterior(self) -> None:
        resolver = _make_resolver()
        entity = _FakePH()
        decision = resolver._commit(
            entity,
            PosteriorDist({"alice": 0.90, "UNKNOWN": 0.10}),
            PosteriorDist({}),
            PosteriorDist({"alice": 0.90}),
            _NOW,
            evidence_identity_ids=frozenset({"alice"}),
        )
        assert decision.identity_id == "alice"
        assert decision.decision_source == "reid"
        assert decision.authority == "posterior"

    def test_maintenance_window_hold_emits_temporal_prior(self) -> None:
        resolver = _make_resolver(config=ResolverConfig(prior_maintenance_max_age_s=30.0))
        entity = _FakePH(
            current_identity_id="alice",
            current_identity_committed_at=_NOW - timedelta(seconds=5),
            last_independent_identity_evidence_at=_NOW - timedelta(seconds=5),
        )
        # No evidence this frame -- posterior still favors the held identity
        # (mirrors the resolver's own prior contribution), but evidence_identity_ids
        # is empty so has_evidence is False and the hold is prior-only.
        decision = resolver._commit(
            entity,
            PosteriorDist({"alice": 0.6, "UNKNOWN": 0.4}),
            PosteriorDist({}),
            PosteriorDist({}),
            _NOW,
            evidence_identity_ids=frozenset(),
        )
        assert decision.identity_id == "alice"
        assert decision.evidence_backed is False
        assert decision.authority == "temporal_prior"

    def test_decay_to_unknown_emits_none(self) -> None:
        """Window expired, no evidence, previous identity held: demoted to Unknown."""
        resolver = _make_resolver(config=ResolverConfig(prior_maintenance_max_age_s=30.0))
        entity = _FakePH(
            current_identity_id="alice",
            current_identity_committed_at=_NOW - timedelta(seconds=90),
            last_independent_identity_evidence_at=_NOW - timedelta(seconds=90),
        )
        decision = resolver._commit(
            entity,
            PosteriorDist({"UNKNOWN": 1.0}),
            PosteriorDist({}),
            PosteriorDist({}),
            _NOW,
            evidence_identity_ids=frozenset(),
        )
        assert decision.identity_id is None
        assert decision.authority == "none"

    @pytest.mark.asyncio
    async def test_duplicate_active_block_emits_none(self) -> None:
        """A weaker contender demoted by the duplicate-active guard carries authority=none."""
        identities = [_make_identity("alice")]
        gallery = InMemoryGalleryRepository()
        await gallery.upsert_identity(identities[0])

        resolver = _make_resolver(
            identities=identities,
            gallery_repo=gallery,
            config=ResolverConfig(
                enable_duplicate_active_identity_guard=True,
                face_commit_min_confidence=0.60,
            ),
        )

        gt_a = GlobalTrack(
            global_track_id="gt-a",
            camera_ids=["cam_a"],
            tracklet_ids=["t1"],
            started_at=_NOW,
            last_seen_at=_NOW,
            current_identity_id=None,
            state="active",
        )
        gt_b = GlobalTrack(
            global_track_id="gt-b",
            camera_ids=["cam_b"],
            tracklet_ids=["t2"],
            started_at=_NOW,
            last_seen_at=_NOW,
            current_identity_id=None,
            state="active",
        )

        outcome = await resolver.resolve(
            hypotheses=[gt_a, gt_b],
            new_face_anchors=[_face_anchor("alice", confidence=0.95, tracklet_id="t1")],
            captured_at=_NOW,
        )

        by_id = {d.ph_id: d for d in outcome.decisions}
        assert by_id["gt-a"].identity_id == "alice"
        assert by_id["gt-b"].identity_id is None
        assert by_id["gt-b"].authority == "none"


# ---------------------------------------------------------------------------
# 2. Property: no decision ever carries an identity id as its authority
# ---------------------------------------------------------------------------


class TestAuthorityVocabularyProperty:
    @pytest.mark.asyncio
    async def test_randomized_fixture_set_never_leaks_identity_id_as_authority(self) -> None:
        rng = random.Random(20260715)
        identities = [_make_identity(f"person-{i}") for i in range(4)]
        known_ids = [ident.identity_id for ident in identities]
        gallery = InMemoryGalleryRepository()
        for ident in identities:
            await gallery.upsert_identity(ident)

        resolver = _make_resolver(identities=identities, gallery_repo=gallery)

        seen_authorities: set[str] = set()
        for trial in range(60):
            person = rng.choice(known_ids)
            prev = rng.choice([None, *known_ids])
            evidence_age_s = rng.choice([None, 5, 20, 40, 90])
            evidence_at = (
                _NOW - timedelta(seconds=evidence_age_s) if evidence_age_s is not None else None
            )
            gt = GlobalTrack(
                global_track_id=f"gt-{trial}",
                camera_ids=["cam_a"],
                tracklet_ids=[f"t-{trial}"],
                started_at=_NOW,
                last_seen_at=_NOW,
                current_identity_id=prev,
                current_identity_committed_at=evidence_at,
                last_independent_identity_evidence_at=evidence_at,
                state="active",
            )
            anchors: list[FaceAnchor] = []
            if rng.random() < 0.7:
                anchors.append(
                    _face_anchor(
                        person,
                        confidence=rng.uniform(0.5, 0.99),
                        tracklet_id=f"t-{trial}",
                        calibrated_confidence=rng.choice([None, 0.6, 0.85]),
                    )
                )
            if rng.random() < 0.1:
                other = rng.choice([i for i in known_ids if i != person])
                anchors.append(
                    _face_anchor(
                        other,
                        confidence=0.9,
                        tracklet_id=f"t-{trial}",
                        calibrated_confidence=0.9,
                    )
                )

            outcome = await resolver.resolve(
                hypotheses=[gt], new_face_anchors=anchors, captured_at=_NOW
            )
            for decision in outcome.decisions:
                assert isinstance(decision, IdentityDecision)
                seen_authorities.add(decision.authority)
                assert decision.authority in _KNOWN_AUTHORITIES
                assert decision.authority not in known_ids

        # Sanity: the randomized run exercised more than one rung of the ladder.
        assert len(seen_authorities) >= 2

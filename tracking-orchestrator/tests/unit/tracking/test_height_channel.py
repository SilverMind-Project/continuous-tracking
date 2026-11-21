"""Unit tests for HeightChannel likelihood math and IdentityResolver height_mode."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from app.domain import (
    GalleryEmbedding,
    Identity,
    IdentityHeightProfile,
    OrientationBin,
    PersonHypothesis,
    ViewPrototype,
)
from app.observability import metrics as _metrics
from app.storage.base import InMemoryGalleryRepository
from app.storage.height_profiles import InMemoryIdentityHeightProfileRepository
from app.tracking.identity.channels import HeightChannel
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig


def _profile(identity_id: str, height_m: float, sigma_m: float = 0.03) -> IdentityHeightProfile:
    return IdentityHeightProfile(
        identity_id=identity_id,
        height_m=height_m,
        height_sigma_m=sigma_m,
        source="learned",
        samples=250,
        cameras=2,
        updated_at=datetime(2026, 11, 21, 12, 0, 0, tzinfo=UTC),
    )


def _gt(gt_id: str = "gt-1", embedding: list[float] | None = None) -> PersonHypothesis:
    now = datetime(2026, 11, 21, 12, 0, 0, tzinfo=UTC)
    emb = embedding or [1.0, 0.0, 0.0, 0.0]
    return PersonHypothesis(
        ph_id=gt_id,
        state_mean=(1.0, 1.0, 0.0, 0.0),
        state_cov=(
            0.2,
            0.0,
            0.0,
            0.0,
            0.0,
            0.2,
            0.0,
            0.0,
            0.0,
            0.0,
            0.5,
            0.0,
            0.0,
            0.0,
            0.0,
            0.5,
        ),
        born_at=now,
        last_seen_at=now,
        last_seen_camera="cam-a",
        observation_count=5,
        gallery_mean=emb,
        active_cameras=frozenset({"cam-a"}),
        view_prototypes=(
            ViewPrototype(
                orientation=OrientationBin.FRONT,
                embedding=tuple(emb),
                count=5,
            ),
        ),
    )


def test_height_channel_exact_likelihood_ratio_and_entity_id_lookup() -> None:
    """1.52 m vs 1.80 m residents with 1.55 m PH gives exact Gaussian ratio keyed by identity_id."""
    alice_p = _profile("id-alice", 1.52, 0.03)
    bob_p = _profile("id-bob", 1.80, 0.03)
    profiles = {"id-alice": alice_p, "id-bob": bob_p}
    ch = HeightChannel(
        height_profiles_getter=lambda: profiles,
        height_weight_multiplier=0.15,
    )

    ev = ch.evaluate_heights(
        _gt("gt-1"),
        ph_heights_m={"gt-1": 1.55},
        ph_height_sigmas_m={"gt-1": 0.04},
    )
    assert bool(ev.distribution.distribution)
    assert ev.evidence_at is None
    assert ev.support_multiplier == pytest.approx(1.0)

    sigma_comb = math.hypot(0.04, 0.03)  # 0.05 m
    z_alice = (1.55 - 1.52) / sigma_comb  # 0.6
    z_bob = (1.55 - 1.80) / sigma_comb  # -5.0
    raw_alice = max(0.01, math.exp(-0.5 * 0.15 * z_alice * z_alice))
    raw_bob = max(0.01, math.exp(-0.5 * 0.15 * z_bob * z_bob))
    expected_ratio = raw_alice / raw_bob

    actual_ratio = ev.distribution.distribution["id-alice"] / ev.distribution.distribution["id-bob"]
    assert actual_ratio == pytest.approx(expected_ratio, rel=1e-6)


def test_height_channel_missing_profile_is_neutral_and_mismatch_below_unknown() -> None:
    # No profiles -> channel emits empty distribution (completely neutral).
    empty_ch = HeightChannel(
        height_profiles_getter=lambda: {},
        height_weight_multiplier=0.15,
    )
    empty_ev = empty_ch.evaluate_heights(
        _gt("gt-1"),
        ph_heights_m={"gt-1": 1.55},
        ph_height_sigmas_m={"gt-1": 0.04},
    )
    assert not empty_ev.distribution.distribution

    # Single resident (1.80 m) severely mismatched against 1.52 m PH must NOT beat UNKNOWN.
    bob_profiles = {"bob": _profile("bob", 1.80, 0.03)}
    mismatch_ch = HeightChannel(
        height_profiles_getter=lambda: bob_profiles,
        height_weight_multiplier=0.15,
    )
    ev = mismatch_ch.evaluate_heights(
        _gt("gt-1"),
        ph_heights_m={"gt-1": 1.52},
        ph_height_sigmas_m={"gt-1": 0.04},
    )
    assert bool(ev.distribution.distribution)
    assert ev.distribution.distribution["bob"] < ev.distribution.distribution["UNKNOWN"]


@pytest.mark.asyncio
async def test_resolver_height_mode_off_shadow_and_enabled() -> None:
    """Identical ReID tie stays unresolved in off/shadow, counts shadow, resolves in enabled."""
    now = datetime(2026, 11, 21, 12, 0, 0, tzinfo=UTC)
    gallery = InMemoryGalleryRepository()
    identities = [
        Identity(identity_id="alice", display_name="Alice", enrolled_at=now),
        Identity(identity_id="bob", display_name="Bob", enrolled_at=now),
    ]
    for ident in identities:
        await gallery.upsert_identity(ident)

    # Both Alice and Bob wear identical clothes -> identical ReID embedding.
    shared_emb = [1.0, 0.0, 0.0, 0.0]
    await gallery.upsert_gallery_entry(
        GalleryEmbedding(
            gallery_entry_id="e-alice",
            identity_id="alice",
            embedding=shared_emb,
            quality=0.95,
            seen_at=now,
            face_confirmed=True,
            orientation=OrientationBin.FRONT,
            state="operator_verified",
        )
    )
    await gallery.upsert_gallery_entry(
        GalleryEmbedding(
            gallery_entry_id="e-bob",
            identity_id="bob",
            embedding=shared_emb,
            quality=0.95,
            seen_at=now,
            face_confirmed=True,
            orientation=OrientationBin.FRONT,
            state="operator_verified",
        )
    )

    profiles = InMemoryIdentityHeightProfileRepository(
        [
            _profile("alice", 1.52, 0.03),
            _profile("bob", 1.80, 0.03),
        ]
    )
    gt = _gt(embedding=shared_emb)

    # 1. mode="off" -> tie stays UNKNOWN
    res_off = IdentityResolver(
        gallery_repo=gallery,
        identities=identities,
        config=ResolverConfig(
            height_mode="off",
            enable_sticky_maintenance=False,
            enable_multiview_gallery=True,
        ),
        height_profile_repo=profiles,
    )
    out_off = await res_off.resolve(
        hypotheses=[gt],
        new_face_anchors=[],
        captured_at=now,
        ph_heights={"gt-1": 1.53},
        ph_height_sigmas={"gt-1": 0.03},
    )
    assert out_off.decisions[0].identity_id is None

    # 2. mode="shadow" -> live decision still UNKNOWN, shadow mismatch metric increments
    shadow_counter = _metrics.metrics.identity_shadow_mismatch_total.labels(feature="height")
    shadow_before = shadow_counter._value.get()
    res_shadow = IdentityResolver(
        gallery_repo=gallery,
        identities=identities,
        config=ResolverConfig(
            height_mode="shadow",
            enable_sticky_maintenance=False,
            enable_multiview_gallery=True,
        ),
        height_profile_repo=profiles,
    )
    out_shadow = await res_shadow.resolve(
        hypotheses=[gt],
        new_face_anchors=[],
        captured_at=now,
        ph_heights={"gt-1": 1.53},
        ph_height_sigmas={"gt-1": 0.03},
    )
    assert out_shadow.decisions[0].identity_id == out_off.decisions[0].identity_id
    assert out_shadow.decisions[0].identity_id is None
    assert shadow_counter._value.get() == shadow_before + 1.0

    # 3. mode="enabled" -> height breaks the tie cleanly to Alice, and increments height metric
    height_frames_before = _metrics.metrics.height_evidence_frames_total._value.get()
    res_on = IdentityResolver(
        gallery_repo=gallery,
        identities=identities,
        config=ResolverConfig(
            height_mode="enabled",
            enable_sticky_maintenance=False,
            enable_multiview_gallery=True,
        ),
        height_profile_repo=profiles,
    )
    out_on = await res_on.resolve(
        hypotheses=[gt],
        new_face_anchors=[],
        captured_at=now,
        ph_heights={"gt-1": 1.53},
        ph_height_sigmas={"gt-1": 0.03},
    )
    assert out_on.decisions[0].identity_id == "alice"
    assert out_on.decisions[0].posterior.distribution["alice"] > 0.6
    assert _metrics.metrics.height_evidence_frames_total._value.get() == height_frames_before + 1.0

"""M12 metrics: bounded label cardinality and emission wiring.

The identity-integrity program forbids unbounded labels: PH, decision, and
identity IDs belong in structlog, never in a Prometheus label. This test asserts
the new M12 series exist, carry only finite/ID-free labels, and that the page-now
rejected-vector invariant counter is actually wired.
"""

from __future__ import annotations

from datetime import UTC, datetime

from prometheus_client import CollectorRegistry

from app.domain import GalleryEmbedding
from app.observability.metrics import build_metrics
from app.observability.metrics import metrics as global_metrics
from app.storage.gallery import InMemoryGalleryRepository
from app.tracking.identity_resolver import IdentityResolver

# Substrings that would indicate an unbounded ID leaked into a label name.
_FORBIDDEN_LABEL_TOKENS = (
    "ph_id",
    "decision_id",
    "identity_id",
    "gallery_entry_id",
    "tracklet_id",
    "revision_id",
    "correction_id",
)

_M12_METRICS = (
    "cts_identity_duplicate_active_blocks_total",
    "cts_reid_rejected_vector_vote_attempts_total",
    "cts_identity_prior_only_updates_total",
)


def test_m12_metrics_exist_with_no_labels() -> None:
    registry = CollectorRegistry()
    build_metrics(registry=registry)
    families = {f.name: f for f in registry.collect()}
    # prometheus_client strips the _total suffix from the family name.
    for full_name in _M12_METRICS:
        family_name = full_name.removesuffix("_total")
        assert family_name in families, f"missing metric family {family_name}"
        for sample in families[family_name].samples:
            assert sample.labels == {}, f"{full_name} must have no labels, got {sample.labels}"


def test_no_metric_uses_an_id_label() -> None:
    registry = CollectorRegistry()
    build_metrics(registry=registry)
    offenders: list[tuple[str, str]] = []
    for family in registry.collect():
        for sample in family.samples:
            for label_name in sample.labels:
                if any(tok in label_name for tok in _FORBIDDEN_LABEL_TOKENS):
                    offenders.append((family.name, label_name))
    assert not offenders, f"ID-like metric labels found: {offenders}"


async def test_rejected_vector_vote_attempt_is_counted() -> None:
    """A non-operator_verified gallery vector reaching the vote must increment."""
    resolver = IdentityResolver(gallery_repo=InMemoryGalleryRepository())
    counter = global_metrics.reid_rejected_vector_vote_attempts_total
    before = counter._value.get()

    pending_entry = GalleryEmbedding(
        gallery_entry_id="g-pending",
        identity_id="amma",
        embedding=[1.0] * 8,
        seen_at=datetime.now(UTC),
        state="pending_review",  # not operator_verified
    )
    verified_entry = GalleryEmbedding(
        gallery_entry_id="g-verified",
        identity_id="amma",
        embedding=[1.0] * 8,
        seen_at=datetime.now(UTC),
        state="operator_verified",
    )
    resolver._score_gallery_hits([(pending_entry, 0.9), (verified_entry, 0.9)])

    # Exactly one attempt (the pending vector); the verified one is legitimate.
    assert counter._value.get() == before + 1

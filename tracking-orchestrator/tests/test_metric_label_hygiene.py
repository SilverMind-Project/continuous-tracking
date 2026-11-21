"""Guard that every registered CTS Prometheus metric uses only bounded allowlisted labels.

Identity ids, person ids, member ids, tag ids, PH ids, names, and arbitrary free text must never
appear as Prometheus labels: unbounded cardinality breaks scraping and leaks household identities
into metrics.
"""

from __future__ import annotations

from prometheus_client import REGISTRY

from app.observability import metrics as _metrics

# Closed allowlist of bounded label names permitted on CTS metrics. Any new label name must be a
# bounded vocabulary (declared as a closed enum or bounded infrastructure key) and added here
# deliberately.
_ALLOWED_LABEL_NAMES: frozenset[str] = frozenset(
    {
        "action",
        "actor",
        "arm",
        "band",
        "calibrated",
        "camera_id",
        "direction",
        "dropped",
        "endpoint",
        "event_type",
        "feature",
        "gate",
        "grade",
        "kept",
        "kind",
        "mode",
        "model",
        "outcome",
        "path",
        "posture",
        "reason",
        "recognition_state",
        "resource",
        "result",
        "room_id",
        "severity",
        "signal_kind",
        "source",
        "stage",
        "state",
        "stream",
        "table",
    }
)

_FORBIDDEN_LABEL_NAMES: frozenset[str] = frozenset(
    {
        "identity_id",
        "person_id",
        "member_id",
        "resident_id",
        "tag_id",
        "ph_id",
        "name",
        "display_name",
    }
)


def test_cts_metric_labels_are_bounded_allowlisted_names() -> None:
    # Touch the central metrics module so every CTS collector is registered.
    assert _metrics.metrics is not None

    seen_metrics: set[str] = set()
    violations: list[str] = []

    for collector in list(REGISTRY._collector_to_names):
        label_names = tuple(getattr(collector, "_labelnames", ()))
        for metric_family in collector.collect():
            name = metric_family.name
            if not name.startswith("cts_") or name in seen_metrics:
                continue
            seen_metrics.add(name)
            for label in label_names:
                if label in _FORBIDDEN_LABEL_NAMES:
                    violations.append(f"{name}: forbidden identity/free-text label {label!r}")
                elif label not in _ALLOWED_LABEL_NAMES:
                    violations.append(f"{name}: unapproved label {label!r} (not in allowlist)")

    # Non-vacuity canary: CTS registers well over 100 metrics.
    assert len(seen_metrics) >= 100, f"Expected >= 100 CTS metrics, found {len(seen_metrics)}"
    assert not violations, "Unbounded or unapproved Prometheus label names found:\n" + "\n".join(
        sorted(violations)
    )

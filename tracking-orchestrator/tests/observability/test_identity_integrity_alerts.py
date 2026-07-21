"""M12 alert rules: wellformedness, metric-name correctness, and firing logic.

No Prometheus rule engine runs in CI, so this exercises the four page-now rules
two ways: it checks each rule references a metric the orchestrator actually
exports (catching typos against the live registry), and it feeds a synthetic
counter increase through a minimal expression evaluator to prove the rule fires
on a breach value and stays quiet at zero.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from prometheus_client import CollectorRegistry

from app.observability.metrics import build_metrics

_ALERTS = (
    Path(__file__).resolve().parents[2]
    / "app"
    / "observability"
    / "alerts"
    / "identity_integrity_alerts.yml"
)

# increase(<metric>[<window>]) <op> <threshold>
_EXPR = re.compile(
    r"increase\(\s*(?P<metric>[a-z0-9_]+)\s*\[\s*\d+[smhd]\s*\]\s*\)\s*(?P<op>>|>=)\s*(?P<thr>\d+)"
)

# cts_correction_projection_failures_total is exported by cognitive-companion;
# every other alert metric is a CTS counter.
_CC_ONLY_METRICS = {"cts_correction_projection_failures_total"}


def _rules() -> list[dict]:
    doc = yaml.safe_load(_ALERTS.read_text())
    return [rule for group in doc["groups"] for rule in group["rules"]]


def _cts_metric_names() -> set[str]:
    registry = CollectorRegistry()
    build_metrics(registry=registry)
    names: set[str] = set()
    for family in registry.collect():
        names.add(family.name)
        names.add(f"{family.name}_total")
    return names


def _fires(expr: str, increase_value: float) -> bool:
    m = _EXPR.search(expr)
    assert m, f"unparseable alert expr: {expr}"
    threshold = float(m.group("thr"))
    return increase_value > threshold if m.group("op") == ">" else increase_value >= threshold


def test_all_four_page_now_rules_present() -> None:
    names = {rule["alert"] for rule in _rules()}
    assert names == {
        "IdentityDuplicateAuthoritativeAfterCommit",
        "RejectedVectorParticipatedInVote",
        "PriorOnlyAdvancedEvidenceTime",
        "CorrectionProjectionFailuresBeyondRetry",
    }


def test_rules_are_critical_and_page_immediately() -> None:
    for rule in _rules():
        assert rule["labels"]["severity"] == "critical"
        assert rule["for"] == "0m"
        assert rule["labels"]["domain"] == "identity_integrity"


def test_every_rule_references_a_real_metric() -> None:
    cts_names = _cts_metric_names()
    for rule in _rules():
        m = _EXPR.search(rule["expr"])
        assert m, rule["expr"]
        metric = m.group("metric")
        if metric in _CC_ONLY_METRICS:
            continue
        assert metric in cts_names, f"{rule['alert']} references unknown CTS metric {metric}"


def test_rules_fire_on_breach_and_stay_quiet_at_zero() -> None:
    for rule in _rules():
        expr = rule["expr"]
        m = _EXPR.search(expr)
        assert m
        threshold = int(m.group("thr"))
        # At/just-above threshold fires; zero does not.
        assert _fires(expr, threshold + 1) is True, rule["alert"]
        assert _fires(expr, 0) is False, rule["alert"]

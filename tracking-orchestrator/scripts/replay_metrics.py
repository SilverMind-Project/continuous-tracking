"""Tracking quality metrics for WorldTracker association parameter sweep.

Computes per-run metrics from replay results matched against ground-truth
sidecar files.  All functions are pure (no I/O, no async).

Metrics:
    identity_preservation  -- IDF1-like score adapted to PersonHypothesis lineage
    phantom_rate           -- PH-frames not matching any true person
    fragmentation          -- PHs per true person (1.0 is ideal)
    identity_contamination -- frames where committed identity mismatches truth
                              (absolute guardrail: any nonzero flags inadmissible)

Usage::

    from scripts.replay_metrics import SweepRunResult, score_run

    result = SweepRunResult(...)
    metrics = score_run(result, truth)

Truth format (loaded from *.truth.json sidecar)::

    {
      "persons": ["alice", "bob"],
      "detection_truth": {"det-id-0": "alice", "det-id-1": "bob"},
      "events": [...]
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FrameRecord:
    """Per-frame tracker output, captured during replay."""

    step: int
    det_to_ph: dict[str, str]
    ph_identities: dict[str, str | None]


@dataclass
class SweepRunResult:
    """Accumulated output of one WorldTracker replay for metric scoring."""

    fixture_name: str
    frames: list[FrameRecord] = field(default_factory=list)


@dataclass
class RunMetrics:
    """Scored quality metrics for one (fixture, param_combo) run."""

    fixture_name: str
    # IDF1-like score across all true persons [0, 1]; higher is better.
    identity_preservation: float
    # Fraction of PH-observation-frames with no true-person match [0, 1]; lower is better.
    phantom_rate: float
    # Average number of distinct PH lineages per true person; 1.0 is ideal.
    fragmentation: float
    # Frames where a PH carries the wrong committed identity; must be 0 for admission.
    identity_contamination: int
    # True-person count in the fixture (for context).
    person_count: int

    @property
    def admissible(self) -> bool:
        """A combination is inadmissible if any contamination is observed."""
        return self.identity_contamination == 0


def score_run(result: SweepRunResult, truth: dict) -> RunMetrics:
    """Score a replay run against the truth sidecar.

    Args:
        result:  Accumulated FrameRecords from the WorldTracker replay.
        truth:   Parsed truth sidecar dict (persons, detection_truth, events).

    Returns:
        RunMetrics with all four tracking quality scores.
    """
    detection_truth: dict[str, str] = truth.get("detection_truth", {})
    persons: list[str] = truth.get("persons", [])

    if not persons:
        return RunMetrics(
            fixture_name=result.fixture_name,
            identity_preservation=1.0,
            phantom_rate=0.0,
            fragmentation=1.0,
            identity_contamination=0,
            person_count=0,
        )

    # --- Build per-observation assignment across all frames ---
    # ph_obs_truth: ph_id → {person: count}
    ph_obs_truth: dict[str, dict[str, int]] = {}
    # ph_total_obs: ph_id → total observation count
    ph_total_obs: dict[str, int] = {}
    # person_total_obs: person → total frame count
    person_total_obs: dict[str, int] = {}
    # contamination: frames where PH's committed identity != true person label
    contamination_count = 0
    # phantom: PH-observation-frames where detection_id has no truth entry
    phantom_count = 0
    total_ph_obs = 0

    for frame in result.frames:
        for det_id, ph_id in frame.det_to_ph.items():
            total_ph_obs += 1
            true_person = detection_truth.get(det_id)
            if true_person is None:
                phantom_count += 1
                continue

            # Accumulate person_total_obs
            person_total_obs[true_person] = person_total_obs.get(true_person, 0) + 1

            # Accumulate ph_obs_truth
            if ph_id not in ph_obs_truth:
                ph_obs_truth[ph_id] = {}
            ph_obs_truth[ph_id][true_person] = ph_obs_truth[ph_id].get(true_person, 0) + 1
            ph_total_obs[ph_id] = ph_total_obs.get(ph_id, 0) + 1

            # Check identity contamination
            committed_id = frame.ph_identities.get(ph_id)
            if committed_id is not None and committed_id != true_person:
                contamination_count += 1

    # For persons not seen in frames (e.g. empty fixture)
    for p in persons:
        if p not in person_total_obs:
            person_total_obs[p] = 0

    # --- IDF1 computation ---
    # For each true person P, find the best-covering PH H* = argmax coverage(H, P).
    # Coverage(H, P) = count of H's observations that are truly P.
    #
    # TP(P) = max_H coverage(H, P)   (observations of P assigned to H*)
    # FN(P) = person_total_obs[P] - TP(P)  (P's frames not in H*)
    # FP(P) = ph_total_obs[H*] - TP(P)     (H*'s frames not from P)
    # IDF1(P) = 2*TP / (2*TP + FP + FN)

    idf1_scores: list[float] = []
    ph_per_person: list[int] = []

    for person in persons:
        p_total = person_total_obs.get(person, 0)
        if p_total == 0:
            idf1_scores.append(1.0)
            ph_per_person.append(0)
            continue

        # Find best PH for this person
        best_tp = 0
        best_ph_id: str | None = None
        for ph_id, truth_counts in ph_obs_truth.items():
            tp_candidate = truth_counts.get(person, 0)
            if tp_candidate > best_tp:
                best_tp = tp_candidate
                best_ph_id = ph_id

        if best_ph_id is None or best_tp == 0:
            idf1_scores.append(0.0)
            ph_per_person.append(0)
            continue

        fn = p_total - best_tp
        fp = ph_total_obs[best_ph_id] - best_tp
        denom = 2 * best_tp + fp + fn
        idf1_p = (2 * best_tp / denom) if denom > 0 else 0.0
        idf1_scores.append(idf1_p)

        # Count distinct PHs that covered this person
        distinct_phs = sum(1 for counts in ph_obs_truth.values() if counts.get(person, 0) > 0)
        ph_per_person.append(max(distinct_phs, 1))

    identity_preservation = sum(idf1_scores) / len(idf1_scores) if idf1_scores else 1.0
    fragmentation = sum(ph_per_person) / len(ph_per_person) if ph_per_person else 1.0
    phantom_rate = phantom_count / total_ph_obs if total_ph_obs > 0 else 0.0

    return RunMetrics(
        fixture_name=result.fixture_name,
        identity_preservation=identity_preservation,
        phantom_rate=phantom_rate,
        fragmentation=fragmentation,
        identity_contamination=contamination_count,
        person_count=len(persons),
    )


def aggregate_metrics(metrics_list: list[RunMetrics]) -> dict[str, float]:
    """Aggregate a list of per-fixture metrics into one summary dict.

    Contamination is summed (any nonzero = inadmissible for the combination).
    Other metrics are averaged across fixtures.
    """
    if not metrics_list:
        return {
            "identity_preservation": 1.0,
            "phantom_rate": 0.0,
            "fragmentation": 1.0,
            "identity_contamination": 0.0,
            "admissible": 1.0,
        }

    n = len(metrics_list)
    total_contamination = sum(m.identity_contamination for m in metrics_list)
    return {
        "identity_preservation": sum(m.identity_preservation for m in metrics_list) / n,
        "phantom_rate": sum(m.phantom_rate for m in metrics_list) / n,
        "fragmentation": sum(m.fragmentation for m in metrics_list) / n,
        "identity_contamination": float(total_contamination),
        "admissible": 1.0 if total_contamination == 0 else 0.0,
    }


def format_summary_table(rows: list[dict]) -> str:
    """Format a list of result dicts as a markdown table, ranked by score.

    Each row must contain the aggregated metrics plus the param combo dict.
    Admissible combinations are ranked first, then by identity_preservation desc.
    """
    if not rows:
        return "_No results._\n"

    admissible = [r for r in rows if r["admissible"] == 1.0]
    inadmissible = [r for r in rows if r["admissible"] != 1.0]

    admissible.sort(key=lambda r: -r["identity_preservation"])
    inadmissible.sort(key=lambda r: r["identity_contamination"])

    lines: list[str] = []

    def _param_str(row: dict) -> str:
        p = row.get("params", {})
        return ", ".join(f"{k}={v}" for k, v in sorted(p.items()))

    def _table(section_rows: list[dict], header: str) -> list[str]:
        out = [f"### {header}", ""]
        out.append(
            "| Rank | gate_chi2 | alpha_geo | alpha_app | obs_noise_m | grace_s"
            " | IDP | phantom | frag | contamination |"
        )
        out.append(
            "|------|-----------|-----------|-----------|-------------|---------|-----|---------|------|---------------|"
        )
        for rank, row in enumerate(section_rows, 1):
            p = row.get("params", {})
            out.append(
                f"| {rank}"
                f" | {p.get('gate_chi2', '-')}"
                f" | {p.get('alpha_geo', '-')}"
                f" | {p.get('alpha_app', '-')}"
                f" | {p.get('observation_noise_m', '-')}"
                f" | {p.get('ph_close_grace_s', '-')}"
                f" | {row['identity_preservation']:.3f}"
                f" | {row['phantom_rate']:.3f}"
                f" | {row['fragmentation']:.2f}"
                f" | {int(row['identity_contamination'])} |"
            )
        out.append("")
        return out

    if admissible:
        lines += _table(admissible, "Admissible combinations (contamination = 0)")
    if inadmissible:
        lines += _table(inadmissible, "Inadmissible combinations (contamination > 0)")

    return "\n".join(lines)

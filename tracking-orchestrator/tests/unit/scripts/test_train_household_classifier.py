"""Unit tests for scripts/train_household_classifier.py.

Uses synthetic multi-day datasets with stable per-person components plus day-specific
clothing offsets to prove that leave-one-day-out CV generalizes across days better
than same-day training, and verifies every acceptance gate (audit bar, day/sample
minimums, minority recall, open-set false-match, and fixed-seed determinism).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

# scripts/ is outside app/ (app/ is forbidden from importing scripts).
_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from train_household_classifier import (  # noqa: E402
    TrainerConfig,
    TrainingGateError,
    fit_linear_head,
    train_household_classifier,
    wilson_lower_bound,
)

from app.domain import (  # noqa: E402
    AppearanceLabelSource,
    AppearanceTrainingSample,
    OrientationBin,
)
from app.tracking.identity.household_classifier import (  # noqa: E402
    HouseholdClassifierEvaluator,
)

_PASSING_AUDIT = {"sample_size": 400, "correct_count": 400}


def _make_synthetic_dataset(
    *,
    classes: tuple[str, ...] = ("alice", "bob"),
    n_days: int = 8,
    samples_per_class_per_day: int = 80,
    negatives_per_day: int = 40,
    dim: int = 24,
    person_signal: float = 2.0,
    clothing_scale: float = 0.7,
    noise_scale: float = 0.25,
    negative_near_class0: bool = False,
    seed: int = 7,
) -> list[AppearanceTrainingSample]:
    """Generate synthetic L2-normalized body embeddings with stable person + daily clothing."""
    rng = np.random.default_rng(seed)
    base_day = date(2026, 11, 1)
    # Orthogonal person prototypes occupy the first len(classes) * 2 dims;
    # daily clothing offsets occupy the remaining dimensions so memorizing clothing hurts new days.
    person_vecs: dict[str, np.ndarray] = {}
    for idx, cls_id in enumerate(classes):
        vec = np.zeros(dim, dtype=np.float64)
        vec[idx * 2] = person_signal
        vec[idx * 2 + 1] = person_signal * 0.5
        person_vecs[cls_id] = vec

    samples: list[AppearanceTrainingSample] = []
    sample_counter = 0
    for day_offset in range(n_days):
        cur_day = base_day + timedelta(days=day_offset)
        day_start = datetime(cur_day.year, cur_day.month, cur_day.day, 12, 0, tzinfo=UTC)

        for cls_id in classes:
            clothing = np.zeros(dim, dtype=np.float64)
            clothing[len(classes) * 2 :] = rng.normal(
                0.0, clothing_scale, size=dim - len(classes) * 2
            )
            for j in range(samples_per_class_per_day):
                raw = person_vecs[cls_id] + clothing + rng.normal(0.0, noise_scale, size=dim)
                emb = raw / np.linalg.norm(raw)
                ts = day_start + timedelta(seconds=j * 15)
                sample_counter += 1
                samples.append(
                    AppearanceTrainingSample(
                        sample_id=f"s-{sample_counter:06d}",
                        observed_at=ts,
                        day=cur_day,
                        identity_id=cls_id,
                        label_source=AppearanceLabelSource.WEARABLE_ATTESTED,
                        camera_id="cam-living",
                        orientation=OrientationBin.FRONT,
                        quality=0.8,
                        feature_version="reid-solider",
                        embedding=tuple(float(v) for v in emb),
                    )
                )

        # Open-set negatives: either orthogonal visitor directions or adversarial near class 0.
        for j in range(negatives_per_day):
            if negative_near_class0:
                raw = person_vecs[classes[0]] + rng.normal(0.0, 0.15, size=dim)
            else:
                raw = np.zeros(dim, dtype=np.float64)
                raw[len(classes) * 2 :] = rng.normal(0.0, 1.5, size=dim - len(classes) * 2)
            emb = raw / np.linalg.norm(raw)
            ts = day_start + timedelta(seconds=j * 20)
            sample_counter += 1
            samples.append(
                AppearanceTrainingSample(
                    sample_id=f"neg-{sample_counter:06d}",
                    observed_at=ts,
                    day=cur_day,
                    identity_id=None,
                    label_source=AppearanceLabelSource.NEGATIVE,
                    camera_id="cam-entry",
                    orientation=OrientationBin.FRONT,
                    quality=0.8,
                    feature_version="reid-solider",
                    embedding=tuple(float(v) for v in emb),
                )
            )
    return samples


def test_lodo_cv_generalizes_to_new_day_better_than_same_day_model() -> None:
    """With strong day-specific clothing shortcuts, LODO-CV beats a same-day-trained model."""
    rng = np.random.default_rng(123)
    dim = 24
    person_a = np.zeros(dim)
    person_a[0] = 1.6
    person_a[1] = 0.8
    person_b = np.zeros(dim)
    person_b[2] = 1.6
    person_b[3] = 0.8

    # Day 1 has strong orthogonal clothing cues (~2x person norm) that swap on the unseen test day.
    cloth_a_d1 = np.zeros(dim)
    cloth_a_d1[4:14] = 1.15
    cloth_b_d1 = np.zeros(dim)
    cloth_b_d1[14:] = 1.15

    # Build 10 days (7 pre-eval + 3 eval) where daily clothing rotates independently across days,
    # plus a separate unseen test day where alice and bob swap their day-1 clothing.
    samples: list[AppearanceTrainingSample] = []
    counter = 0
    for d_idx in range(10):
        cur_day = date(2026, 11, 1) + timedelta(days=d_idx)
        day_start = datetime(cur_day.year, cur_day.month, cur_day.day, 10, 0, tzinfo=UTC)
        if d_idx == 0:
            c_a, c_b = cloth_a_d1, cloth_b_d1
        else:
            c_a = np.zeros(dim)
            c_a[4:] = rng.normal(0.0, 0.55, size=dim - 4)
            c_b = np.zeros(dim)
            c_b[4:] = rng.normal(0.0, 0.55, size=dim - 4)

        for cls_id, p_vec, c_vec in (("alice", person_a, c_a), ("bob", person_b, c_b)):
            for j in range(60):
                raw = p_vec + c_vec + rng.normal(0.0, 0.25, size=dim)
                emb = raw / np.linalg.norm(raw)
                counter += 1
                samples.append(
                    AppearanceTrainingSample(
                        sample_id=f"d-{counter:06d}",
                        observed_at=day_start + timedelta(seconds=j * 15),
                        day=cur_day,
                        identity_id=cls_id,
                        label_source=AppearanceLabelSource.WEARABLE_ATTESTED,
                        camera_id="cam-1",
                        orientation=OrientationBin.FRONT,
                        quality=0.8,
                        feature_version="reid-solider",
                        embedding=tuple(float(v) for v in emb),
                    )
                )
        for j in range(30):
            raw = np.zeros(dim)
            raw[4:] = rng.normal(0.0, 1.5, size=dim - 4)
            emb = raw / np.linalg.norm(raw)
            counter += 1
            samples.append(
                AppearanceTrainingSample(
                    sample_id=f"n-{counter:06d}",
                    observed_at=day_start + timedelta(seconds=j * 20),
                    day=cur_day,
                    identity_id=None,
                    label_source=AppearanceLabelSource.NEGATIVE,
                    camera_id="cam-1",
                    orientation=OrientationBin.FRONT,
                    quality=0.8,
                    feature_version="reid-solider",
                    embedding=tuple(float(v) for v in emb),
                )
            )

    # New unseen day where alice wears bob's day-1 shirt and bob wears alice's day-1 shirt.
    x_new_list: list[np.ndarray] = []
    y_new_list: list[int] = []
    for cls_idx, (p_vec, swapped_cloth) in enumerate(
        ((person_a, cloth_b_d1), (person_b, cloth_a_d1))
    ):
        for _ in range(100):
            raw = p_vec + swapped_cloth + rng.normal(0.0, 0.25, size=dim)
            x_new_list.append(raw / np.linalg.norm(raw))
            y_new_list.append(cls_idx)
    x_new = np.asarray(x_new_list)
    y_new = np.asarray(y_new_list)

    # 1. Same-day model trained only on Day 1 memorizes day-1 clothing.
    d1_pos = [
        s for s in samples if s.day == date(2026, 11, 1) and s.identity_id in ("alice", "bob")
    ]
    x_d1 = np.asarray([s.embedding for s in d1_pos])
    y_d1 = np.asarray([0 if s.identity_id == "alice" else 1 for s in d1_pos], dtype=np.int64)
    w_d1, b_d1 = fit_linear_head(x_d1, y_d1, 2, c_value=10.0, seed=42)
    same_day_acc = float(np.mean(np.argmax(x_new @ w_d1.T + b_d1, axis=1) == y_new))

    # 2. Multi-day LODO-CV model trained via train_household_classifier (full default gates).
    res = train_household_classifier(
        samples,
        feature_version="reid-solider",
        association_audit=_PASSING_AUDIT,
        config=TrainerConfig(seed=42),
    )
    w_lodo = np.asarray(res.artifact.weights)
    b_lodo = np.asarray(res.artifact.bias)
    lodo_acc = float(np.mean(np.argmax(x_new @ w_lodo.T + b_lodo, axis=1) == y_new))

    assert same_day_acc < 0.20
    assert lodo_acc > 0.85
    assert lodo_acc > same_day_acc + 0.60


@pytest.mark.parametrize("classes", [("alice", "bob"), ("alice", "bob", "carol")])
def test_trainer_passes_all_gates_for_binary_and_multiclass_and_is_deterministic(
    classes: tuple[str, ...],
) -> None:
    samples = _make_synthetic_dataset(classes=classes, n_days=8, samples_per_class_per_day=75)
    cfg = TrainerConfig(seed=99)

    res_a = train_household_classifier(
        samples,
        feature_version="reid-solider",
        association_audit=_PASSING_AUDIT,
        config=cfg,
        created_at="2026-11-21T00:00:00+00:00",
    )
    res_b = train_household_classifier(
        samples,
        feature_version="reid-solider",
        association_audit=_PASSING_AUDIT,
        config=cfg,
        created_at="2026-11-21T00:00:00+00:00",
    )

    assert json.dumps(res_a.artifact.model_dump(mode="json"), sort_keys=True) == json.dumps(
        res_b.artifact.model_dump(mode="json"), sort_keys=True
    )
    assert res_a.artifact.classes == list(classes)
    assert len(res_a.artifact.eval_days) == 3
    assert not (set(res_a.artifact.trained_days) & set(res_a.artifact.eval_days))
    assert res_a.artifact.metrics.macro_top1_at_0_8 >= 0.85
    assert res_a.artifact.metrics.ece <= 0.05
    assert res_a.artifact.metrics.open_set_false_match_rate_at_0_8 <= 0.02
    assert all(r >= 0.75 for r in res_a.artifact.metrics.per_class_recall_at_0_8.values())

    # Exported artifact loads cleanly into runtime evaluator (no sklearn needed at runtime).
    evaluator = HouseholdClassifierEvaluator.from_artifact(
        res_a.artifact,
        expected_feature_version="reid-solider",
        expected_embedding_dim=24,
    )
    assert evaluator.is_ready
    pred = evaluator.predict_proba(samples[0].embedding)
    assert pred is not None
    cal_probs, _ = pred
    assert sum(cal_probs.values()) == pytest.approx(1.0)


def test_trainer_refuses_single_day_per_class_and_single_class() -> None:
    one_day = _make_synthetic_dataset(n_days=1, samples_per_class_per_day=350)
    with pytest.raises(TrainingGateError, match="distinct positive days"):
        train_household_classifier(
            one_day,
            feature_version="reid-solider",
            association_audit=_PASSING_AUDIT,
        )

    one_class = _make_synthetic_dataset(classes=("alice",), n_days=8, samples_per_class_per_day=80)
    with pytest.raises(TrainingGateError, match="qualifying classes"):
        train_household_classifier(
            one_class,
            feature_version="reid-solider",
            association_audit=_PASSING_AUDIT,
        )


def test_trainer_refuses_failing_association_audit() -> None:
    samples = _make_synthetic_dataset(n_days=8, samples_per_class_per_day=75)
    # 99/100 has Wilson 95% lower bound ~0.945 < 0.99.
    assert wilson_lower_bound(99, 100) < 0.99
    with pytest.raises(TrainingGateError, match="Wilson 95% lower bound"):
        train_household_classifier(
            samples,
            feature_version="reid-solider",
            association_audit={"sample_size": 100, "correct_count": 99},
        )
    with pytest.raises(TrainingGateError, match="below minimum"):
        train_household_classifier(
            samples,
            feature_version="reid-solider",
            association_audit={"sample_size": 50, "correct_count": 50},
        )


def test_open_set_gate_fails_when_negatives_cluster_near_a_class() -> None:
    samples = _make_synthetic_dataset(
        n_days=8,
        samples_per_class_per_day=75,
        negative_near_class0=True,
    )
    with pytest.raises(TrainingGateError, match="open_set_fmr"):
        train_household_classifier(
            samples,
            feature_version="reid-solider",
            association_audit=_PASSING_AUDIT,
        )
